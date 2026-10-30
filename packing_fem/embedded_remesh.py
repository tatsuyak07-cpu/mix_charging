"""Exact nested surface refinement with unchanged embedded material samples.

Only retained-node bisections whose pre-existing cubature partitions exactly
into full child-face rules are admissible. No new friction history is invented.
This is not arbitrary surface reconstruction, sample insertion or coarsening.
"""
from copy import deepcopy
import numpy as np
from scipy.sparse import csr_matrix


def coordinates(triangle,points,tolerance):
    E=(triangle[1:]-triangle[0]).T
    uv=(np.asarray(points)-triangle[0])@np.linalg.pinv(E).T
    B=np.c_[1-uv.sum(1),uv]
    if np.max(abs(B@triangle-points))>tolerance or B.min()<-1e-10:
        return None
    return B


def refine_rules(old,new_faces,proposal):
    """Partition existing full-face rules, preserving every point and weight.

    Return construction rules and (old face, old point)->(new face,new point).
    A cut with nonmatching sample mass is rejected, never reweighted.
    """
    X=proposal['X'];tol=old.L*1e-10;n=len(old.X)
    if not np.array_equal(X[:n],old.X):raise ValueError('Embedded refinement must retain original material nodes')
    lookup={tuple(sorted(map(int,face))):i for i,face in enumerate(old.faces)}
    parent={};children={i:[] for i in range(len(old.faces))}
    for j,ids in enumerate(new_faces):
        key=tuple(sorted(map(int,ids)))
        if key in lookup:i=lookup[key]
        else:
            matches=[i for i,face in enumerate(old.faces) if coordinates(old.X[face],X[ids],tol) is not None]
            if len(matches)!=1:raise ValueError('Refined surface lacks a unique material parent face')
            i=matches[0]
        parent[j]=i;children[i].append(j)
    rules={};point_map={}
    for i,child in children.items():
        if not child:raise ValueError('Refinement lost a material source face')
        split=len(child)>1
        original=old.rules.get(str(i))
        if split and original is None:raise ValueError('Active embedded split requires a pre-existing partition-aligned quadrature rule')
        if original is None:continue
        if split and len(original.get('partition',[])):
            raise ValueError('Splitting retained integration partitions requires explicit cell-intersection migration')
        B=np.asarray(original['bary']);weights=np.asarray(original['weights']);points=B@old.X[old.faces[i]]
        assignment={j:[] for j in child};representations={}
        for k,point in enumerate(points):
            for j in sorted(child):
                bary=coordinates(X[new_faces[j]],point[None],tol)
                if bary is not None:
                    assignment[j].append(k);representations[(j,k)]=bary[0];break
            else:raise ValueError('A retained integration point has no refined source face')
        for j,indices in assignment.items():
            tri=X[new_faces[j]];area=np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))/2
            if not indices or abs(weights[indices].sum()-area)>max(1e-12,area*1e-10):
                raise ValueError('Existing sample weights do not partition child-face area; active reweighting prohibited')
            rule=deepcopy(original)
            C=coordinates(old.X[old.faces[i]],X[new_faces[j]],tol)
            for name in ('atlas','atlas_controls'):
                for pose in rule.get(name,[]):pose['source']=(C@np.asarray(pose['source'])).tolist()
            if 'partition' in rule:rule['partition']=(np.asarray(rule['partition'])@np.linalg.inv(C)).tolist()
            rule.update(bary=[representations[j,k].tolist() for k in indices],weights=weights[indices].tolist())
            if split:
                # Retain EVERY historical/control pose in child coordinates,
                # then independently audit its inherited, unchanged samples.
                from . import surface_patch
                from .contact_atlas import density,scales
                audits=[]
                for name in ('atlas','atlas_controls'):
                    for k,pose in enumerate(rule.get(name,[])):
                        reference=surface_patch.integrate(np.asarray(pose['source']),np.asarray(pose['targets']),area,
                            kn=.5*old.kn,thickness=old.thickness,tolerance=old.quadrature_tolerance/16,
                            vertex_map=np.asarray(pose['mapping'],int),groups=pose['groups'],pointwise=True,error_budget='normalized',tie_rule=pose.get('tie_rule','stop'))['value']
                        applied=np.asarray(rule['weights'])@density(np.asarray(rule['bary']),pose,.5*old.kn,old.thickness)
                        error=abs(applied-reference)/scales(reference)
                        audits.append(dict(kind=name,index=k,energy_error=float(error[0]),force_error=float(error[1:].max())))
                        if error.max()>old.quadrature_tolerance:raise ValueError('Inherited child cubature fails retained-pose audit; no reweighting/history reset')
                rule['refinement_pose_audits']=audits
            rules[str(j)]=rule
            for local,k in enumerate(indices):point_map[i,k]=(j,local)
    return rules,point_map,parent


def transfer(old,new,proposal,H,point_map,parent,old_response,tolerance):
    """Transactional transfer of committed AND applied lagged friction states."""
    n=len(old.X);N=len(new.X);mapping=np.arange(len(old.inner.X))
    for face,(nodes,_,_) in old.face_points.items():
        for k,node in enumerate(nodes):
            j,local=point_map[face,k];mapping[node]=new.face_points[j][0][local]
    if len(np.unique(mapping))!=len(mapping):raise ValueError('Material sample correspondence is not one-to-one')
    xold=old.P@proposal['x'][:n];xnew=new.P@proposal['x']
    if not np.allclose(new.inner.X[mapping],old.inner.X,rtol=0,atol=old.L*1e-10) or not np.allclose(xnew[mapping],xold,rtol=0,atol=old.L*1e-10):
        raise ValueError('Refinement moved a retained material sample')
    before=old.inner.integration_area0;after=new.inner.integration_area0
    if not np.allclose(after[mapping],before,rtol=1e-10,atol=1e-12) or np.any(abs(after[n:N])>1e-12):
        raise ValueError('Refinement changed material integration weights')
    # This evaluates independent normal-integration/collision guards as well as
    # the new patch identities. It is not committed and never initializes old
    # slipping histories with these zero-state histories.
    fresh=new.evaluate(proposal['x'],proposal['x']*0,H,H*0,0.)[2]
    inverse={int(value):i for i,value in enumerate(mapping)}
    def migrate(history):
        mapped={};covered=set()
        for key,state in fresh.items():
            node=inverse.get(key[1])
            if node is None:raise ValueError('Refinement created an active material sample')
            matches=[]
            for oldkey,value in history.items():
                if oldkey[1]!=node:continue
                # A patch may move to an adjacent parent face, but creation,
                # merging and disconnected targets are not transfer operations.
                if old.inner.adjacent_targets(value['face'],value['shift'],parent[state['face']],state['shift']):matches.append((oldkey,value))
            if len(matches)!=1 or matches[0][0] in covered:raise ValueError('Active material patch correspondence is not one-to-one')
            oldkey,value=matches[0];kept=deepcopy(value)
            ref=value['bary']@(old.inner.X[value['anchor_ids']]+old.L*np.asarray(value['anchor_shift']))
            cur=value['bary']@(xold[value['anchor_ids']]+H@np.asarray(value['anchor_shift']))
            anchor_faces={i for i,ids in enumerate(old.faces) if set(map(int,ids))==set(map(int,value['anchor_ids']))}
            anchors=[]
            for j,face in enumerate(new.faces):
                if parent[j] not in anchor_faces:continue
                bary=coordinates(new.X[face]+old.L*np.asarray(value['anchor_shift']),ref[None],old.L*1e-10)
                if bary is not None and np.max(abs(bary[0]@(proposal['x'][face]+H@np.asarray(value['anchor_shift']))-cur))<old.L*1e-10:
                    anchors.append((face,bary[0]))
            if not anchors:raise ValueError('Retained friction anchor lacks an exact child-face representation')
            ids,bary=anchors[0]
            kept.update(peer=int(mapping[value['peer']]),second=int(mapping[value['second']]),anchor_ids=ids.copy(),bary=bary.copy(),face=state['face'],shift=np.asarray(state['shift']).copy())
            mapped[key]=kept;covered.add(oldkey)
        if covered!=set(history):raise ValueError('Refinement lost active material history')
        return mapped
    committed=migrate(old.history)
    applied=old.inner.accepted_force_history
    if applied is None:raise ValueError('Active transfer requires an accepted force-evaluation state')
    new.history=committed;new.inner.accepted_force_history=migrate(applied)
    evaluated=new.replay_force_state(proposal['x'],proposal['x']*0,H)
    # Compare work on every original periodic DOF, including cell-image work.
    rows=np.repeat(np.arange(N),2);cols=proposal['supports'].ravel();values=proposal['weights'].ravel()
    R=csr_matrix((values,(rows,cols)),shape=(N,n))
    f0=old.periodic.force(old_response[0],old_response[1]);f1=old.periodic.force(np.asarray(R.T@evaluated[0]),evaluated[1])
    scale=max(float(np.linalg.norm(f0)),1e-16);force_error=float(np.linalg.norm(f1-f0))/scale
    report=dict(retained_samples=len(mapping),active_histories=len(committed),new_material_samples=0,
                periodic_virtual_work_relative_error=force_error,history_interpolation=False)
    if force_error>tolerance:raise ValueError('Refinement changes periodic contact virtual work: '+str(report))
    if abs(evaluated[-1]['friction_increment'])>1e-12 or abs(evaluated[-1]['released_history_energy'])>1e-12:
        raise ValueError('Refinement invents frictional dissipation or releases history')
    return evaluated,report
