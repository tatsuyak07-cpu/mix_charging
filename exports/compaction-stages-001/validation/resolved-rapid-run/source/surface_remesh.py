"""Conforming periodic edge bisection with transactional state transfer.

This refines existing planar facets, not curved-surface reconstruction or a
quality-repair/coarsening algorithm. Failed contact/energy audits reject the
whole proposal. Reference and current configurations are both retained.
"""
from copy import deepcopy
import numpy as np
from .explicit_material import geometry,oriented,response
from .adaptive_tet import boundary,quality
from .area_contact import frame


def closest_edge_candidates(X,x,t,b,ep,V0,velocity,feature,L):
    """Rank the two trial closest-point insertions by accepted-state quality.

    Geometry alone does not authorize a commit. Callers must audit each
    candidate's state, contact history and energy before accepting it.
    """
    from .contact_tracking import closest_segments
    ids=np.asarray(feature['ids'],int);points=np.asarray(feature['points'],float)
    if feature['kind']!='ee' or ids.shape!=(4,) or points.shape!=(4,3) or not np.all(np.isfinite(points)):
        raise ValueError('A finite EE feature is required')
    hit=closest_segments(*points);candidates=[]
    for side,key in enumerate(('s','t')):
        edge=ids[2*side:2*side+2];fraction=float(hit[key])
        row=dict(side=side,edge=edge.tolist(),fraction=fraction,geometrically_valid=False)
        try:
            p=bisect(X,x,t,b,ep,V0,velocity,edge,L,fraction=fraction)
            row.update(geometrically_valid=True,minimum_quality=p['report']['minimum_quality_after'])
        except ValueError as exc:row['reason']=str(exc)
        candidates.append(row)
    return sorted(candidates,key=lambda r:(not r['geometrically_valid'],-r.get('minimum_quality',0.),tuple(r['edge']),r['fraction']))


def bisect(X,x,t,b,ep,V0,velocity,edge,L,fraction=.5):
    """Split all incident tets/copies at (1-fraction)*edge[0]+fraction*edge[1].

    The default retains midpoint splitting. Off-centre proposals require the
    same external momentum, energy, friction and quality audits as midpoints.
    """
    if len(np.unique(X,axis=0))!=len(X):raise ValueError('Coincident distinct nodes require explicit body correspondence')
    if not np.isfinite(fraction) or not 0<fraction<1:raise ValueError("Interior split fraction required")
    edge=np.asarray(edge,int)
    if edge.shape!=(2,) or edge[0]==edge[1]:raise ValueError('Two distinct edge nodes required')
    edges=np.unique(np.sort(np.concatenate([t[:,[i,j]] for i in range(4) for j in range(i+1,4)]),axis=1),axis=0)
    if not np.any(np.all(edges==np.sort(edge),axis=1)):raise ValueError('Selected edge not in mesh')
    # Match translated endpoint pairs, never identify distinct coincident bodies.
    a,bpoint=X[edge];matches=[]
    for pair in edges:
        for ordered in (pair,pair[::-1]):
            shift=(X[ordered[0]]-a)/L
            if np.allclose(shift,np.rint(shift),atol=1e-10,rtol=0) and np.allclose(X[ordered[1]]-bpoint,L*np.rint(shift),atol=L*1e-10,rtol=0):
                matches.append(pair);break
    # One bisection per tet in a transaction, so no hanging nodes or ordering ambiguity.
    incident=[np.flatnonzero(np.sum(np.isin(t,pair),axis=1)==2) for pair in matches]
    flat=np.concatenate(incident)
    if len(np.unique(flat))!=len(flat):raise ValueError('Periodic edge images share a tetrahedron; refine a different edge')
    supports=[(int(i),int(i)) for i in range(len(X))];weights=[(.5,.5)]*len(X);rows=[];parents=[];removed=set(map(int,flat))
    for pair,owners in zip(matches,incident):
        midpoint=len(supports);supports.append(tuple(map(int,pair)))
        f=fraction if (X[pair[1]]-X[pair[0]])@(bpoint-a)>0 else 1-fraction
        weights.append((1-f,f))
        for owner in owners:
            rest=[int(i) for i in t[owner] if i not in pair]
            rows.extend([[int(pair[0]),midpoint,*rest],[midpoint,int(pair[1]),*rest]]);parents.extend([int(owner)]*2)
    keep=[i for i in range(len(t)) if i not in removed];parents=np.r_[keep,parents].astype(int)
    supports=np.asarray(supports);weights=np.asarray(weights)
    def interpolate(value):
        shape=(len(weights),)+(1,)*(value.ndim-1)
        return weights[:,0].reshape(shape)*value[supports[:,0]]+weights[:,1].reshape(shape)*value[supports[:,1]]
    X2=interpolate(X);x2=interpolate(x);t2=oriented(X2,np.r_[t[keep],np.asarray(rows,int)])
    ref=geometry(X2,t2)[0];refold=geometry(X,t)[0];fractions=ref/refold[parents]
    V2=V0[parents]*fractions
    geometry(x2,t2)
    return dict(X=X2,x=x2,t=t2,b=b[parents].copy(),ep=ep[parents].copy(),V0=V2,velocity=interpolate(velocity),supports=supports,weights=weights,parents=parents,
                report=dict(operation='periodic_edge_bisection',fraction=float(fraction),periodic_edges=len(matches),new_nodes=len(X2)-len(X),reference_volume_error=float(abs(V2.sum()-V0.sum())),current_volume_error=float(abs(geometry(x2,t2)[0].sum()-geometry(x,t)[0].sum())),minimum_quality_before=float(quality(x,t).min()),minimum_quality_after=float(quality(x2,t2).min())))


def transfer_contact(old,new,proposal,H):
    """Interpolate elastic slip, not plastic offsets, then audit externally.

New active quadrature nodes require matching histories at both edge endpoints.
Unknown patch creation, loss, split, or ambiguous interpolation is rejected.
Old anchor nodes survive bisection and remain material points.
"""
    x=proposal['x'];new.capture_audit=False
    baseline=new.evaluate(x,x*0,H,H*0,0.);fresh=baseline[2];mapped={};covered=set()
    for key,state in fresh.items():
        node=key[1];sources=proposal['supports'][node];states=[]
        weights=proposal['weights'][node] if 'weights' in proposal else np.array([.5,.5])
        for source in sources:
            matches=[]
            for oldkey,value in old.history.items():
                if oldkey[1]!=source:continue
                # Reference target facets must overlap in area, not just at a vertex.
                oldtri=old.X[old.faces[value['face']]]+old.L*np.asarray(value['shift'])
                newtri=new.X[new.faces[state['face']]]+new.L*np.asarray(state['shift'])
                D=np.column_stack((oldtri[1]-oldtri[0],oldtri[2]-oldtri[0]));w=np.linalg.lstsq(D,(newtri-oldtri[0]).T,rcond=None)[0].T
                projected=oldtri[0]+w@D.T
                coplanar=np.max(abs(projected-newtri))<old.L*1e-9
                contained=np.all(w>=-1e-9) and np.all(w.sum(1)<=1+1e-9)
                adjacent=np.min(np.linalg.norm(oldtri[:,None]-newtri[None,:],axis=2))<old.L*1e-9
                if coplanar and (contained or adjacent):matches.append((oldkey,value))
            if len(matches)!=1:raise ValueError('Remesh contact history lacks a unique parent patch')
            matched_key,matched_state=matches[0]
            if node<len(old.X):covered.add(matched_key)
            states.append(matched_state)
        if node<len(old.X):
            # Preserve the material anchor and its strain derivative exactly.
            state=dict(deepcopy(states[0]),face=state['face'],shift=state['shift'])
        else:
            def anchor(value,reference):
                z=old.X if reference else x
                cell=np.eye(3)*old.L if reference else H
                return value['bary']@(z[value['anchor_ids']]+cell@value['anchor_shift'])
            reference_anchor=sum(w*anchor(v,True) for w,v in zip(weights,states))
            current_anchor=sum(w*anchor(v,False) for w,v in zip(weights,states))
            represented=None
            # An interpolated material anchor must have one exact retained
            # surface representation in BOTH configurations. No nearest-point
            # re-anchoring: that would change elastic work under affine strain.
            shifts=set(tuple(map(int,v['anchor_shift'])) for v in states)
            for shift in sorted(shifts):
                ref=old.X[old.faces]+old.L*np.array(shift)
                for face,tri in enumerate(ref):
                    D=np.column_stack((tri[1]-tri[0],tri[2]-tri[0]))
                    w=np.linalg.lstsq(D,reference_anchor-tri[0],rcond=None)[0]
                    bary=np.r_[1-w.sum(),w]
                    if bary.min()<-1e-9 or np.linalg.norm(bary@tri-reference_anchor)>old.L*1e-9:continue
                    target=x[old.faces[face]]+H@np.array(shift)
                    if np.linalg.norm(bary@target-current_anchor)>old.L*1e-9:continue
                    represented=(old.faces[face].copy(),np.array(shift),bary);break
                if represented is not None:break
            if represented is None:raise ValueError('Interpolated friction anchor has no exact material representation')
            def world(value,source):
                z=np.zeros((9,3));z[0]=x[source];z[4]=x[value['peer']];z[8]=x[value['second']]
                return frame(z,value['director'])@value['spring']
            world_spring=sum(w*world(value,int(source)) for w,source,value in zip(weights,sources,states))
            state=deepcopy(state);z=np.zeros((9,3));z[0]=x[node];z[4]=x[state['peer']];z[8]=x[state['second']]
            T=frame(z,state['director']);spring=T.T@world_spring
            if abs(np.linalg.norm(T@spring)-np.linalg.norm(world_spring))>1e-9:raise ValueError('Remesh friction normal mismatch')
            anchor_ids,anchor_shift,bary=represented
            coordinate=T.T@(x[node]-current_anchor)
            state.update(anchor_ids=anchor_ids,anchor_shift=anchor_shift,bary=bary,coordinate=coordinate,spring=spring,plastic=coordinate-spring)
        mapped[key]=state
    # Losing active history without accounting for its energy is forbidden.
    if covered!=set(old.history):raise ValueError('Remesh changes active source histories')
    new.history=mapped
    evaluated=new.evaluate(x,x*0,H,H*0,0.)
    new.history=evaluated[2]
    return evaluated


def audit(old,proposal,c,old_contact_report,new_contact_report,tolerance=1e-6,old_response=None,new_response=None):
    """Return explicit acceptance flags; caller commits only if every flag passes."""
    X,x,t,b,ep,V0,velocity=old
    def metrics(t,V,v,z):
        mass=np.zeros(len(z));np.add.at(mass,t.ravel(),np.repeat(V*c['density_tonne_mm3']/4,4))
        return mass.sum(),(mass[:,None]*v).sum(0),np.cross(z,mass[:,None]*v).sum(0),float(.5*np.sum(mass[:,None]*v*v))
    a=metrics(t,V0,velocity,x);p=proposal;d=metrics(p['t'],p['V0'],p['velocity'],p['x'])
    E0=float(V0@response(b,ep,c)[1]);E1=float(p['V0']@response(p['b'],p['ep'],c)[1]);scale=max(abs(E0),a[3],c['young_mpa']*V0.sum()*1e-12)
    energy=lambda r:r['normal_energy']+r['tangent_energy']
    pscale=max(np.sqrt(2*a[0]*scale),1e-30);L=c['length_mm']
    errors=dict(mass=abs(d[0]-a[0])/a[0],linear_momentum=float(np.linalg.norm(d[1]-a[1])/pscale),angular_momentum=float(np.linalg.norm(d[2]-a[2])/(pscale*L)),kinetic_energy=abs(d[3]-a[3])/scale,material_energy=abs(E1-E0)/scale,contact_energy=abs(energy(new_contact_report)-energy(old_contact_report))/max(abs(energy(old_contact_report)),scale*1e-8))
    for name in ('normal_energy','tangent_energy'):
        errors[name]=float(abs(new_contact_report[name]-old_contact_report[name])/max(abs(old_contact_report[name]),scale*1e-8))
    if old_response is not None and new_response is not None:
        from .explicit_periodic import affine_basis
        macro=lambda z,r:np.einsum('aij,ai->j',affine_basis(z),r[0])+r[1]
        force0=macro(X,old_response);force1=macro(p['X'],new_response)
        errors['contact_macro_force']=float(np.linalg.norm(force1-force0)/max(np.linalg.norm(force0),c['young_mpa']*L**3*1e-12))
    checks={name:bool(value<=tolerance) for name,value in errors.items()};checks['coulomb']=new_contact_report['max_coulomb_ratio']<=c['friction']+1e-10
    return dict(accepted=all(checks.values()),checks=checks,scaled_errors=errors,tolerance=tolerance)
