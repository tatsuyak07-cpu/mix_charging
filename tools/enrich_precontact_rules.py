"""Add positive material samples ONLY before contact; require full bulk replay."""
import argparse,hashlib,json
from copy import deepcopy
from pathlib import Path
import numpy as np
from packing_fem.checkpoint import load
from packing_fem.triangle_quadrature import gauss_rule
from packing_fem import surface_patch


def enriched(rules,level,state):
    if not state.get('contact_free_prefix',False) or state.get('contact',{}).get('history'):
        raise ValueError('New material samples require certified contact-free replay; active histories cannot be initialized')
    if level not in (0,1,2):raise ValueError('Supported composite refinement levels are 0, 1, 2')
    patches=[np.eye(3)]
    for _ in range(level):
        refined=[]
        for a,b,c in patches:
            ab=(a+b)/2;bc=(b+c)/2;ca=(c+a)/2
            refined.extend([np.array([a,ab,ca]),np.array([ab,b,bc]),np.array([ca,bc,c]),np.array([ab,bc,ca])])
        patches=refined
    B,w=gauss_rule(4);dense=np.concatenate([B@patch for patch in patches]);dw=np.tile(w/len(patches),len(patches))
    retained_fraction=1/(1+4**level);result=deepcopy(rules);reports=[]
    moments=lambda b:np.array([np.ones(len(b)),b[:,0],b[:,1],b[:,0]**2,b[:,0]*b[:,1],b[:,1]**2])
    for face,rule in result.items():
        if 'parent_vertices' in rule or rule.get('partition'):raise ValueError('Enrichment needs original unsplit reference faces')
        oldB=np.asarray(rule['bary']);oldw=np.asarray(rule['weights']);area=float(oldw.sum())
        if area<=0 or not np.isfinite(oldw).all() or np.any(oldw<=0):raise ValueError('Positive old material rule required')
        points=np.r_[oldB,dense];weights=np.r_[retained_fraction*oldw,(1-retained_fraction)*area*dw]
        target=area*np.array([1.,1/3,1/3,1/6,1/12,1/6]);error=float(max(abs(moments(points)@weights-target))/area)
        if error>1e-10:raise ValueError('Enriched rule loses reference moments')
        rule.update(bary=points.tolist(),weights=weights.tolist())
        reports.append(dict(face=face,old_samples=len(oldB),new_samples=len(dense),total_samples=len(points),retained_fraction=retained_fraction,moment_error=error))
    return result,dict(level=level,composite_triangles=len(patches),gauss_order=4,faces=reports,new_samples=sum(r['new_samples'] for r in reports),requires_full_coupled_replay=True,active_history_initialization=False)


def frozen_support_poses(rules,state,config,stopped):
    """Geometry-only projection of the old stop; NEVER transfer its histories."""
    from packing_fem.embedded_contact import EmbeddedContact
    from packing_fem.explicit_periodic import Periodic,deformation
    X=state['X'];L=config['length_mm'];x=stopped['x'][:len(X)];H=deformation(stopped['q'][-6:])*L
    c=EmbeddedContact(X,state['t'],Periodic(X,L),config['contact_penalty_mpa_mm'],config['friction'],config['contact_skin_mm'],quadrature_rules=rules,error_budget='normalized',closest_tie_rule='clarke')
    c.inner.guard.accepted_geometry=(c.P@x,H.copy());c.inner.guard.accepted_shifts=((0,0,0),);c.inner.guard.prepare_geometry(c.P@x,H);c._x=x;c._H=H;c._candidates(list(map(int,rules)))
    poses={}
    edge=config.get('surface_remesh_edge',[]);midpoint=None
    if len(edge)==2:
        midref=(X[edge[0]]+X[edge[1]])/2;index=int(np.argmin(np.linalg.norm(stopped['X']-midref,axis=1)))
        if np.linalg.norm(stopped['X'][index]-midref)<L*1e-10:midpoint=stopped['x'][index]
    for face in map(int,rules):
        keys=c._face_candidates[face]
        identities=[(int(c.periodic.classes[n]),*np.rint((X[n]+L*np.asarray(s))/(L*1e-10)).astype(np.int64)) for j,s in keys for n in c.faces[j]]
        unique=sorted(set(identities));mapping=np.array([unique.index(v) for v in identities]).reshape(-1,3)
        groups=surface_patch.components(np.array([c.faces[j] for j,_ in keys]),[s for _,s in keys],X,L)
        target_list=[];map_list=[];children={};mid_ids={};next_id=len(unique)
        for old,(j,shift) in enumerate(keys):
            ids=c.faces[j];tri=x[ids]+H@shift;children[old]=[]
            if midpoint is not None and all(v in ids for v in edge):
                i,k=[int(np.flatnonzero(ids==v)[0]) for v in edge];other=next(v for v in range(3) if v not in (i,k));key=tuple(shift)
                if key not in mid_ids:mid_ids[key]=next_id;next_id+=1
                mid=midpoint+H@shift;m=mid_ids[key]
                for points,vertices in [([tri[i],mid,tri[other]],[mapping[old,i],m,mapping[old,other]]),([mid,tri[k],tri[other]],[m,mapping[old,k],mapping[old,other]])]:
                    children[old].append(len(target_list));target_list.append(points);map_list.append(vertices)
            else:children[old].append(len(target_list));target_list.append(tri);map_list.append(mapping[old])
        remapped=[[j for old in group for j in children[old]] for group in groups]
        pose=dict(source=x[c.faces[face]].tolist(),targets=np.asarray(target_list).tolist(),mapping=np.asarray(map_list).tolist(),groups=remapped,tie_rule='clarke')
        ids=c.faces[face]
        if midpoint is not None and all(v in ids for v in edge):
            i,k=[int(np.flatnonzero(ids==v)[0]) for v in edge];other=next(v for v in range(3) if v not in (i,k))
            pose['child_sources']=[[x[ids[i]].tolist(),midpoint.tolist(),x[ids[other]].tolist()],[midpoint.tolist(),x[ids[k]].tolist(),x[ids[other]].tolist()]]
            I=np.eye(3);mid=(I[i]+I[k])/2;pose['source_cells']=np.array([[I[i],mid,I[other]],[mid,I[k],I[other]]]).tolist()
        poses[str(face)]=pose
    return poses


def adaptive_enriched(rules,level,state,config,support_poses=None):
    # Validate the same certified-free prerequisite before examining poses.
    enriched({},level,state)
    from packing_fem.area_contact import AreaContact
    from packing_fem.explicit_periodic import Periodic
    X=state['X'];t=state['t'];L=config['length_mm']
    base=AreaContact(X,t,Periodic(X,L),config['contact_penalty_mpa_mm'],config['friction'],config['contact_skin_mm'])
    result=deepcopy(rules);reports=[];order=4 if level==0 else 8;B,w=gauss_rule(order)
    tolerance=config.get('surface_quadrature_tolerance',.001);fraction=tolerance/(16*4**level)
    for face,rule in result.items():
        if 'parent_vertices' in rule or rule.get('partition'):raise ValueError('Adaptive enrichment requires unsplit reference rules')
        ids=base.faces[int(face)];seed=np.eye(3)[None];edge=config.get('surface_remesh_edge',[])
        if len(edge)==2 and all(v in ids for v in edge):
            i,j=[int(np.flatnonzero(ids==v)[0]) for v in edge];k=next(k for k in range(3) if k not in (i,j));I=np.eye(3);mid=(I[i]+I[j])/2;seed=np.array([[I[i],mid,I[k]],[mid,I[j],I[k]]])
        oldB=np.asarray(rule['bary']);oldw=np.asarray(rule['weights']);area=float(oldw.sum());partition=seed
        poses=list(rule.get('atlas',[]))+list(rule.get('atlas_controls',[]))
        if support_poses is not None:poses.append(support_poses[face])
        if not poses:raise ValueError('Retained contact poses required for adaptive enrichment')
        for pose in poses:
            reference=surface_patch.integrate(np.asarray(pose['source']),np.asarray(pose['targets']),area,kn=.5*config['contact_penalty_mpa_mm'],thickness=config.get('barrier_thickness_mm',.02),tolerance=tolerance/(8*2**level),vertex_map=np.asarray(pose['mapping']),groups=pose['groups'],pointwise=True,error_budget='normalized',initial_partition=partition,compress_rule=False,tie_rule=pose.get('tie_rule','stop'))
            partition=reference['partition']
            print(json.dumps(dict(face=face,reference_cells=len(partition),stage='support_layout_only')),flush=True)
        points=[];weights=[]
        for cell in partition:
            relative=abs(np.linalg.det((cell[1:,:2]-cell[0,:2])))
            points.extend(B@cell);weights.extend(area*relative*w)
        points=np.asarray(points);weights=np.asarray(weights)
        if len(points)>config.get('atlas_max_candidate_points',160000):raise ValueError('Adaptive material point budget exceeded')
        rule.update(bary=np.r_[oldB,points].tolist(),weights=np.r_[fraction*oldw,(1-fraction)*weights].tolist())
        audit=surface_patch.validate_partition(partition)
        b=np.asarray(rule['bary']);v=np.asarray(rule['weights']);moment=np.array([np.ones(len(b)),b[:,0],b[:,1],b[:,0]**2,b[:,0]*b[:,1],b[:,1]**2])@v
        error=float(np.max(abs(moment/area-[1,1/3,1/3,1/6,1/12,1/6])))
        if error>1e-10:raise ValueError('Adaptive enrichment loses material moments')
        reports.append(dict(face=face,old_samples=len(oldB),new_samples=len(points),total_samples=len(b),partition_cells=len(partition),old_mass_fraction=fraction,moment_error=error,retained_poses=len(poses)))
    return result,dict(method='uncompressed_common_pose_partition',level=level,gauss_order=order,faces=reports,new_samples=sum(r['new_samples'] for r in reports),requires_full_coupled_replay=True,active_history_initialization=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--rules',type=Path,required=True);p.add_argument('--level',type=int,required=True);p.add_argument('--adaptive',action='store_true');p.add_argument('--config',type=Path);p.add_argument('--pose-checkpoint',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    state,_=load(a.checkpoint);original=json.loads(a.rules.read_text())
    config=json.loads(a.config.read_text()) if a.config else {}
    support=frozen_support_poses(original,state,config,load(a.pose_checkpoint)[0]) if a.pose_checkpoint else None
    rules,report=adaptive_enriched(original,a.level,state,config,support) if a.adaptive else enriched(original,a.level,state)
    report['support_geometry_only']=str(a.pose_checkpoint) if a.pose_checkpoint else None
    report['input_rules_sha256']=hashlib.sha256(a.rules.read_bytes()).hexdigest()
    (a.output/'rules.json').write_text(json.dumps(rules)+'\n');(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
