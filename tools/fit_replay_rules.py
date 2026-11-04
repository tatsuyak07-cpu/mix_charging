"""Positive enriched cubature fitted before contact, including planned child DOFs.

Reference geometry controls choose quadrature only. Every new history is later
obtained from a full coupled replay; no frozen-stop history is transferred.
"""
import argparse,hashlib,json,shutil
from pathlib import Path
from copy import deepcopy
import numpy as np
from packing_fem import surface_patch as patch
from packing_fem.checkpoint import load
from packing_fem.area_contact import AreaContact
from packing_fem.explicit_periodic import Periodic
from packing_fem.porosity_cycle import PorosityCycle
from packing_fem.contact_atlas import refine_weights,scales
from scipy.optimize import linprog
from scipy.sparse import csr_matrix,hstack,vstack
from tools.enrich_precontact_rules import enriched,frozen_support_poses


def fit(rules,state,config,controls,level,rebuild=False):
    enriched({},level,state)
    X=state['X'];L=config['length_mm'];kn=.5*config['contact_penalty_mpa_mm'];thickness=config.get('barrier_thickness_mm',.02);tol=config.get('surface_quadrature_tolerance',.001)
    base=AreaContact(X,state['t'],Periodic(X,L),2*kn,config['friction'],config['contact_skin_mm']);result=deepcopy(rules);reports=[]
    for face,rule in result.items():
        ids=base.faces[int(face)];seed=np.eye(3)[None];edge=config.get('surface_remesh_edge',[])
        if len(edge)==2 and all(v in ids for v in edge):
            i,j=[int(np.flatnonzero(ids==v)[0]) for v in edge];k=next(k for k in range(3) if k not in (i,j));I=np.eye(3);mid=(I[i]+I[j])/2;seed=np.array([[I[i],mid,I[k]],[mid,I[j],I[k]]])
        oldB=np.asarray(rule['bary']);oldw=np.asarray(rule['weights']);area=float(oldw.sum());added=controls[face] if isinstance(controls[face],list) else [controls[face]];poses=list(rule.get('atlas',[]))+list(rule.get('atlas_controls',[]))+added
        cases=[];pool=[oldB]
        for pose in poses:
            for cell_index,cell in enumerate(seed):
                fraction=abs(np.linalg.det(cell[1:,:2]-cell[0,:2]));source=np.asarray(pose['child_sources'][cell_index]) if 'child_sources' in pose else cell@np.asarray(pose['source']);targets=np.asarray(pose['targets']);mapping=np.asarray(pose['mapping'])
                r=patch.integrate(source,targets,area*fraction,kn=kn,thickness=thickness,tolerance=tol/(8*2**level),vertex_map=mapping,groups=pose['groups'],pointwise=True,error_budget='normalized',tie_rule=pose.get('tie_rule','stop'))
                pool.append(r['bary']@cell);cases.append(dict(cell=cell,source=source,pose=pose,reference=r['value']))
        B=np.unique(np.concatenate(pool),axis=0);n=len(B);rows=6+sum(len(c['reference']) for c in cases)
        if n>config.get('atlas_max_candidate_points',160000) or n*rows>config.get('atlas_max_matrix_entries',48000000):raise ValueError('Replay-fit candidate matrix budget exceeded')
        M=[np.ones(n),B[:,0],B[:,1],B[:,0]**2,B[:,0]*B[:,1],B[:,1]**2];rhs=[1.,1/3,1/3,1/6,1/12,1/6]
        assignments=[];claimed=np.zeros(n,bool)
        for cell in seed:
            local=B@np.linalg.inv(cell);inside=np.all(local>=-1e-12,axis=1)&~claimed;claimed|=inside;assignments.append((inside,local))
        if not claimed.all():raise ValueError('Replay samples lose child support')
        for index,c in enumerate(cases):
            pose=c['pose'];inside,local=assignments[index%len(seed)];field=np.zeros((n,len(c['reference'])))
            field[inside]=patch.grouped_field(local[inside],c['source'],np.asarray(pose['targets']),pose['groups'],kn,thickness,np.asarray(pose['mapping']),True,pose.get('tie_rule','stop'))[0]
            scale=scales(c['reference']);M.extend((area*field/scale).T);rhs.extend(c['reference']/scale)
        M=np.asarray(M);rhs=np.asarray(rhs);floor=np.zeros(n);lookup={tuple(b):i for i,b in enumerate(B)};retained=0. if rebuild else tol/(16*4**level)
        for b,w in zip(oldB,oldw):floor[lookup[tuple(b)]]+=retained*w/area
        residual=rhs-M@floor
        # Exactly equivalent positive variables y=column_scale*z keep the
        # very small corner-panel masses out of the LP's coefficient range.
        column_scale=1/np.maximum(np.max(abs(M),axis=0),1.)
        P=M*column_scale;response=csr_matrix(P[6:]);allowance=tol/(4*2**level)
        options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10,'time_limit':60.}
        solution=linprog(np.zeros(n),A_eq=csr_matrix(P[:6]),b_eq=residual[:6],A_ub=vstack([response,-response]),b_ub=np.r_[residual[6:]+allowance,-residual[6:]+allowance],bounds=(0,None),method='highs-ipm',options=options)
        if solution.status in (1,4):
            solution=linprog(np.zeros(n),A_eq=csr_matrix(P[:6]),b_eq=residual[:6],A_ub=vstack([response,-response]),b_ub=np.r_[residual[6:]+allowance,-residual[6:]+allowance],bounds=(0,None),method='highs-ds',options=dict(options,presolve=False))
        if not solution.success:raise ValueError('Positive replay fit infeasible: '+solution.message)
        positive=refine_weights(M,residual,column_scale*np.maximum(solution.x,0.))+floor
        error=float(max(abs(M[6:]@positive-rhs[6:]),default=0.));moment=float(max(abs(M[:6]@positive-rhs[:6])))
        if error>tol/2 or moment>1e-10:raise ValueError(f'Replay cubature independent fit audit failed: response={error}, moment={moment}')
        keep=positive>0;rule.update(bary=B[keep].tolist(),weights=(area*positive[keep]).tolist(),atlas_controls=list(rule.get('atlas_controls',[]))+deepcopy(added))
        report=dict(face=face,candidates=n,old_samples=len(oldB),total_samples=int(keep.sum()),retained_fraction=retained,training_cases=len(cases),response_error=error,moment_error=moment,child_source_basis=True)
        reports.append(report);print(json.dumps(report),flush=True)
    return result,dict(method='positive_fit_with_child_source_basis',level=level,faces=reports,requires_coupled_replay=True,history_transfer_from_stop=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--pose-checkpoint',type=Path,action='append',required=True);p.add_argument('--rules',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--level',type=int,required=True);p.add_argument('--rebuild',action='store_true',help='Allow zero old-point mass ONLY at the certified inactive prefix');p.add_argument('--include-trial',action='store_true',help='Also train the rejected Verlet trial geometry, without committing history');p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    state,_=load(a.checkpoint);config=json.loads(a.config.read_text());rules=json.loads(a.rules.read_text());controls={f:[] for f in rules}
    for path in a.pose_checkpoint:
        saved,metadata=load(path);current=frozen_support_poses(rules,state,config,saved)
        for face in rules:controls[face].append(current[face])
        summary_path=path.parent/'summary.json'
        if a.include_trial and summary_path.exists():
            summary=json.loads(summary_path.read_text());events=[e for e in summary.get('surface_resolution_attempts',[]) if 'trial_dt_s' in e and abs(e['accepted_time_s']-saved['time'])<1e-12]
            if events:
                dt=events[-1]['trial_dt_s'];q=saved['q']+dt*saved['v']+.5*dt*dt*saved['a'];periodic=Periodic(saved['X'],config['length_mm'],mass_frame=config.get('mass_frame','pinned'))
                if saved['prescribed_active']:q[periodic.prescribed]=PorosityCycle(metadata['config'],saved['porosity_cycle_state']).displacement(saved['time']+dt)[0]
                trial=dict(saved,q=q,x=periodic.positions(q));current=frozen_support_poses(rules,state,config,trial)
                for face in rules:
                    current[face].update(control_kind='rejected_trial_geometry',control_time_s=saved['time']+dt)
                    controls[face].append(current[face])
    updated,report=fit(rules,state,config,controls,a.level,rebuild=a.rebuild)
    report.update(input_rules_sha256=hashlib.sha256(a.rules.read_bytes()).hexdigest(),pose_checkpoints=list(map(str,a.pose_checkpoint)),rejected_trials_included=a.include_trial,rebuild_at_free_prefix=a.rebuild)
    (a.output/'rules.json').write_text(json.dumps(updated)+'\n');(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    source=a.output/'source';source.mkdir();shutil.copy2(__file__,source/Path(__file__).name)


if __name__=='__main__':main()
