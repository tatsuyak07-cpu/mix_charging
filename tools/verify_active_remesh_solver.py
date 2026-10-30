"""Bounded dynamic transaction test; not a quasi-static compaction result."""
import argparse,json
from pathlib import Path
import numpy as np
from packing_fem.embedded_contact import EmbeddedContact
from packing_fem.embedded_remesh import coordinates
from packing_fem.explicit_periodic import Periodic
from packing_fem.explicit_material import geometry
from packing_fem.explicit_solver import solve
from packing_fem.surface_remesh import bisect
from packing_fem.triangle_quadrature import RULES
from packing_fem.area_contact import SurfaceResolutionError
from tools.verify_plane_contact import fixture


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--remesh-step',type=int,default=1);parser.add_argument('--cfl',type=float,default=.2);parser.add_argument('--velocity-transfer',choices=['l2','momentum_energy'],default='momentum_energy');parser.add_argument('--tie-rule',choices=['stop','clarke'],default='clarke');a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    _,t,X,_,_=fixture(1,.015,0.);L=1.;H=np.eye(3)
    b=np.repeat(np.eye(3)[None],len(t),0);V=geometry(X,t)[0]
    base=EmbeddedContact(X,t,Periodic(X,L),20.,.3,.1,thickness=.02,error_budget='normalized',closest_tie_rule=a.tie_rule)
    face=next(face for face in base.faces if np.all(np.isclose(X[face,2],.415)))
    edges=[face[[i,j]] for i,j in ((0,1),(1,2),(2,0))];edge=max(edges,key=lambda e:np.linalg.norm(X[e[1]]-X[e[0]]))
    p=bisect(X,X,t,b,np.zeros(len(t)),V,X*0,edge,L)
    child=EmbeddedContact(p['X'],p['t'],Periodic(p['X'],L),20.,.3,.1,thickness=.02,error_budget='normalized',closest_tie_rule=a.tie_rule)
    child_rules={};B,w=RULES[8]
    for j,ids in enumerate(child.faces):
        tri=p['X'][ids];A=np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))/2
        child_rules[str(j)]=dict(bary=B.tolist(),weights=(A*w).tolist())
    for replay in range(12):
        c=EmbeddedContact(p['X'],p['t'],Periodic(p['X'],L),20.,.3,.1,thickness=.02,error_budget='normalized',quadrature_rules=child_rules,closest_tie_rule=a.tie_rule)
        try:c.evaluate(p['X'],p['X']*0,H,H*0,0.);break
        except SurfaceResolutionError as exc:
            if not getattr(exc,'feature',{}).get('rules'):raise
            child_rules.update(exc.feature['rules'])
    else:raise RuntimeError('Initial normal-integration preparation failed')
    rules={}
    for i,face in enumerate(base.faces):
        points=[];weights=[]
        for j,ids in enumerate(child.faces):
            C=coordinates(X[face],p['X'][ids],1e-10)
            if C is not None:
                points.extend(np.asarray(child_rules[str(j)]['bary'])@C);weights.extend(child_rules[str(j)]['weights'])
        rules[str(i)]=dict(bary=np.asarray(points).tolist(),weights=weights)
    c=json.loads(Path('configs/explicit-prototype.json').read_text())
    c.update(length_mm=L,contact_backend='embedded_area',surface_error_budget='normalized',embedded_quadrature_rules=rules,
             compression_fraction=0.,ramp_s=1e-5,hold_s=0.,max_steps=3,output_every=1,remesh_every=0,surface_remesh_steps=[a.remesh_step],surface_remesh_edge=edge.tolist(),surface_remesh_max_events=1,
             active_embedded_surface_remesh=True,surface_velocity_transfer=a.velocity_transfer,closest_tie_rule=a.tie_rule,cfl=a.cfl,save_checkpoint=True,save_contact_audit=True)
    results={}
    for name,flag in (('enabled',True),('disabled',False)):
        result=solve(a.output/name,X,t,dict(c,active_embedded_surface_remesh=flag))
        results[name]=dict(failure=result['failure'],steps=result['steps'],events=result['remesh_events'])
    stopped=solve(a.output/'checkpoint-stop',X,t,dict(c,checkpoint_stop_after=1))
    resumed=solve(a.output/'checkpoint-resume',X,t,dict(c,resume_checkpoint=stopped['checkpoint_paths'][-1]))
    with np.load(a.output/'enabled/last-state.npz') as whole,np.load(a.output/'checkpoint-resume/last-state.npz') as continued:
        restart_exact=all(np.array_equal(whole[k],continued[k]) for k in whole.files)
    # The disabled transaction must leave exactly the no-remesh trajectory.
    baseline=solve(a.output/'baseline',X,t,dict(c,surface_remesh_steps=[],active_embedded_surface_remesh=False))
    with np.load(a.output/'disabled/last-state.npz') as z,np.load(a.output/'baseline/last-state.npz') as y:
        unchanged=all(np.array_equal(z[k],y[k]) for k in z.files)
    enabled=results['enabled']['events'];disabled=results['disabled']['events']
    checks=dict(active_restart_exact=restart_exact,active_transfer_committed=bool(enabled) and enabled[0]['accepted'] and 'retained_embedded_samples' in enabled[0],
                opt_in_required=bool(disabled) and not disabled[0]['accepted'],rejected_trajectory_unchanged=unchanged,
                bounded_three_steps=all(r['steps']==3 for r in results.values()))
    summary=dict(passed=all(checks.values()),checks=checks,scope=__doc__,results=results,full_compaction_completed=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
    if not summary['passed']:raise SystemExit(2)


if __name__=='__main__':main()
