"""Hybrid parent-audit / conservative existing-sample redistribution controls."""
from pathlib import Path
import json
import numpy as np
from packing_fem.embedded_contact import EmbeddedContact
from packing_fem.embedded_remesh import refine_parent_rules,transfer
from packing_fem.explicit_periodic import Periodic
from packing_fem.explicit_material import geometry
from packing_fem.surface_remesh import bisect
from tools.verify_plane_contact import fixture


def case(centroid=False,slip=0.,quadrature_order=None,tolerance=.001,factor=2.):
    X,t,_,_,upper=fixture(1,.015,0.);p=Periodic(X,1.);base=EmbeddedContact(X,t,p,20.,.3,.1,thickness=.02)
    B=np.array([[1/3]*3]) if centroid else np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
    weights=np.full(len(B),1/len(B))
    if quadrature_order is not None:
        from packing_fem.triangle_quadrature import RULES
        B,weights=RULES[quadrature_order]
    rules={}
    for i,ids in enumerate(base.faces):
        tri=X[ids];A=np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))/2
        rules[str(i)]=dict(bary=B.tolist(),weights=(A*weights).tolist())
    old=EmbeddedContact(X,t,p,20.,.3,.1,thickness=.02,quadrature_rules=rules,error_budget='normalized',closest_tie_rule='clarke',audit_mode='parent',quadrature_tolerance=tolerance,redistribution_factor=factor)
    for gap in np.linspace(.025,.015,5):
        _,_,x,H,_=fixture(1,float(gap),0.);r=old.evaluate(x,x*0,H,H*0,0.);old.history=r[2];old.accept_geometry(x,H)
    if slip:
        for d in np.linspace(0,slip,5)[1:]:
            _,_,x,H,_=fixture(1,.015,0.);x[upper,0]+=d;H[0,2]+=d
            r=old.evaluate(x,x*0,H,H*0,.0001);old.history=r[2];old.accept_geometry(x,H)
    ids=next(ids for ids in old.faces if np.all(np.isclose(X[ids,2],.6)))
    edge=max((ids[[0,1]],ids[[1,2]],ids[[2,0]]),key=lambda e:np.linalg.norm(X[e[1]]-X[e[0]]))
    V=geometry(X,t)[0];b=np.repeat(np.eye(3)[None],len(t),0)
    proposal=bisect(X,x,t,b,np.zeros(len(t)),V,x*0,edge,1.)
    fresh=EmbeddedContact(proposal['X'],proposal['t'],Periodic(proposal['X'],1.),20.,.3,.1,thickness=.02)
    newrules,mapping,parent=refine_parent_rules(old,fresh.faces,proposal)
    new=EmbeddedContact(proposal['X'],proposal['t'],Periodic(proposal['X'],1.),20.,.3,.1,thickness=.02,quadrature_rules=newrules,error_budget='normalized',closest_tie_rule='clarke',audit_mode='parent',quadrature_tolerance=tolerance,redistribution_factor=factor)
    return old,new,proposal,H,r,mapping,parent



def main():
    import argparse,csv,shutil
    from packing_fem.embedded_remesh import transfer_parent
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for tolerance in (.001,.0005,.00025):
        for slip in (.0001,.01):
            row=dict(tolerance=tolerance,slip_mm=slip,passed=False)
            try:
                old,new,p,H,before,mapping,parent=case(slip=slip,quadrature_order=8,tolerance=tolerance)
                after,report=transfer_parent(old,new,p,H,mapping,parent,before,1e-8,redistribute=True)
                new.accept_geometry(p['x'],H)
                y=p['x'].copy();G=H.copy();upper=p['X'][:,2]>.5;y[upper,0]-=.0001;G[0,2]-=.0001
                later=new.evaluate(y,y*0,G,G*0,.0001)
                row.update(passed=True,transfer=report,continued_coulomb=later[-1]['max_coulomb_ratio'],continued_dissipation_Nmm=later[-1]['friction_increment'],parent_reports={str(k):v['report'] for k,v in new.parent_contexts.items()})
                if tolerance==.001 and slip==.01:
                    np.savez_compressed(a.output/'mesh-and-samples.npz',reference_points=p['X'],points=p['x'],tets=p['t'],samples=new.P@p['x'],sample_weights=new.inner.integration_area0)
            except ValueError as exc:row['reason']=str(exc)
            rows.append(row);print(json.dumps(row),flush=True)
    result=dict(passed=all(r['passed'] for r in rows),scope=__doc__,cases=rows,full_compaction_completed=False)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    fields=['tolerance','slip_mm','passed','redistribution_used','relative_weight_change','work_error','reason']
    with (a.output/'comparison.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=fields);w.writeheader()
        for r in rows:
            t=r.get('transfer',{});w.writerow(dict(tolerance=r['tolerance'],slip_mm=r['slip_mm'],passed=r['passed'],redistribution_used=t.get('redistribution_used'),relative_weight_change=t.get('redistribution',{}).get('relative_area_weight_change'),work_error=t.get('redistribution_work_error'),reason=r.get('reason')))
    source=a.output/'source';source.mkdir()
    for p in Path('packing_fem').glob('*.py'):shutil.copy2(p,source/p.name)
    shutil.copy2(__file__,source/Path(__file__).name)
    if not result['passed']:raise SystemExit(2)


if __name__=='__main__':main()
