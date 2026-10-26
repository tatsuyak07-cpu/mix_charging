"""Prescribed sticking/sliding tests of persistent material contact integration."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from packing_fem.area_contact import AreaContact
from packing_fem.explicit_material import geometry
from packing_fem.explicit_periodic import Periodic
from packing_fem.surface_remesh import bisect,transfer_contact,audit
from packing_fem.barrier_contact import barrier
from packing_fem.contact_audit import plain,save_capture
from tools.verify_plane_contact import fixture


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for slip in (.0001,.01):
        for fraction in (.2,.5,.8):
            root=args.output/f'slip-{slip}-fraction-{fraction}';root.mkdir()
            X,t,x,H,upper=fixture(2,.015,.25)
            c=json.loads(Path('configs/explicit-prototype.json').read_text());c.update(length_mm=1.,friction=.3)
            V=geometry(X,t)[0];b=np.repeat(np.eye(3)[None],len(t),axis=0);ep=np.linspace(0,.2,len(t))
            make=lambda X,t:AreaContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02,quadrature_mode='material')
            old=make(X,t);old.capture_audit=True;old.history=old.evaluate(x,x*0,H,H*0,0.)[2]
            x=x.copy();H=H.copy();x[upper,0]+=slip;H[0,2]+=slip
            before=old.evaluate(x,x*0,H,H*0,.001);old.history=before[2];old.accept_geometry(x,H)
            ndof=Periodic(X,1.).ndof;save_capture(root/'before-contact.json',old,np.zeros(ndof),np.zeros(ndof),0.,1)
            edges=np.unique(np.sort(np.concatenate([t[:,[i,j]] for i in range(4) for j in range(i+1,4)]),axis=1),axis=0)
            edge=next(e for e in edges if np.all(np.isclose(X[e,2],.6)) and np.linalg.norm(X[e[1]]-X[e[0]])==.5)
            p=bisect(X,x,t,b,ep,V,x*0,edge,1.,fraction=fraction);new=make(p['X'],p['t']);after=transfer_contact(old,new,p,H)
            report=audit((X,x,t,b,ep,V,x*0),p,c,before[-1],after[-1],1e-7,old_response=before,new_response=after)
            new.capture_audit=True;after=new.replay_force_state(p['x'],p['x']*0,H);new.accept_geometry(p['x'],H)
            save_capture(root/'after-contact.json',new,np.zeros(Periodic(p['X'],1.).ndof),np.zeros(Periodic(p['X'],1.).ndof),0.,1)
            np.savez_compressed(root/'quadrature.npz',reference_points=p['X'],points=p['x'],tets=p['t'],area_weights=new.integration_area0)
            restricted=np.zeros_like(x)
            for j in (0,1):np.add.at(restricted,p['supports'][:,j],p['weights'][:,j,None]*after[0])
            normal_energy,traction,_=barrier(.015,20.,.02);spring=min(slip,.3*traction/10.);expected_tangent=.5*10*spring**2
            state_error=max(float(np.max(abs(old.history[k]['plastic']-new.history[k]['plastic']))) for k in old.history)
            checks=dict(transfer=report['accepted'],same_material_histories=set(old.history)==set(new.history),plastic_preserved=state_error<1e-14,
                        reference_areas_identical=bool(np.array_equal(new.integration_area0[:len(X)],old.integration_area0)),no_new_quadrature=not bool(np.any(new.integration_area0[len(X):])),
                        same_force_work=float(np.max(abs(restricted-before[0])))<1e-11,
                        analytic_normal_energy=abs(after[-1]['normal_energy']-normal_energy)<1e-12,
                        analytic_tangent_energy=abs(after[-1]['tangent_energy']-expected_tangent)<1e-12,
                        coulomb=after[-1]['max_coulomb_ratio']<=.3+1e-12)
            result=plain(dict(slip_mm=slip,fraction=fraction,checks=checks,passed=all(checks.values()),audit=report,normal_before=before[-1]['normal_energy'],normal_after=after[-1]['normal_energy'],tangent_before=before[-1]['tangent_energy'],tangent_after=after[-1]['tangent_energy'],maximum_plastic_offset_error_mm=state_error))
            (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');rows.append(result)
    summary=dict(scope=__doc__,passed=all(r['passed'] for r in rows),cases=rows,pressure_solution_verified=False)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (args.output/'energy-comparison.csv').open('w') as stream:
        fields=['slip_mm','fraction','normal_before','normal_after','tangent_before','tangent_after','maximum_plastic_offset_error_mm','passed'];writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in rows)
    (args.output/'verify_material_quadrature.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(dict(passed=summary['passed'],cases=len(rows)),indent=2))
    if not summary['passed']:raise SystemExit(1)


if __name__=='__main__':main()
