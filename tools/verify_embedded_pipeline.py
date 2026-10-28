"""Prescribed friction replay and active-contact checkpoint verification."""
import argparse,json,time
from pathlib import Path
import numpy as np
from packing_fem.embedded_contact import EmbeddedContact
from packing_fem.explicit_periodic import Periodic
from packing_fem.area_contact import SurfaceResolutionError
from packing_fem.checkpoint import save,load,contact_state,restore_contact
from packing_fem.barrier_contact import barrier
from packing_fem.contact_audit import plain
from tools.verify_plane_contact import fixture


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',type=float,default=0.);p.add_argument('--slip',type=float,default=.01)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);rules={};start=time.perf_counter();attempts=[]
    for attempt in range(12):
        X,t,_,_,upper=fixture(1,.015,a.phase)
        c=EmbeddedContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02,quadrature_rules=rules)
        print(json.dumps(dict(attempt=attempt,virtual_points=c.P.shape[0]-len(X),elapsed_s=time.perf_counter()-start)),flush=True)
        try:
            trace=[]
            for gap in np.linspace(.025,.015,9):
                _,_,x,H,_=fixture(1,float(gap),a.phase)
                value=c.evaluate(x,x*0,H,H*0,0.);c.history=value[2];c.accept_geometry(x,H)
                trace.append(dict(gap_mm=gap,slip_mm=0.,normal_energy=value[-1]['normal_energy'],tangent_energy=value[-1]['tangent_energy']))
                print(json.dumps(dict(attempt=attempt,gap_mm=gap,step='normal',elapsed_s=time.perf_counter()-start)),flush=True)
            original=x.copy();cell=H.copy()
            for slip in np.linspace(0,a.slip,9)[1:]:
                x=original.copy();x[upper,0]+=slip;H=cell.copy();H[0,2]+=slip
                value=c.evaluate(x,x*0,H,H*0,.001/8);c.history=value[2];c.accept_geometry(x,H)
                trace.append(dict(gap_mm=.015,slip_mm=slip,normal_energy=value[-1]['normal_energy'],tangent_energy=value[-1]['tangent_energy']))
                print(json.dumps(dict(attempt=attempt,slip_mm=slip,step='shear',elapsed_s=time.perf_counter()-start)),flush=True)
            break
        except SurfaceResolutionError as error:
            feature=getattr(error,'feature',None)
            if not feature or not feature.get('requires_quadrature_refinement'):raise
            requests=feature['rules'];rules.update(requests)
            record=dict(attempt=attempt,requests={key:dict(points=len(rule['weights']),force_error=rule['normal_force_relative_error']) for key,rule in requests.items()})
            attempts.append(record);print(json.dumps(record),flush=True)
            (a.output/f'rules-{attempt:03d}.json').write_text(json.dumps(rules)+'\n')
    else:raise RuntimeError('Prescribed replay budget exhausted')
    save(a.output/'active-contact-checkpoint',dict(contact=contact_state(c),X=X,t=t,x=x,H=H,rules=rules),dict(scope=__doc__))
    state,_=load(a.output/'active-contact-checkpoint')
    other=EmbeddedContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02,quadrature_rules=state['rules']);restore_contact(other,state['contact'])
    replay=other.replay_force_state(x,x*0,H)
    error=float(np.max(abs(replay[0]-value[0])))
    E,F,_=barrier(.015,20.,.02);spring=min(a.slip,.3*F/10.)
    checks=dict(active_checkpoint_replays_force=error<1e-12,normal_energy=abs(value[-1]['normal_energy']-E)<1e-8,
                analytic_tangent=abs(value[-1]['tangent_energy']-5*spring**2)<max(1e-12,5*spring**2*2e-3),
                coulomb=value[-1]['max_coulomb_ratio']<=.3+1e-12)
    summary=plain(dict(checks=checks,passed=all(checks.values()),attempts=attempts,force_replay_error_N=error,trace=trace,wall_s=time.perf_counter()-start))
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
    if not summary['passed']:raise SystemExit(1)


if __name__=='__main__':main()
