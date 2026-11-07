"""Check the full self-consistent Coulomb contact Jacobian on a saved FE mesh."""
import argparse,json
from pathlib import Path
import numpy as np
from packing_fem.explicit_periodic import Periodic
from packing_fem.ipc_periodic import PeriodicIPC


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--current',required=True);p.add_argument('--previous',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    c=json.loads(Path(a.config).read_text())
    with np.load(a.current) as stream:current=dict(stream)
    with np.load(a.previous) as stream:previous=dict(stream)
    if any(not np.array_equal(current[key],previous[key]) for key in ['X','t']):raise ValueError('Contact verification requires matching periodic charts')
    periodic=Periodic(current['X'],c['length_mm']);contact=PeriodicIPC(current['X'],current['t'],periodic,c)
    q=current['q'];old=previous['q'];_,g,J,n=contact.evaluate_coupled(q,old,True)
    contact.set_lag(q,old);_,lagged_g,_,_=contact.evaluate(q)
    force_error=float(np.max(abs(g-lagged_g)));direction=np.zeros(periodic.ndof)
    direction[:periodic.nmicro]=np.random.default_rng(886).normal(size=periodic.nmicro)/np.sqrt(periodic.nmicro)
    checks=[]
    for h in [1e-5,1e-6]:
        plus=contact.evaluate_coupled(q+h*direction,old)[1];minus=contact.evaluate_coupled(q-h*direction,old)[1]
        fd=((plus-minus)/(2*h))[:periodic.nmicro];error=(J@direction)[:periodic.nmicro]-fd
        checks.append(dict(h=h,relative_l2_error=float(np.linalg.norm(error)/max(np.linalg.norm(fd),1e-30)),max_absolute_error=float(np.max(abs(error)))))
    report=dict(current=a.current,previous=a.previous,contacts=n,force_definition_max_difference_N=force_error,checks=checks,passed=bool(force_error<1e-8 and checks[-1]['relative_l2_error']<1e-3))
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
