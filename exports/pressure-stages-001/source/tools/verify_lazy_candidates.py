"""Verify one RVE onset step against the preserved eager-search result."""
import argparse,json,time
from pathlib import Path
import numpy as np
from packing_fem.checkpoint import load
from packing_fem.explicit_solver import solve


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--rules',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    initial,metadata=load(a.checkpoint);reference,_=load(a.reference)
    c=dict(metadata['config'],resume_checkpoint=str(a.checkpoint),embedded_quadrature_rules=json.loads(a.rules.read_text()),
           inactive_quadrature_replay=True,allow_contact_operator_update=True,checkpoint_stop_after=reference['steps'])
    start=time.perf_counter();result=solve(a.output/'calculation',initial['X'],initial['t'],c)
    actual,_=load(result['checkpoint_paths'][-1]);errors={}
    for key in ('q','v','a','x','b','ep','f','f_int','volume','energy','J','sigma'):
        errors[key]=float(abs(actual[key]-reference[key]).max())
    checks=dict(reached_reference_step=actual['steps']==reference['steps'],
                positions=errors['x']<1e-12,material=errors['b']<1e-12 and errors['ep']<1e-12,
                forces=errors['f']<1e-12 and errors['f_int']<1e-12,
                velocity=bool(np.allclose(actual['v'],reference['v'],rtol=1e-12,atol=1e-12)),
                active_contact=actual['ci']['contacts']>0)
    summary=dict(scope=__doc__,passed=all(checks.values()),checks=checks,maximum_array_differences=errors,
                 wall_s=time.perf_counter()-start,contact_points=actual['ci']['contacts'],
                 note='One accepted onset step; requested stop is deliberate. This is not a pressure-stage equilibrium or timing convergence study.')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
    if not summary['passed']:raise SystemExit(1)


if __name__=='__main__':main()
