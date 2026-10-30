"""Audit a deliberately bounded contact-onset prefix, not a compaction solution."""
import argparse,json
from pathlib import Path


def assess(run,start_step=90206,end_step=90250):
    run=Path(run);summary=json.loads((run/'summary.json').read_text());attempts=summary.get('attempts',[])
    if not attempts:return dict(passed=False,reason=summary.get('stop'),full_compaction_completed=False)
    last=attempts[-1];directory=run/f"attempt-{last['attempt']:03d}";result=json.loads((directory/'summary.json').read_text())
    rows=[json.loads(line) for line in (directory/'progress.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['step']>start_step]
    required=('closed_boundary_volume','material_volume_consistency','positive_J','physical_porosity','energy_balance','low_kinetic_energy','coulomb_bound','reference_volume_conserved','penetration_bound','contact_numerical_energy_small','volume_remesh_energy_small')
    checks={name:result['checks'].get(name) is True for name in required}
    checks.update(reached_planned_step=result['steps']>=end_step,continuous_accepted_steps=[r['step'] for r in rows]==list(range(start_step+1,end_step+1)),
                  actual_contact=bool(rows) and max(r['contacts'] for r in rows)>0,
                  planned_stop=result.get('failure')=="RuntimeError('Step budget exhausted; no extrapolation')")
    return dict(passed=all(checks.values()),checks=checks,start_step=start_step,end_step=result['steps'],accepted_contact_steps=len(rows),
                final_transient_porosity=result['final_porosity'],failure=result.get('failure'),replays=len(attempts)-1,
                full_compaction_completed=summary['completed'],scope=__doc__,
                limits=['No static-equilibrium acceptance for the ramp endpoint.','No loaded 10% target, unloading or spatial convergence result.'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    result=assess(a.run);(a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    if not result['passed']:raise SystemExit(2)


if __name__=='__main__':main()
