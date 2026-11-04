"""Bounded counterexample-guided parent quadrature replay from a free prefix.

Never install new samples in an active state. Every attempt replays the FULL
coupled interval, and every failed attempt and source snapshot is retained.
"""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
from packing_fem.checkpoint import load


def require_free_prefix(state):
    if not state.get('contact_free_prefix') or state.get('contact',{}).get('history'):
        raise ValueError('Parent replay must begin before ALL contact history')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--resume',type=Path,required=True);p.add_argument('--seed-rules',type=Path,required=True);p.add_argument('--initial-rules',type=Path);p.add_argument('--control-checkpoint',type=Path,action='append',default=[]);p.add_argument('--output',type=Path,required=True);p.add_argument('--level',type=int,default=1);p.add_argument('--max-attempts',type=int,default=2);p.add_argument('--required-time',type=float,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    state,_=load(a.resume)
    require_free_prefix(state)
    if a.max_attempts<1 or a.required_time<=state['time']:raise ValueError('Positive bounded replay interval required')
    rules=a.initial_rules;controls=list(a.control_checkpoint);attempts=[];stop=None;passed=False
    for index in range(a.max_attempts):
        if rules is None:
            destination=a.output/f'fit-{index:03d}'
            cmd=[sys.executable,'tools/fit_replay_rules.py','--checkpoint',str(a.resume),'--rules',str(a.seed_rules),'--config',str(a.config),'--level',str(a.level),'--rebuild','--include-trial','--output',str(destination)]
            for path in controls:cmd.extend(['--pose-checkpoint',str(path)])
            with (a.output/f'fit-{index:03d}.log').open('x') as log:fit=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            if fit.returncode:stop='Positive replay-fit generation failed; no physical state accepted';break
            rules=destination/'rules.json'
        destination=a.output/f'run-{index:03d}'
        cmd=[sys.executable,'tools/run_porosity_cycle.py','--config',str(a.config),'--output',str(destination),'--resume',str(a.resume),'--rules',str(rules),'--allow-contact-operator-update','--max-replays','0']
        with (a.output/f'run-{index:03d}.log').open('x') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        path=destination/'summary.json'
        if not path.exists():stop='Coupled runner failed before summary; inspect retained log';break
        cycle=json.loads(path.read_text())
        if not cycle['attempts']:stop=cycle['stop'];break
        directory=destination/f"attempt-{cycle['attempts'][-1]['attempt']:03d}";result=json.loads((directory/'summary.json').read_text())
        last,_=load(result['checkpoint_paths'][-1]);gates=('closed_boundary_volume','material_volume_consistency','positive_J','physical_porosity','energy_balance','low_kinetic_energy','coulomb_bound','reference_volume_conserved','penetration_bound','contact_numerical_energy_small','volume_remesh_energy_small')
        remesh=any(e.get('accepted') and e.get('operation')=='periodic_edge_bisection' for e in result['remesh_events'])
        passed=last['time']>=a.required_time and remesh and all(result['checks'].get(k,False) for k in gates)
        record=dict(index=index,rules=str(rules),rules_sha256=hashlib.sha256(rules.read_bytes()).hexdigest(),steps=result['steps'],time_s=last['time'],porosity=result['final_porosity'],required_time_reached=last['time']>=a.required_time,short_interval_passed=passed,checks=result['checks'],failure=result['failure'],checkpoint=result['checkpoint_paths'][-1])
        attempts.append(record);(a.output/'attempts.json').write_text(json.dumps(attempts,indent=2)+'\n');print(json.dumps(record),flush=True)
        if passed:stop='Requested bounded interval passed; no equilibrium/full-compaction claim';break
        if (result.get('failure_feature') or {}).get('kind')!='parent_quadrature':stop='Failure requires a different repair; no blind face-ID replay';break
        controls.append(Path(result['checkpoint_paths'][-1]));rules=None
    else:stop='Counterexample replay attempt budget exhausted'
    summary=dict(scope=__doc__,attempts=attempts,stop=stop,short_interval_passed=passed,required_time_s=a.required_time,free_checkpoint=str(a.resume),control_checkpoints=list(map(str,controls)),history_zeroed_in_active_state=False,full_compaction_completed=False,convergence_verified=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if not passed:raise SystemExit(2)


if __name__=='__main__':main()
