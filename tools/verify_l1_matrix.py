"""Frozen archived quadrature constraints; never advance a physical trajectory."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
from packing_fem.conservative_quadrature import project


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--matrices',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    with np.load(a.matrices) as z:
        w=z['weights'];C=z['conservation'];contexts=[];current=[];trial=[]
        indices=sorted(int(k.split('_')[1]) for k in z.files if k.startswith('context_') and k.endswith('_matrix'))
        positive=[i for i in indices if np.any(z[f'context_{i}_budgets']>0)]
        half=len(positive)//2
        for i in indices:
            c={k:z[f'context_{i}_{k}'] for k in ('matrix','exact','scales')};c.update(energy_budget=0.,force_budget=0.)
            if i in positive:
                # Original run B=2.5e-7; the actual assembled audit uses B/2.
                pose='accepted' if i in positive[:half] else 'trial'
                c.update(energy_budget=1.25e-7,force_budget=1.25e-7,budget_group=pose,global_energy_budget=1.25e-7,global_force_budget=1.25e-7)
                (current if pose=='accepted' else trial).append(c)
            contexts.append(c)
        cases=[]
        for name,block in [('accepted_pose_only',current),('trial_only',trial),('without_trial',[c for c in contexts if c.get('budget_group')!='trial']),('full',contexts)]:
            row=dict(block=name,feasible=False)
            try:
                proposed,report=project(w,C,block,.001,conservation_scales=z['conservation_scales'],reference_weights=z['reference_weights'])
                row.update(feasible=True,weights_unchanged=bool(np.array_equal(w,proposed)),report=report)
            except ValueError as exc:row['reason']=str(exc)
            cases.append(row);print(json.dumps(row),flush=True)
        diagnostics=[]
        moment_rows=6*len(current)
        for name,count in [('energy_and_moments',moment_rows+4),('moments_only',moment_rows)]:
            row=dict(block=name,verified=False,accepted_as_physical_result=False)
            try:
                _,report=project(w,C[-count:],trial,.001,conservation_scales=z['conservation_scales'][-count:],reference_weights=z['reference_weights'])
                row.update(verified=True,report=report)
            except ValueError as exc:row['reason']=str(exc)
            diagnostics.append(row);print(json.dumps(row),flush=True)
    passed=cases[0]['feasible'] and cases[0]['weights_unchanged']
    summary=dict(accepted_rule_admissible=passed,cases=cases,conservation_diagnostics=diagnostics,scope=__doc__,input_sha256=hashlib.sha256(a.matrices.read_bytes()).hexdigest(),full_compaction_completed=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    source=a.output/'source';source.mkdir()
    shutil.copy2('packing_fem/conservative_quadrature.py',source/'conservative_quadrature.py');shutil.copy2(__file__,source/Path(__file__).name)
    if not passed:raise SystemExit(2)


if __name__=='__main__':main()
