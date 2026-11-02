"""Frozen one-step constraint diagnosis; reduced fits are NEVER accepted."""
import argparse,json
from pathlib import Path
import numpy as np
from packing_fem.checkpoint import load
from packing_fem.explicit_solver import solve
import packing_fem.conservative_quadrature as quadrature


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    state,meta=load(a.checkpoint);original=quadrature.project;diagnostics=[]
    # The production algorithm and physical trajectory still see the FULL fit.
    # Only after it fails do separate LPs diagnose individual accuracy blocks.
    def diagnostic(weights,conservation,contexts,*args,**kwargs):
        try:return original(weights,conservation,contexts,*args,**kwargs)
        except ValueError as full_error:
            arrays=dict(weights=weights,conservation=conservation,conservation_scales=kwargs['conservation_scales'],reference_weights=kwargs.get('reference_weights'))
            for i,c in enumerate(contexts):
                for key in ('matrix','exact','scales'):arrays[f'context_{i}_{key}']=c[key]
                arrays[f'context_{i}_budgets']=np.array([c.get('energy_budget',0.),c.get('force_budget',0.)])
            np.savez_compressed(a.output/'constraint-matrices.npz',**arrays)
            count=len({tuple(rule['parent_vertices']) for rule in state['runtime_rules'].values() if 'parent_vertices' in rule})
            current=[c for c in contexts[:-count] if 'source_nodes' in c]
            tolerance=args[0] if args else .001
            differences=[abs(ctx['matrix']@weights-ctx['exact']) for ctx in current]
            energy_excess=sum(max(float(d[0]-tolerance*ctx['scales'][0]),0.) for ctx,d in zip(current,differences))
            force_excess=sum(float(np.maximum(d[1:]-tolerance*ctx['scales'][1:],0.).sum()) for ctx,d in zip(current,differences))
            component_ratios=[]
            for ctx,d in zip(current,differences):
                allowance=tolerance*ctx['scales']+np.r_[ctx.get('energy_budget',0.),np.full(len(d)-1,ctx.get('force_budget',0.)/max(1,len(d)-1))]
                component_ratios.append(float(np.max(d/np.maximum(allowance,1e-30))))
            baseline=dict(accepted_energy_excess_Nmm=energy_excess,accepted_force_excess_L1_N=force_excess,
                energy_limit_Nmm=meta['config'].get('contact_absolute_energy_budget_Nmm',0.)/2,
                force_limit_L1_N=meta['config'].get('contact_absolute_force_budget_N',0.)/2,
                max_uniform_component_allowance_ratio=max(component_ratios),max_half_margin_component_ratio=2*max(component_ratios),
                note='Ratios >1 fail the projection inequalities; L1 totals use the actual global audit.')
            baseline['actual_global_audit_passed']=energy_excess<=baseline['energy_limit_Nmm'] and force_excess<=baseline['force_limit_L1_N']
            (a.output/'accepted-budget-consistency.json').write_text(json.dumps(baseline,indent=2)+'\n')
            blocks=[('full',contexts),('without_next_pose',contexts[:-count]),('accepted_pose_only',current),('next_pose_only',contexts[-count:]),('no_accuracy_constraints',[])]
            for name,block in blocks:
                row=dict(block=name,normal_response_blocks=len(block),samples=len(weights),conservation_rows=len(conservation),accepted_as_physical_result=False)
                try:
                    _,report=original(weights,conservation,block,*args,**kwargs);row.update(feasible=True,report=report)
                except ValueError as exc:row.update(feasible=False,reason=str(exc))
                diagnostics.append(row);print(json.dumps(row),flush=True)
            raise full_error
    quadrature.project=diagnostic
    c=dict(meta['config'],resume_checkpoint=str(a.checkpoint),max_steps=state['steps']+1,inactive_quadrature_replay=False)
    try:result=solve(a.output/'unchanged-full-fit',state['X'],state['t'],c)
    finally:quadrature.project=original
    summary=dict(scope=__doc__,cases=diagnostics,steps=result['steps'],failure=result['failure'],physical_trajectory_advanced=result['steps']!=state['steps'],full_compaction_completed=False)
    if summary['physical_trajectory_advanced']:raise ValueError('Diagnostic unexpectedly advanced the failed full-fit trajectory')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
