"""Compare audited same-mesh solves before attributing differences to refinement."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np


def read_study(path):
    path=Path(path);report=json.loads((path/'assessment.json').read_text())
    inputs=json.loads((path/'inputs.json').read_text())
    if not report['screen_completed']:raise ValueError('Mesh screen did not complete')
    records=[]
    for case,item in zip(report['cases'],inputs['cases'],strict=True):
        if hashlib.sha256(Path(item['mesh']).read_bytes()).hexdigest()!=item['mesh_sha256']:
            raise ValueError('Input mesh no longer matches its recorded digest')
        run=Path(case['run']);audit=json.loads((run/'audit.json').read_text())
        with (run/'history.csv').open() as stream:row=list(csv.DictReader(stream))[-1]
        checkpoint=run/f"state-{int(row['step']):04d}.npz"
        final=audit['states'][-1];config=json.loads((run/'config.json').read_text())
        if not (case['passed'] and audit['passed'] and
                audit['original_physics_config_sha256']==inputs['physics_config_sha256'] and
                final.get('state_sha256')==hashlib.sha256(checkpoint.read_bytes()).hexdigest()):
            raise ValueError('Independent audit is missing or not bound to the actual state')
        pressure=float(row['nominal_pressure_MPa']);phi=float(row['porosity'])
        if (abs(pressure-final['pressure_MPa'])>1e-10 or
            abs(phi-final['matrix_region_porosity'])>1e-10):raise ValueError('History differs from replay')
        if (pressure!=case['pressure_MPa'] or phi!=case['porosity'] or
            float(row['compression'])!=case['compression']):raise ValueError('Screen summary differs from actual history')
        target=config.get('implicit_lbfgs_solve_tolerance_N',config['implicit_force_tolerance_N'])
        if config['implicit_optimizer']!='lagged_lbfgs':raise ValueError('Numerical target is specific to lagged L-BFGS')
        if not 0<target<=config['implicit_force_tolerance_N']:raise ValueError('Invalid accuracy target')
        if max(final[k] for k in ['free_residual_N','rigid_force_max_N','rigid_torque_over_radius_max_N'])>=target:
            raise ValueError('Saved state did not meet the declared numerical accuracy')
        with np.load(checkpoint) as state:
            plastic_fraction=float(state['V0'][state['ep']>1e-12].sum()/state['V0'].sum())
            mean_ep=float(state['V0']@state['ep']/state['V0'].sum())
            maximum_ep=float(state['ep'].max())
        records.append(dict(level=case['level'],tetrahedra=case['input_tetrahedra'],
            compression=float(row['compression']),pressure_MPa=pressure,porosity=phi,
            initial_porosity=audit['states'][0]['matrix_region_porosity'],
            numerical_target_N=target,free_residual_N=final['free_residual_N'],
            rigid_force_max_N=final['rigid_force_max_N'],
            rigid_torque_over_radius_max_N=final['rigid_torque_over_radius_max_N'],
            reference_volume_plastic_fraction=plastic_fraction,reference_volume_mean_ep=mean_ep,
            maximum_ep=maximum_ep,
            sparse_solver=config.get('implicit_sparse_solver','superlu'),
            inner_force_target_N=target*config.get('implicit_lag_inner_tolerance_fraction',1.),
            state_sha256=final['state_sha256'],mesh_sha256=item['mesh_sha256'],
            physical_sha256=inputs['physics_config_sha256']))
    return report,records


def compare(baseline,strict,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    old,first=read_study(baseline);new,second=read_study(strict)
    if len(first)!=len(second) or len(second)<3:raise ValueError('Need matching three-level families')
    if any(r['compression']!=second[0]['compression'] for r in second):raise ValueError('Need a common compression across meshes')
    rows=[]
    for a,b in zip(first,second,strict=True):
        if any(a[k]!=b[k] for k in ['level','tetrahedra','mesh_sha256','physical_sha256','compression']):
            raise ValueError('Cannot isolate solver accuracy when mesh, physics or strain differ')
        if b['numerical_target_N']>=a['numerical_target_N']:raise ValueError('Second solve must be stricter')
        rows.append(dict(level=b['level'],tetrahedra=b['tetrahedra'],compression=b['compression'],
            baseline_pressure_MPa=a['pressure_MPa'],strict_pressure_MPa=b['pressure_MPa'],
            pressure_change_relative=abs(b['pressure_MPa']-a['pressure_MPa'])/abs(b['pressure_MPa']),
            baseline_porosity=a['porosity'],strict_porosity=b['porosity'],
            porosity_change_absolute=abs(b['porosity']-a['porosity']),
            initial_porosity=b['initial_porosity'],
            compression_porosity_drop=b['initial_porosity']-b['porosity'],
            baseline_free_residual_N=a['free_residual_N'],strict_free_residual_N=b['free_residual_N'],
            strict_rigid_force_max_N=b['rigid_force_max_N'],
            strict_rigid_torque_over_radius_max_N=b['rigid_torque_over_radius_max_N'],
            reference_volume_plastic_fraction=b['reference_volume_plastic_fraction'],
            reference_volume_mean_ep=b['reference_volume_mean_ep']))
    report=dict(scope=f"Initial{100*second[0]['compression']:g}% only; observed nonlinear-solve sensitivity, not an error bound or full-path convergence proof.",
        baseline=str(baseline),strict=str(strict),same_mesh_physics_and_strain_verified=True,
        independent_original_force_audits_passed=True,
        maximum_observed_pressure_solve_change_relative=max(r['pressure_change_relative'] for r in rows),
        numerical_settings=[dict(level=a['level'],baseline_outer_target_N=a['numerical_target_N'],strict_outer_target_N=b['numerical_target_N'],baseline_inner_target_N=a['inner_force_target_N'],strict_inner_target_N=b['inner_force_target_N'],baseline_sparse_solver=a['sparse_solver'],strict_sparse_solver=b['sparse_solver']) for a,b in zip(first,second,strict=True)],
        interpretation='Observed combined numerical-search sensitivity. Outer/inner solve tolerances tightened and independently benchmarked equivalent sparse backend changed; not a one-parameter error bound.',
        original_mesh_screen=old,strict_mesh_screen=new,
        full_loaded_path_mesh_convergence_verified=False,rows=rows)
    (out/'assessment.json').write_text(json.dumps(report,indent=2)+'\n')
    with (out/'comparison.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    counts=[r['tetrahedra'] for r in rows]
    axes[0].plot(counts,[r['baseline_pressure_MPa'] for r in rows],'o--',label='Original solve: 0.005 N')
    axes[0].plot(counts,[r['strict_pressure_MPa'] for r in rows],'s-',label=f"Strict solve: {second[0]['numerical_target_N']:g} N")
    axes[0].set(xlabel='T4 elements',ylabel='Nominal pressure (MPa)',title=f"Axial compression {100*second[0]['compression']:g}%");axes[0].legend()
    axes[1].plot(counts,[100*r['initial_porosity'] for r in rows],'o--',label='Initial geometry')
    axes[1].plot(counts,[100*r['strict_porosity'] for r in rows],'s-',label='Compressed geometry')
    axes[1].set(xlabel='T4 elements',ylabel='Matrix-region porosity (%)',title='Fixed transverse periodic cell');axes[1].legend()
    for ax in axes:
        ax.set_xticks(counts,[f'{count:,}' for count in counts]);ax.grid(alpha=.25)
    fig.savefig(out/'mesh-accuracy.png',dpi=180);plt.close(fig)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--strict',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();compare(a.baseline,a.strict,a.out)
