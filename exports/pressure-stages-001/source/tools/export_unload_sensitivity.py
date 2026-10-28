"""Archive analytical controls and connected-cell unloading sensitivities."""
import argparse,csv,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.verify_unload_quasistatic import cycle_metrics


def read(root):
    r=json.loads((root/'run/summary.json').read_text());c=json.loads((root/'run/config.json').read_text());b=np.loadtxt(root/'run/accepted-step-audit.csv.gz',delimiter=',',skiprows=1)
    return r,c,b


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--controls',type=Path,required=True);p.add_argument('--resolved-controls',type=Path,required=True);p.add_argument('--connected',type=Path,nargs=3,required=True,help='baseline, half CFL, slow unload');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    controls=json.loads((a.controls/'summary.json').read_text());resolved=json.loads((a.resolved_controls/'summary.json').read_text());plot_cases=[(item,a.controls) for item in controls['cases']]+[(item,a.resolved_controls) for item in resolved['cases']]
    if a.output.exists() or a.output.with_suffix('.zip').exists():raise FileExistsError('Exclusive export required')
    cases=[read(root) for root in a.connected]
    if any(r['failure'] is not None for r,c,b in cases):raise ValueError('Incomplete connected trajectory')
    for r,c,b in cases:
        if len(b)!=r['steps']+1:raise ValueError('Accepted-step count mismatch')
    r0,c0,b0=cases[0];r1,c1,b1=cases[1];r2,c2,b2=cases[2]
    if c1['cfl']!=c0['cfl']/2 or c2['unload_s']!=2*c0['unload_s']:raise ValueError('Expected half CFL and double unload duration')
    for key in c0:
        if key not in ('cfl',) and c0[key]!=c1.get(key):raise ValueError(f'CFL comparison changes {key}')
        if key not in ('unload_s',) and c0[key]!=c2.get(key):raise ValueError(f'Rate comparison changes {key}')
    tangent=Path('exports/layered-contact-transient-001/connected-initial-stiffness/summary.json');S=np.array(json.loads(tangent.read_text())['macro_stiffness']);load=np.array([0.,0.,-c0['pressure_mpa']*4**3,0.,0.,0.]);q=np.linalg.solve(S,load);energy=float(.5*load@q);strain=abs(float(q[2]));rows=[]
    a.output.mkdir(parents=True);shutil.copytree(a.controls,a.output/'analytic-controls');shutil.copytree(a.resolved_controls,a.output/'resolved-negative-controls');shutil.copy2(tangent,a.output/'connected-initial-tangent.json')
    for name,root,(r,c,b) in zip(('baseline','half-cfl','slow-unload'),a.connected,cases):
        shutil.copytree(root,a.output/f'connected-{name}');metric=cycle_metrics(b,c['ramp_s'],c['hold_s'],c['unload_s'],energy)
        rows.append(dict(case=name,cfl=c['cfl'],unload_ms=1000*c['unload_s'],steps=r['steps'],final_axial_compression=r['final_axial_compression'],final_porosity=r['final_porosity'],original_solver_all_passed=all(r['checks'].values()),original_late_KE_ratio=r['kinetic_internal_ratio_loaded'],peak_cycle_KE_over_linear_reference=metric['peak_kinetic_over_reference'],unload_peak_KE_over_linear_reference=metric['unloading']['peak_kinetic_over_reference'],recovery_peak_KE_over_linear_reference=metric['recovery']['peak_kinetic_over_reference'],energy_relative_error=r['energy_relative_error'],static_residual_ratio=r['final_static_residual_ratio'],peak_contacts=int(b[:,10].max()),cycle_metrics=metric))
    times=np.unique(np.r_[b0[:,0],b1[:,0]]);times=times[times<=min(b0[-1,0],b1[-1,0])];start=c0['ramp_s']+c0['hold_s'];phase=np.unique(np.r_[0.,1.,(b0[:,0]-start)/c0['unload_s'],(b2[:,0]-start)/c2['unload_s']]);phase=phase[(phase>=0)&(phase<=1)]
    timestep_axial=float(np.max(abs(np.interp(times,b0[:,0],b0[:,1])-np.interp(times,b1[:,0],b1[:,1])))/strain)
    timestep_porosity=float(np.max(abs(np.interp(times,b0[:,0],b0[:,2])-np.interp(times,b1[:,0],b1[:,2]))))
    rate_axial=float(np.max(abs(np.interp(start+phase*c0['unload_s'],b0[:,0],b0[:,1])-np.interp(start+phase*c2['unload_s'],b2[:,0],b2[:,1])))/strain)
    rate_porosity=float(np.max(abs(np.interp(start+phase*c0['unload_s'],b0[:,0],b0[:,2])-np.interp(start+phase*c2['unload_s'],b2[:,0],b2[:,2]))))
    kinetic_rate_ratio=rows[2]['unload_peak_KE_over_linear_reference']/rows[0]['unload_peak_KE_over_linear_reference']
    comparison=dict(CFL_halving_max_axial_difference_over_linear_peak=timestep_axial,CFL_halving_max_porosity_difference=timestep_porosity,slow_unload_phase_axial_difference_over_linear_peak=rate_axial,slow_unload_phase_porosity_difference=rate_porosity,slow_to_baseline_unload_peak_KE_ratio=kinetic_rate_ratio)
    neg0,neg1=resolved['cases'];nb0=np.loadtxt(a.resolved_controls/neg0['name']/'accepted-step-audit.csv.gz',delimiter=',',skiprows=1);nb1=np.loadtxt(a.resolved_controls/neg1['name']/'accepted-step-audit.csv.gz',delimiter=',',skiprows=1);nt=np.unique(np.r_[nb0[:,0],nb1[:,0]]);nt=nt[nt<=min(nb0[-1,0],nb1[-1,0])]
    negative_comparison=dict(CFL_halving_axial_difference_over_analytic_peak=float(np.max(abs(np.interp(nt,nb0[:,0],nb0[:,1])-np.interp(nt,nb1[:,0],nb1[:,1])))/resolved['analytic_peak_compression']),energy_balance_both_passed=all(item['solver_checks']['energy_balance'] for item in resolved['cases']),peak_pressure_both_reached=all(item['solver_checks']['requested_pressure_reached'] for item in resolved['cases']),both_reject_quasistatic=all(not item['supplementary_pass'] for item in resolved['cases']))
    checks=dict(CFL_axial_difference_below_0_1percent=timestep_axial<.001,rate_phase_difference_below_1percent=rate_axial<.01,slow_unload_KE_reduced_at_least_twofold=kinetic_rate_ratio<.5,all_cycle_KE_below_1percent=all(row['peak_cycle_KE_over_linear_reference']<.01 for row in rows),analytical_controls_expectations_met=controls['benchmark_expectations_met'],resolved_negative_controls=all(negative_comparison[k] for k in ('energy_balance_both_passed','peak_pressure_both_reached','both_reject_quasistatic')) and negative_comparison['CFL_halving_axial_difference_over_analytic_peak']<.001)
    summary=dict(scope='elastic_controls_and_connected_4mm_0.25percent_fixture_not_particle_packing',connected_reference='initial relaxed linear tangent, fixed scale only; not exact finite pressure trajectory',reference_energy_Nmm=energy,reference_axial_compression=strain,comparison_sampling='union of accepted times, or accepted unloading phases, with linear interpolation',comparisons=comparison,resolved_negative_comparison=negative_comparison,supplementary_checks=checks,connected_cases=rows,solver_acceptance_changed=False,plastic_contact_unloading_verified=False,particle_compaction_verified=False)
    (a.output/'verification-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (a.output/'comparison.csv').open('w') as f:
        columns=[k for k in rows[0] if k!='cycle_metrics'];w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows({k:row[k] for k in columns} for row in rows)
    analytic_rows=[dict(case=item['name'],negative_control=item['negative_control'],cfl=item['config']['cfl'],ramp_s=item['config']['ramp_s'],unload_s=item['config']['unload_s'],max_axial_error_over_analytic_peak=item['max_axial_error_over_peak'],unload_max_axial_error_over_analytic_peak=item['unload_max_axial_error_over_peak'],peak_KE_over_analytic_reference=item['metrics']['peak_kinetic_over_reference'],unload_peak_KE_over_analytic_reference=item['metrics']['unloading']['peak_kinetic_over_reference'],original_energy_balance_passed=item['solver_checks']['energy_balance'],original_peak_pressure_reached=item['solver_checks']['requested_pressure_reached'],supplementary_pass=item['supplementary_pass']) for item,root in plot_cases]
    with (a.output/'analytic-comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(analytic_rows[0]));w.writeheader();w.writerows(analytic_rows)
    fig,axs=plt.subplots(2,2,figsize=(12,9))
    for item,control_root in plot_cases:
        data=np.loadtxt(control_root/item['name']/'analytic-comparison.csv.gz',delimiter=',',skiprows=1)
        axs[0,0].plot(data[:,0]/item['config']['ramp_s'],data[:,2],label=item['name']);axs[0,1].semilogy(data[:,0]/item['config']['ramp_s'],np.maximum(data[:,4],1e-12),label=item['name'])
    data=np.loadtxt(a.controls/'baseline/analytic-comparison.csv.gz',delimiter=',',skiprows=1);axs[0,0].plot(data[:,0]/controls['cases'][0]['config']['ramp_s'],data[:,3],'k:',label='quasistatic exact, baseline')
    axs[0,0].set(xlabel='Time / loading ramp duration',ylabel='Axial compression',title='Homogeneous 1 mm cube: true pressure 0.02 MPa')
    axs[0,1].axhline(.01,color='black',linestyle=':');axs[0,1].set(xlabel='Time / loading ramp duration',ylabel='Absolute axial error / exact peak compression',title='Independent finite-strain elastic reference')
    for name,(r,c,b) in zip(('baseline','half CFL','slow unload'),cases):
        axs[1,0].plot((b[:,0]-start)/c['unload_s'],b[:,1],label=name);axs[1,1].semilogy(b[:,0]*1000,np.maximum(b[:,4]/energy,1e-18),label=name)
    axs[1,0].set(xlim=(0,1),xlabel='Unloading phase',ylabel='Axial compression',title='Connected 4 mm fixture, initial porosity 0.25%')
    axs[1,1].axhline(.01,color='black',linestyle=':');axs[1,1].set(xlabel='Time [ms]',ylabel='KE / fixed initial-linear peak-energy scale',title='All steps, including zero pressure')
    for ax in axs.ravel():ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.suptitle('Unloading sensitivity diagnostics; original solver acceptance unchanged');fig.tight_layout();fig.savefig(a.output/'unload-sensitivity.png',dpi=180);plt.close(fig)
    shutil.copy2(__file__,a.output/'export_unload_sensitivity.py');shutil.copy2('tools/verify_unload_quasistatic.py',a.output/'verify_unload_quasistatic.py');shutil.copy2('tests/test_unload_reference.py',a.output/'test_unload_reference.py')
    logs=a.output/'logs';logs.mkdir()
    for name in ('unload-analytic-controls-001.log','unload-reference-tests-001.log','unload-reference-tests-followup-001.log','unload-resolved-negative-controls-001.log','connected-unload-cfl01-001.log','connected-unload-slow-001.log'):
        source=Path('/tmp')/name
        if source.exists():shutil.copy2(source,logs/name)
    (a.output/'README.md').write_text('Unloading analytical controls and connected-cell CFL/rate sensitivity.\nOriginal solver acceptance is unchanged; see verification-summary.json for separate supplementary diagnostics.\nThe rapid negative control is expected to fail quasistatic checks. No plastic/contact unloading or particle compaction validation.\n')
    hashes={str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.output.rglob('*')) if f.is_file()};(a.output/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    with zipfile.ZipFile(a.output.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(a.output.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(a.output.parent))
    with zipfile.ZipFile(a.output.with_suffix('.zip')) as z:
        if z.testzip():raise RuntimeError('ZIP CRC failed')
    print(json.dumps(summary,indent=2));print('ZIP CRC passed')

if __name__=='__main__':main()
