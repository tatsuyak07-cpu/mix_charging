"""Archive plastic-cycle references and connected-cell contact/unloading tests."""
import argparse,csv,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from tools.plastic_cycle_audit import unloaded_face_audit


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analytic',type=Path,nargs='+',required=True);p.add_argument('--connected',type=Path,nargs=2,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--failed-analytic',type=Path);p.add_argument('--failure-log',type=Path);a=p.parse_args()
    if a.output.exists() or a.output.with_suffix('.zip').exists():raise FileExistsError('Exclusive output required')
    controls=[]
    for root in a.analytic:
        data=json.loads((root/'summary.json').read_text());controls.extend((item,root) for item in data['cases'])
    cases=[]
    for root in a.connected:
        s=json.loads((root/'summary.json').read_text());r=json.loads((root/'run/summary.json').read_text());c=json.loads((root/'run/config.json').read_text());b=np.loadtxt(root/'run/accepted-step-audit.csv.gz',delimiter=',',skiprows=1);h=np.atleast_1d(np.genfromtxt(root/'run/history.csv',delimiter=',',names=True))
        if len(b)!=r['steps']+1:raise ValueError('Accepted-step integrity failed')
        cases.append((root,s,r,c,b,h))
    c0,c1=cases[0][3],cases[1][3]
    if c1['unload_s']!=2*c0['unload_s']:raise ValueError('Expected doubled unloading time')
    if {k for k in c0.keys()|c1.keys() if c0.get(k)!=c1.get(k)}!={'unload_s'}:raise ValueError('Uncontrolled case differences')
    a.output.mkdir(parents=True)
    for root in a.analytic:shutil.copytree(root,a.output/root.name)
    old=Path('exports/connected-pressure-verification-001/connected-pressure-P01-T32-cfl02-001/run');old_b=np.loadtxt(old/'accepted-step-audit.csv.gz',delimiter=',',skiprows=1);shutil.copytree(old,a.output/'previous-load-only-reference')
    rows=[];independent=[]
    for root,s,r,c,b,h in cases:
        shutil.copytree(root,a.output/root.name)
        with np.load(root/'run/last-state.npz') as z:state={k:z[k] for k in z.files}
        endpoint=unloaded_face_audit(state,c,int(b[-1,10]));independent.append(endpoint);release=c['ramp_s']+c['hold_s'];loading=b[:,0]<=release
        sample=np.unique(np.r_[b[loading,0],old_b[:,0]]);sample=sample[sample<=min(release,b[-1,0],old_b[-1,0])]
        prefix_error=float(np.max(abs(np.interp(sample,b[:,0],b[:,1])-np.interp(sample,old_b[:,0],old_b[:,1]))))
        mean_stress=float(np.max(abs(np.asarray(r['final_work_equivalent_cauchy_stress_MPa']))));shape_error=float(np.max(abs(state['points']-state['reference_points'])))
        rows.append(dict(case=root.name,pressure_mpa=c['pressure_mpa'],ramp_ms=1000*c['ramp_s'],unload_ms=1000*c['unload_s'],cfl=c['cfl'],steps=r['steps'],final_time_ms=1000*b[-1,0],solver_endpoint_pass=s['solver_and_endpoint_checks_passed'],residual_eligible_under_prototype_checks=bool(s['solver_and_endpoint_checks_passed'] and r['checks'].get('unload_completed') and (not endpoint['performed'] or all(endpoint['checks'].values()))),final_command_pressure_MPa=float(b[-1,3]),initial_porosity=float(b[0,2]),minimum_porosity=float(b[:,2].min()),residual_porosity=float(b[-1,2]),maximum_compression=float(b[:,1].max()),last_axial_compression=float(b[-1,1]),reported_residual_compression=r['residual_compression_strain'],final_peak_peeq=float(state['peeq'].max()),peak_contacts=int(b[:,10].max()),final_contacts=int(b[-1,10]),maximum_sampled_Coulomb_ratio=float(h['coulomb_ratio'].max()),friction_work_Nmm=float(h['friction_Nmm'][-1]),late_kinetic_ratio=r['kinetic_internal_ratio_loaded'],whole_loaded_kinetic_ratio=None if r['whole_path_audit']['loaded'] is None else r['whole_path_audit']['loaded']['peak_kinetic_internal_contact_ratio'],energy_relative_error=r['energy_relative_error'],static_residual_ratio=r['final_static_residual_ratio'],maximum_final_work_equivalent_stress_MPa=mean_stress,loaded_prefix_max_axial_difference=prefix_error,maximum_coordinate_return_difference_mm=shape_error,independent_contact_free_force_audit_performed=endpoint['performed'],failure=r['failure']))
        (a.output/root.name/'independent-face-audit.json').write_text(json.dumps(endpoint,indent=2)+'\n')
    b0,b1=cases[0][4],cases[1][4];start=c0['ramp_s']+c0['hold_s'];phase=np.unique(np.r_[0.,1.,(b0[:,0]-start)/c0['unload_s'],(b1[:,0]-start)/c1['unload_s']]);phase=phase[(phase>=0)&(phase<=1)]
    phase_error=float(np.max(abs(np.interp(start+phase*c0['unload_s'],b0[:,0],b0[:,1])-np.interp(start+phase*c1['unload_s'],b1[:,0],b1[:,1]))))
    completed=all(r['checks'].get('unload_completed',False) for root,s,r,c,b,h in cases)
    comparison=dict(both_cycles_completed=completed,both_endpoint_states_passed_native_checks=all(s['solver_and_endpoint_checks_passed'] for root,s,r,c,b,h in cases),residual_axial_absolute_difference=abs(b0[-1,1]-b1[-1,1]),residual_porosity_absolute_difference=abs(b0[-1,2]-b1[-1,2]),unloading_phase_max_axial_difference=phase_error)
    rate_checks=dict(residual_axial_difference_below_1e_4=bool(completed and comparison['both_endpoint_states_passed_native_checks'] and comparison['residual_axial_absolute_difference']<1e-4),residual_porosity_difference_below_1e_7=bool(completed and comparison['both_endpoint_states_passed_native_checks'] and comparison['residual_porosity_absolute_difference']<1e-7))
    summary=dict(scope='connected_4mm_initial_porosity_0.25percent_not_particle_packing',analytic_controls=[dict(case=item['case'],root=str(root),reference_checks=item['reference_checks'],solver_checks=item['solver_checks']) for item,root in controls],connected_cases=rows,rate_comparison=comparison,additional_rate_checks=rate_checks,independent_endpoint_audits=independent,solver_acceptance_changed=False,contact_cycle_CFL_sensitivity_verified=False,mesh_convergence_verified=False,particle_40_to_10_completed=False)
    (a.output/'verification-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (a.output/'connected-comparison.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    analytic_rows=[dict(case=item['case'],ramp_ms=1000*item['config']['ramp_s'],unload_ms=1000*item['config']['unload_s'],cfl=item['config']['cfl'],path_error_over_analytic_peak=item['max_axial_error_over_peak'],residual=item['final_residual_compression'],reference_residual=item['reference_residual_compression'],peeq_relative_error=item['final_peeq_relative_error'],solver_and_reference_pass=all(item['solver_checks'].values()) and all(item['reference_checks'].values())) for item,root in controls]
    with (a.output/'analytic-comparison.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(analytic_rows[0]));w.writeheader();w.writerows(analytic_rows)
    fig,axs=plt.subplots(2,2,figsize=(12,9))
    for item,root in controls:
        data=np.loadtxt(root/item['case']/'analytic-comparison.csv.gz',delimiter=',',skiprows=1);label=f"{item['case']}, CFL {item['config']['cfl']}";axs[0,0].plot(data[:,1],data[:,2],label=label);axs[0,1].semilogy(data[:,0]*1000,np.maximum(data[:,4],1e-12),label=label)
    item,root=controls[0];data=np.loadtxt(root/item['case']/'analytic-comparison.csv.gz',delimiter=',',skiprows=1);axs[0,0].plot(data[:,1],data[:,3],'k:',label='Analytical quasistatic plastic cycle');axs[0,0].set(xlabel='Commanded Cauchy compression [MPa]',ylabel='Axial compression',title='Homogeneous 1 mm cube: permanent deformation')
    axs[0,1].axhline(.01,color='black',linestyle=':');axs[0,1].set(xlabel='Time [ms]',ylabel='Axial error / analytical peak compression',title='Coaxial isotropic-hardening reference')
    for root,s,r,c,b,h in cases:
        label=f"Unload {1000*c['unload_s']:g} ms";axs[1,0].plot(b[:,3],100*b[:,2],label=label);axs[1,0].scatter([b[-1,3]],[100*b[-1,2]],s=20);axs[1,1].plot(b[:,0]*1000,b[:,1],label=label)
    axs[1,0].set(xlabel='Commanded Cauchy compression [MPa]',ylabel='Porosity [%]',title='Connected 4 mm, initial porosity 0.25% (not packing)');axs[1,1].set(xlabel='Time [ms]',ylabel='Axial compression',title='Connected-cell unloading and recovery')
    for ax in axs.ravel():ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.suptitle('Plastic cycle verification: E=10 MPa, trial J2 hardening; no material calibration');fig.tight_layout();fig.savefig(a.output/'plastic-unloading.png',dpi=180);plt.close(fig)
    with np.load(cases[0][0]/'run/initial-mesh.npz') as z:X=z['points'];t0=z['tets']
    meshes=[('Initial',X,t0)]
    for root,s,r,c,b,h in cases:
        with np.load(root/'run/last-state.npz') as z:meshes.append((f"Last accepted, unload {1000*c['unload_s']:g} ms",z['points'],z['tets']))
    lo=np.min([m[1].min(0) for m in meshes],axis=0)-.1;hi=np.max([m[1].max(0) for m in meshes],axis=0)+.1;fig=plt.figure(figsize=(15,5))
    for panel,(name,points,tets) in enumerate(meshes,1):
        edges=np.array(sorted({tuple(sorted((int(t[i]),int(t[j])))) for t in tets for i in range(4) for j in range(i+1,4)}));ax=fig.add_subplot(1,3,panel,projection='3d');ax.add_collection3d(Line3DCollection(points[edges],colors='navy',linewidths=.5,alpha=.7));ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),xlabel='x [mm]',ylabel='y [mm]',zlabel='z [mm]',title=f'{name}: {len(tets)} tetrahedra');ax.set_box_aspect(hi-lo)
    fig.suptitle('Actual connected verification meshes, deformation scale 1; not a sphere packing');fig.tight_layout();fig.savefig(a.output/'residual-mesh.png',dpi=180);plt.close(fig)
    for source in [Path(__file__),Path('tools/plastic_cycle_audit.py'),Path('tools/verify_plastic_cycle_reference.py'),Path('tests/test_plastic_cycle_reference.py')]:shutil.copy2(source,a.output/source.name)
    if a.failed_analytic is not None:
        if a.failure_log is None:raise ValueError('Failure log required for failed-driver archive')
        failed=a.failed_analytic;failure=a.output/'driver-error-record';failure.mkdir();shutil.copy2(failed/'source/tools/verify_plastic_cycle_reference.py',failure/'failed-driver.py');shutil.copy2(failed/'source-sha256.json',failure/'original-source-sha256.json');shutil.copy2(a.failure_log,failure/'failure.log')
        original={str(f.relative_to(failed/'unload8')):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((failed/'unload8').rglob('*')) if f.is_file()};reused=a.output/a.analytic[0].name/'unload8'
        if not all((reused/k).is_file() and hashlib.sha256((reused/k).read_bytes()).hexdigest()==v for k,v in original.items()):raise ValueError('Reused completed cycle mismatch')
        (failure/'reused-original-files-sha256.json').write_text(json.dumps(original,indent=2)+'\n')
    logs=a.output/'logs';logs.mkdir()
    for name in ('plastic-cycle-analytic-002.log','plastic-cycle-slow-loading-001.log','plastic-cycle-tests-001.log','plastic-cycle-tests-followup-001.log','plastic-cycle-tests-final-001.log','connected-plastic-unload8-001.log','connected-plastic-unload16-001.log'):
        f=Path('/tmp')/name
        if f.exists():shutil.copy2(f,logs/name)
    (a.output/'README.md').write_text('Plastic cycle validation controls and a connected 4 mm / 0.25% void fixture.\nSee verification-summary.json for separate original solver, rate comparison, analytical and independent endpoint checks.\nHistory CSV nominal pressure differs from commanded Cauchy pressure. No full contact-cycle CFL or mesh convergence claim.\nThe initial driver-only postprocessing error is preserved; its completed FEM data were reused after config/source/content checks.\n')
    hashes={str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.output.rglob('*')) if f.is_file()};(a.output/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    with zipfile.ZipFile(a.output.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(a.output.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(a.output.parent))
    with zipfile.ZipFile(a.output.with_suffix('.zip')) as z:
        if z.testzip():raise RuntimeError('ZIP CRC failure')
    print(json.dumps(summary,indent=2));print('ZIP CRC passed')

if __name__=='__main__':main()
