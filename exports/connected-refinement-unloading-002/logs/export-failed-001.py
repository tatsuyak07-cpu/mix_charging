"""Archive fixed-geometry elastic refinement and a continuous unloading control.

The 4 mm fixture has 0.25% initial porosity; it is not the requested packing.
Failed solver checks remain visible alongside separately named diagnostics.
"""
import argparse,csv,hashlib,json,shutil,zipfile
from pathlib import Path
from packing_fem.explicit_periodic import deformation
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--refinements',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.output.with_suffix('.zip').exists():raise FileExistsError('Exclusive output required')
    s=json.loads((a.input/'summary.json').read_text());r=json.loads((a.input/'run/summary.json').read_text());c=json.loads((a.input/'run/config.json').read_text())
    if c.get('unload_s',0)<=0:raise ValueError('Unloading required')
    b=np.loadtxt(a.input/'run/accepted-step-audit.csv.gz',delimiter=',',skiprows=1)
    h=np.atleast_1d(np.genfromtxt(a.input/'run/history.csv',delimiter=',',names=True))
    a.output.mkdir(parents=True);shutil.copytree(a.input,a.output/'unloading')
    levels=[];previous=None
    for i,root in enumerate(a.refinements,1):
        ref=json.loads((root/'summary.json').read_text());shutil.copytree(root,a.output/f'initial-refinement-{i}')
        compliance=float(np.linalg.inv(ref['macro_stiffness'])[2,2])
        levels.append(dict(level=i,nodes=ref['nodes'],tetrahedra=ref['tetrahedra'],porosity=ref['initial_porosity'],axial_compliance_per_Nmm=compliance,relative_change_from_previous=None if previous is None else abs(compliance-previous)/compliance,checks=ref['checks']))
        previous=compliance
    with (a.output/'initial-refinement.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=[k for k in levels[0] if k!='checks']);w.writeheader();w.writerows({k:v for k,v in row.items() if k!='checks'} for row in levels)
    peak_pressure=c['pressure_mpa'];scale=r['energy_scale_Nmm'];recovery_start=c['ramp_s']+c.get('hold_s',0)+c['unload_s']
    recovery=b[:,0]>=recovery_start;loaded=b[:,3]>.01*peak_pressure
    cycle_KE=float(np.max(b[:,4])/scale);window_KE=float(np.max(b[loaded,4]/np.maximum(abs(b[loaded,5]-b[0,5])+b[loaded,8]+b[loaded,9],scale*1e-8)))
    # Diagnostic tolerances declared by the export tool, separate from solver acceptance.
    # These do not change the solver, its tolerances, material, or loading history.
    with np.load(a.input/'run/last-state.npz') as state:
        F=deformation(state['generalized_position'][-6:]);position_error=float(np.max(np.linalg.norm(state['points']-state['reference_points'],axis=1)))
    checks=dict(macro_return_absolute_1e_6=float(np.max(abs(F-np.eye(3))))<1e-6,nodal_return_absolute_1e_5mm=position_error<1e-5,accepted_step_count=len(b)==r['steps']+1,final_command_zero=abs(b[-1,3])<peak_pressure*1e-12,no_plasticity_entire_sampled_history=float(np.max(h['peak_peeq']))<1e-10,axial_return_absolute_1e_6=abs(b[-1,1])<1e-6,porosity_return_absolute_1e_8=abs(b[-1,2]-b[0,2])<1e-8,cycle_peak_KE_below_1percent=cycle_KE<.01)
    report=dict(scope='connected_4mm_initial_porosity_0.25percent_elastic_control_not_particle_packing',initial_elastic_levels=levels,solver_checks=r['checks'],solver_and_endpoint_checks_passed=s['solver_and_endpoint_checks_passed'],additional_diagnostic_checks=checks,diagnostic_tolerances=dict(macro_F_absolute=1e-6,nodal_position_mm=1e-5,axial_return_absolute=1e-6,porosity_return_absolute=1e-8,plastic_strain=1e-10,cycle_peak_KE_fraction=.01),whole_pressure_window_KE_ratio=window_KE,cycle_peak_KE_over_energy_scale=cycle_KE,recovery_peak_KE_over_energy_scale=float(np.max(b[recovery,4])/scale) if np.any(recovery) else None,maximum_nodal_return_error_mm=position_error,final_macro_F=F.tolist(),final_axial_compression=float(b[-1,1]),porosity_return_difference=float(b[-1,2]-b[0,2]),peak_sampled_peeq=float(np.max(h['peak_peeq'])),plastic_unloading_verified=False,pressure_path_mesh_convergence_verified=False,particle_40_to_10_completed=False,failure=r['failure'])
    (a.output/'verification-summary.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axs=plt.subplots(1,3,figsize=(15,4.5))
    axs[0].plot([v['tetrahedra'] for v in levels],[v['axial_compliance_per_Nmm'] for v in levels],'o-');axs[0].set(xlabel='Initial tetrahedra',ylabel='Relaxed axial compliance [1/(N mm)]',title='Fixed geometry, initial elastic tangent')
    axs[1].plot(1000*b[:,0],b[:,3],label='Command pressure');axs[1].set(xlabel='Time [ms]',ylabel='Compression pressure [MPa]');ax=axs[1].twinx();ax.plot(1000*b[:,0],b[:,1],color='tab:orange',label='Axial compression');ax.set_ylabel('Axial compression',color='tab:orange')
    axs[2].plot(b[:,3],100*b[:,2]);axs[2].set(xlabel='Command pressure [MPa]',ylabel='Porosity [%]',title='Continuous load / hold / unload / recovery')
    for ax in axs:ax.grid(alpha=.25)
    fig.suptitle('Connected 4 mm verification fixture: E=10 MPa, initial porosity 0.25%; not packing');fig.tight_layout();fig.savefig(a.output/'refinement-unloading.png',dpi=180);plt.close(fig)
    shutil.copy2(__file__,a.output/'export_connected_unloading.py')
    shutil.copy2('tests/test_cauchy_unloading.py',a.output/'test_cauchy_unloading.py')
    logs=a.output/'logs';logs.mkdir()
    for name in ('connected-initial-refinement3-001.log','connected-elastic-unload-001.log','connected-next-tests-001.log','connected-cauchy-unload-test-001.log'):
        path=Path('/tmp')/name
        if path.exists():shutil.copy2(path,logs/name)
    (a.output/'README.md').write_text('Initial elastic refinement (three levels) and continuous elastic unloading control.\nSee verification-summary.json for separate solver acceptance and additional diagnostics.\nNo plastic-path mesh convergence, plastic unloading validation or particle 40% to 10% result.\n')
    hashes={str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.output.rglob('*')) if f.is_file()}
    (a.output/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    with zipfile.ZipFile(a.output.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(a.output.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(a.output.parent))
    with zipfile.ZipFile(a.output.with_suffix('.zip')) as z:
        if z.testzip():raise RuntimeError('ZIP CRC failed')
    print(json.dumps(report,indent=2));print('ZIP CRC passed')

if __name__=='__main__':main()
