"""Export small-cell validation, failed compaction attempts and actual meshes."""
import argparse,csv,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
from packing_fem.contact_audit import plain
from packing_fem.explicit_periodic import deformation
from packing_fem.explicit_material import geometry,response
from packing_fem.packing_rve import continuum_porosity


def load(root,name):
    path=root/name/'calculation';return path,json.loads((path/'summary.json').read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    root=a.input;coarse,rc=load(root,'native-pressure0001-slow-001');fine,rf=load(root,'native-pressure0001-cfl01-001');large,rl=load(root,'native-pressure01-clarke-001');linear=json.loads((root/'linear-reference-003.json').read_text());ref32=json.loads((root/'linear-reference32-001.json').read_text());backend=json.loads((root/'backend-comparison-001/summary.json').read_text());independent=json.loads((root/'low-load-independent-audit-002/summary.json').read_text())
    plan=dict(scope=__doc__,geometry='2 periodic BCC spheres, 8 diagonal necks, fixed L=12.1 mm; not a statistical random RVE or the 100 mm specimen',material='E10 MPa, nu0.33, trial Kirchhoff-yield Hencky J2 y0.05/H0.2 MPa, mu0.3',low_control='0.001 MPa Cauchy pressure, 8 ms quintic ramp + 2 ms hold; no plasticity/contact expected',requested_load='0.1 MPa; rapid 4 ms + 2 ms hold diagnostic, not a quasi-static acceptance result',limits=dict(final_compression_absolute=1e-7,final_porosity_absolute=1e-7,path_porosity_absolute=1e-7,path_compression_over_peak=1e-3,linear_axial_relative=.01,mean_stress_relative=.02),mesh_convergence_verified=False)
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    A=np.loadtxt(coarse/'accepted-step-audit.csv.gz',delimiter=',',skiprows=1);B=np.loadtxt(fine/'accepted-step-audit.csv.gz',delimiter=',',skiprows=1);t=np.unique(np.r_[A[:,0],B[:,0]]);diff=[]
    for column in (1,2):diff.append(np.interp(t,A[:,0],A[:,column])-np.interp(t,B[:,0],B[:,column]))
    metrics=dict(final_compression_absolute=abs(rc['final_axial_compression']-rf['final_axial_compression']),final_porosity_absolute=abs(rc['final_porosity']-rf['final_porosity']),path_porosity_absolute=float(np.max(abs(diff[1]))),path_compression_over_peak=float(np.max(abs(diff[0]))/max(A[:,1].max(),B[:,1].max())),linear_axial_relative=abs(rc['final_axial_compression']/-linear['macro_strain'][2]-1))
    with np.load(coarse/'last-state.npz') as state:
        c=json.loads((coarse/'config.json').read_text());volume=geometry(state['points'],state['tets'])[0];sigma=response(state['elastic_b'],state['peeq'],c)[0];F=deformation(state['generalized_position'][-6:]);mean=np.einsum('e,eij->ij',volume,sigma)/(np.linalg.det(F)*c['length_mm']**3)
    metrics['mean_stress_relative']=float(np.max(abs(mean-np.diag([0.,0.,-.001])))/.001)
    cfl_checks={k:bool(v<=plan['limits'][k]) for k,v in metrics.items()};low_checks=dict(native_coarse=all(rc['checks'].values()),native_fine=all(rf['checks'].values()),linear_reference=all(linear['checks'].values()),backend_agreement=backend['passed'],independent_force_audit=independent['all_checks_passed'],**cfl_checks)
    e16=.001/-linear['macro_strain'][2];e32=.001/-ref32['reference']['macro_strain'][2]
    result=plain(dict(scope=__doc__,small_cell_low_load_checks=low_checks,low_load_baseline_passed=all(low_checks.values()),cfl_and_reference_metrics=metrics,low_load_pressure_MPa=.001,initial_porosity=rc['initial_porosity'],continuum_geometry_porosity=continuum_porosity(12.1),low_load_final_porosity_coarse=rc['final_porosity'],low_load_final_porosity_fine=rf['final_porosity'],low_load_axial_compression=rc['final_axial_compression'],independent_mean_stress_MPa=mean,active_contacts_in_low_load=int(A[:,10].max()),initial_elastic_modulus_n16_MPa=e16,initial_elastic_modulus_n32_MPa=e32,initial_mesh_modulus_relative_change=abs(e32/e16-1),large_load_status=rl['status'],large_load_failure=rl['failure'],large_load_stop_command_MPa=rl['peak_command_pressure_MPa'],large_load_last_porosity=rl['final_porosity'],large_load_last_compression=rl['final_axial_compression'],large_load_native_checks=rl['checks'],large_load_surface_events=rl['remesh_events'],requested_01MPa_quasistatic_compaction_verified=False,porosity40_to10_verified=False,mesh_convergence_verified=False,statistical_representativeness_verified=False,decision='Low-load small-cell baseline verified; retain all stopped/failed large-deformation attempts. No requested-load physical result accepted.'))
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savetxt(out/'cfl-path-differences.csv',np.column_stack([t,*diff]),delimiter=',',header='time_s,compression_coarse_minus_fine,porosity_coarse_minus_fine',comments='')
    shutil.copytree(root,out/'validation')
    source=out/'source';source.mkdir()
    for folder in ('packing_fem','native'):
        target=source/folder;target.mkdir()
        for f in Path(folder).glob('*'):
            if f.is_file() and f.suffix in ('.py','.cpp'):shutil.copy2(f,target/f.name)
    for folder,pattern in [('tests','test_*rve*.py'),('tests','test_native_kernel.py'),('tests','test_closest_tie.py'),('tools','*rve*.py'),('tools','build_native_kernel.py')]:
        target=source/folder;target.mkdir(exist_ok=True)
        for f in Path(folder).glob(pattern):
            if not (target/f.name).exists():shutil.copy2(f,target/f.name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for path,label in ((coarse,'CFL 0.2'),(fine,'CFL 0.1')):
        h=np.loadtxt(path/'history.csv',delimiter=',',skiprows=1);axes[0,0].plot(h[:,0]*1e3,h[:,24]*1e3,label=label);axes[0,1].plot(h[:,0]*1e3,h[:,2]*100,label=label)
    axes[0,0].set(xlabel='Time [ms]',ylabel='Command Cauchy pressure [kPa]',title='Low-load protocol');axes[0,0].legend();axes[0,1].set(xlabel='Time [ms]',ylabel='Porosity [%]',title='Low-load baseline, not large compaction');axes[0,1].legend()
    for name,label in [('native-pressure01-rapid-001','No adaptation: stopped'),('native-pressure01-adaptive-001','Edge refinement: stopped'),('native-pressure01-clarke-001','Tie subgradient trial: diagnostics')]:
        h=np.loadtxt(root/name/'calculation/history.csv',delimiter=',',skiprows=1);axes[1,0].plot(h[:,24],h[:,2]*100,label=label)
    axes[1,0].set(xlabel='Command pressure [MPa]',ylabel='Porosity [%]',title='Transient large-load attempts, NOT equilibria');axes[1,0].legend(fontsize=7)
    axes[1,1].bar(['n=16','n=32'],[e16,e32],color=['steelblue','orange']);axes[1,1].set(ylabel='Initial axial modulus [MPa]',title='Mesh dependence remains unresolved')
    fig.tight_layout();fig.savefig(out/'packing-rve-validation.png',dpi=170);plt.close(fig)
    from packing_fem.adaptive_tet import boundary
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig=plt.figure(figsize=(11,5))
    with np.load(large/'initial-mesh.npz') as z:X=z['points'];T=z['tets']
    with np.load(large/'last-state.npz') as z:x=z['points'];Tlast=z['tets'];q=z['generalized_position']
    for k,(points,tets,title) in enumerate(((X,T,'Initial BCC voxel solid FEM'),(x,Tlast,'Last accepted iterate, NOT equilibrium'))):
        ax=fig.add_subplot(1,2,k+1,projection='3d');faces=boundary(tets);ax.add_collection3d(Poly3DCollection(points[faces],facecolors='steelblue',edgecolors='#253b53',linewidths=.12,alpha=.8));ax.set(xlim=(points[:,0].min(),points[:,0].max()),ylim=(points[:,1].min(),points[:,1].max()),zlim=(points[:,2].min(),points[:,2].max()),xlabel='x [mm]',ylabel='y [mm]',zlabel='z [mm]',title=title);ax.set_box_aspect(np.ptp(points,axis=0));ax.view_init(22,35)
    fig.tight_layout();fig.savefig(out/'actual-packing-mesh.png',dpi=180);plt.close(fig)
    hashes={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.rglob('*')) if f.is_file()};(out/'SHA256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob('*')):
            if f.is_file():z.write(f,str(f.relative_to(out.parent)))
    print(json.dumps(result,indent=2))
    if not result['low_load_baseline_passed']:raise SystemExit(2)


if __name__=='__main__':main()
