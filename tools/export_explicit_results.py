"""Collect verified and failed prototype trials without modifying original runs."""
import argparse,csv,json,shutil,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from packing_fem.adaptive_tet import boundary,closed_surface_volume
from packing_fem.explicit_material import geometry,response


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    root=Path('results/explicit-verification-006');sources={d.name:d for d in root.iterdir() if d.is_dir()}
    sources.update(remesh=Path('results/explicit-remesh-002'),unload=Path('results/explicit-unload-001'),dynamic_impact=Path('results/explicit-impact-004'),incomplete_contact_trial=Path('results/explicit-contact-integration-002'))
    reports={};histories={};configs={}
    for name,src in sources.items():
        dst=out/name;dst.mkdir()
        for f in ['config.json','initial-mesh.npz','history.csv','summary.json','last-state.npz','last-mesh.vtu','progress.jsonl']:
            if (src/f).exists():shutil.copyfile(src/f,dst/f)
        s=json.loads((src/'summary.json').read_text());c=json.loads((src/'config.json').read_text());h=np.genfromtxt(src/'history.csv',names=True,delimiter=',')
        available=h[0]['kinetic_Nmm']+h[0]['normal_contact_Nmm']+h[0]['tangent_contact_Nmm']
        change=np.abs(h['internal_Nmm']-h[0]['internal_Nmm']);scale=max(float(change.max()),float(np.abs(h['external_work_Nmm']).max()),float(available),1e-15)
        loaded=h['axial_compression']>.5*c['compression_fraction']
        ke=float(np.max(h['kinetic_Nmm'][loaded]/np.maximum(change[loaded],scale*1e-8))) if loaded.any() else None
        with np.load(src/'last-state.npz') as state:
            x,t=state['points'],state['tets'];V=geometry(x,t)[0];J=response(state['elastic_b'],state['peeq'],c)[2]
            boundary_error=abs(closed_surface_volume(x,t)-V.sum())/V.sum();material_error=float(np.max(np.abs(V-state['reference_volume']*J)/state['reference_volume']))
        reports[name]={'source':str(src),'solver_status':s['status'],'failure':s['failure'],'original_checks':s['checks'],
          'observed_active_contacts':int(h['contacts'].max()),'accepted_remeshes':sum(e['accepted'] for e in s['remesh_events']),
          'peak_nominal_MPa':float(h['nominal_compression_MPa'].max()),'final_porosity':float(h[-1]['porosity']),
          'energy_residual_Nmm':float(np.abs(h['energy_residual_Nmm']).max()),'energy_scale_Nmm':scale,
          'postprocessed_energy_relative_error':float(np.abs(h['energy_residual_Nmm']).max()/scale),
          'postprocessed_loaded_kinetic_ratio':ke,'closed_boundary_volume_relative_error':float(boundary_error),
          'material_volume_relative_error':material_error,'residual_compression_strain':s.get('residual_compression_strain')}
        histories[name]=h;configs[name]=c
    base=reports['void_time1_cfl0.2'];sensitivity={}
    for name in ['void_time2_cfl0.2','void_time1_cfl0.1','void_density0.5','void_damping0.5']:
        s=reports[name];sensitivity[name]={'peak_stress_relative_difference':abs(s['peak_nominal_MPa']/base['peak_nominal_MPa']-1),'final_porosity_absolute_difference':abs(s['final_porosity']-base['final_porosity'])}
    summary={'runs':reports,'global_response_sensitivity':sensitivity,
      'sensitivity_2percent_stress_1e-4_porosity':all(v['peak_stress_relative_difference']<.02 and v['final_porosity_absolute_difference']<1e-4 for v in sensitivity.values()),
      'dynamic_impact_functional_observations':{'active_contact':reports['dynamic_impact']['observed_active_contacts']>0,'accepted_remesh':reports['dynamic_impact']['accepted_remeshes']>0,'is_quasi_static_validation':False},
      'note':'Postprocessing uses change in internal energy, excluding historic plastic work; nonzero initial kinetic energy belongs in the energy reference scale. Original solver summaries are retained unchanged.',
      'physical_40_to_10percent_result_validated':False,
      'limitations':['linear-tetra locking unverified','node-face contact without edge-edge coverage','local internal 2-to-3 remeshing, no boundary regeneration','density and damping numerical','Hencky Kirchhoff-yield material is not identical to native CalculiX model']}
    (out/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['case','status','peak_nominal_MPa','final_porosity','energy_relative_error','loaded_kinetic_ratio','active_contacts','accepted_remeshes'])
        for name,s in reports.items():w.writerow([name,s['solver_status'],s['peak_nominal_MPa'],s['final_porosity'],s['postprocessed_energy_relative_error'],s['postprocessed_loaded_kinetic_ratio'],s['observed_active_contacts'],s['accepted_remeshes']])
    prepared=out/'prepared-40percent-model';prepared.mkdir()
    for name in ['config.json','model.json','model-mesh.npz']:shutil.copyfile(Path('results/explicit-porosity40-model-001')/name,prepared/name)
    tests=out/'software-tests';tests.mkdir();shutil.copyfile('results/explicit-tests-002/pytest.log',tests/'pytest.log')
    failures=out/'retained-early-failures';failures.mkdir()
    for src,label in [(Path('results/explicit-verification-001/solid-error.json'),'initial-serialization-error.json'),(Path('results/explicit-contact-integration-001/interruption.json'),'initial-remesh-search-interruption.json')]:
        if src.exists():shutil.copyfile(src,failures/label)
    for source in ['explicit-impact-001','explicit-impact-002','explicit-impact-003']:
        dst=failures/source;dst.mkdir()
        for name in ['config.json','summary.json','history.csv','initial-mesh.npz']:
            if (Path('results')/source/name).exists():shutil.copyfile(Path('results')/source/name,dst/name)
    fig,ax=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for name,h in histories.items():
        if name.startswith('void_'):
            ax[0,0].plot(100*h['axial_compression'],h['nominal_compression_MPa'],label=name)
            change=abs(h['internal_Nmm']-h[0]['internal_Nmm']);load=h['axial_compression']>.5*configs[name]['compression_fraction']
            ax[0,1].plot(h['time_s'][load]/configs[name]['ramp_s'],100*h['kinetic_Nmm'][load]/np.maximum(change[load],1e-15),label=name)
    ax[0,0].set(xlabel='Axial compression (%)',ylabel='Nominal compression (MPa)');ax[0,0].legend(fontsize=8)
    ax[0,1].axhline(1,color='red',ls=':',label='Trial 1% criterion');ax[0,1].set(xlabel='Time / ramp duration',ylabel='Kinetic / internal energy change (%)');ax[0,1].legend(fontsize=8)
    h=histories['unload'];ax[1,0].plot(100*h['axial_compression'],h['nominal_compression_MPa']);ax[1,0].set(xlabel='Axial compression (%)',ylabel='Nominal compression (MPa)',title='Plastic compression and load release')
    events=json.loads((sources['remesh']/'summary.json').read_text())['remesh_events'];events=[e for e in events if e['accepted']]
    ids=np.arange(len(events));ax[1,1].bar(ids-.18,[e['quality_before'] for e in events],.36,label='Before');ax[1,1].bar(ids+.18,[e['quality_after'] for e in events],.36,label='After');ax[1,1].set(xlabel='Accepted cavity reconnection',ylabel='Local minimum tetrahedron quality');ax[1,1].legend()
    for aa in ax.flat:aa.grid(alpha=.25)
    fig.suptitle('Research explicit FEM: small verification problems\nNot a validated 40% to 10% compaction calculation');fig.savefig(out/'verification.png',dpi=180);plt.close(fig)
    h=histories['dynamic_impact'];fig,ax=plt.subplots(1,3,figsize=(13,4.4),layout='constrained');time=1e6*h['time_s']
    ax[0].plot(time,h['contacts']);ax[0].set(xlabel='Time (microseconds)',ylabel='Active node-face contacts')
    ax[1].plot(time,h['penetration_mm']);ax[1].axhline(configs['dynamic_impact']['max_penetration_mm'],c='red',ls=':');ax[1].set(xlabel='Time (microseconds)',ylabel='Penetration (mm)')
    ax[2].plot(time,100*h['energy_residual_Nmm']/reports['dynamic_impact']['energy_scale_Nmm']);ax[2].set(xlabel='Time (microseconds)',ylabel='Energy residual / available energy (%)')
    for aa in ax:aa.grid(alpha=.25)
    fig.suptitle('Dynamic diagnostic: contact + initial adaptive reconnection\nStopped at penetration limit; NOT quasi-static validation');fig.savefig(out/'dynamic-contact-diagnostics.png',dpi=180);plt.close(fig)
    with np.load(prepared/'model-mesh.npz') as mesh:X,t=mesh['points'],mesh['tets']
    polygons=X[boundary(t)];fig=plt.figure(figsize=(8,8));aa=fig.add_subplot(projection='3d');aa.add_collection3d(Poly3DCollection(polygons,facecolors='#86b8d3',edgecolors='#263e4a',linewidths=.08,rasterized=True))
    for setlim in [aa.set_xlim,aa.set_ylim,aa.set_zlim]:setlim(0,25)
    aa.set_box_aspect([1,1,1]);aa.view_init(elev=24,azim=-55);aa.set(xlabel='X (mm)',ylabel='Y (mm)',zlabel='Z (mm)',title='Prepared model only | initial porosity 39.8592%\n56,382 conforming tetrahedra | voxel-derived surface')
    fig.savefig(out/'prepared-model-mesh.png',dpi=180,bbox_inches='tight');plt.close(fig)
    (out/'README.md').write_text('Research explicit FEM prototype. See ../../docs/explicit-prototype.md.\n\nSmall quasi-static benchmarks, remesh transfer, plastic unloading and FAILED dynamic integration trials are distinct. Prepared 25 mm model has not been run to 10% porosity. No full 100 mm result is included.\n')
    with zipfile.ZipFile(out.with_suffix('.zip'),'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob('*')):
            if path.is_file():archive.write(path,path.relative_to(out.parent))
    with zipfile.ZipFile(out.with_suffix('.zip')) as archive:assert archive.testzip() is None
    print(json.dumps({'sensitivity':sensitivity,'impact_functional':summary['dynamic_impact_functional_observations'],'zip_bytes':out.with_suffix('.zip').stat().st_size},indent=2))

if __name__=='__main__':main()
