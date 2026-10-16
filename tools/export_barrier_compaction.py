"""Archive completed compaction trials and plot their actual mesh/history."""
import argparse,csv,json,shutil,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--mesh-case',default='barrier-pressure-time32-001');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);rows=[]
    for root in sorted(Path('results').glob('barrier-*')):
        if not (root/'summary.json').exists():
            if 'preflight' in root.name:shutil.copytree(root,a.output/root.name)
            continue
        shutil.copytree(root,a.output/root.name)
        s=json.loads((root/'summary.json').read_text());c=json.loads((root/'config.json').read_text())
        rows.append(dict(case=root.name,length_mm=c['length_mm'],initial_porosity_percent=100*s['initial_porosity'],final_porosity_percent=100*s['final_porosity'],axial_compression=s['final_axial_compression'],ramp_s=c['ramp_s'],hold_s=c['hold_s'],cfl=c['cfl'],barrier_thickness_mm=c.get('barrier_thickness_mm'),penalty=c['contact_penalty_mpa_mm'],damping_per_s=c['damping_per_s'],mass_frame=s.get('mass_frame','pinned'),steps=s['steps'],loaded_kinetic_internal_ratio=s['kinetic_internal_ratio_loaded'],energy_relative_error=s['energy_relative_error'],static_residual_ratio=s['final_static_residual_ratio'],all_checks_passed=all(s['checks'].values()),failed_checks=';'.join(k for k,v in s['checks'].items() if not v),failure=s['failure'],audit_sampling=s.get('audit_sampling','output_history_only')))
    from packing_fem.explicit_periodic import deformation
    from packing_fem.explicit_material import geometry
    for row in rows:
        root=a.output/row['case']
        with np.load(root/'last-state.npz') as state:
            solid=float(geometry(state['points'],state['tets'])[0].sum())
            cell=float(np.linalg.det(deformation(state['generalized_position'][-6:]))*row['length_mm']**3)
        history=np.atleast_1d(np.genfromtxt(root/'history.csv',names=True,delimiter=','))
        row.update(final_cell_volume_mm3=cell,final_solid_volume_mm3=solid,final_pore_volume_mm3=cell-solid,peak_nominal_pressure_MPa=float(history['nominal_compression_MPa'].max()),final_time_s=float(history['time_s'][-1]))
    report='|Case|Initial porosity [%]|Final porosity [%]|Failed checks|\n|---|---:|---:|---|\n'
    for row in rows:report+=f"|{row['case']}|{row['initial_porosity_percent']:.7f}|{row['final_porosity_percent']:.7f}|{row['failed_checks'] or 'none among stored checks'}|\n"
    (a.output/'comparison.md').write_text(report)
    with (a.output/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for name,label in [('barrier-pressure-002','ramp 1 ms'),('barrier-pressure-time8-001','8 ms'),('barrier-pressure-time16-001','16 ms'),('barrier-pressure-time32-001','32 ms'),('barrier-pressure-zero-momentum-001','8 ms, zero momentum'),('barrier-pressure-thicknesshalf-001','16 ms, half barrier')]:
        path=a.output/name/'history.csv'
        if not path.exists():continue
        h=np.genfromtxt(path,names=True,delimiter=',');axes[0].plot(h['nominal_compression_MPa'],100*h['porosity'],label=label);axes[1].plot(h['nominal_compression_MPa'],100*h['axial_compression'],label=label)
        denom=np.maximum(abs(h['internal_Nmm']-h['internal_Nmm'][0]),1e-12);valid=h['time_s']>float(json.loads((path.parent/'config.json').read_text())['ramp_s'])/2
        axes[2].semilogy(h['time_s'][valid]*1000,np.maximum(h['kinetic_Nmm'][valid]/denom[valid],1e-16),label=label)
    axes[0].set(xlabel='Nominal compression [MPa]',ylabel='Porosity [%]');axes[1].set(xlabel='Nominal compression [MPa]',ylabel='Axial compression [%]');axes[2].set(xlabel='Numerical time [ms]',ylabel='Kinetic / internal change');axes[2].axhline(.01,color='black',ls=':',label='1% screening limit')
    for ax in axes:ax.grid(True);ax.legend(fontsize=7)
    fig.suptitle('4 mm verification cell; initial porosity 0.25%. NOT the 40% particle packing.');fig.tight_layout();fig.savefig(a.output/'pressure-porosity.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for name,label in [('barrier-pressure-time16-001','standard'),('barrier-pressure-thicknesshalf-001','half barrier thickness'),('barrier-pressure-penalty2-001','double penalty'),('barrier-pressure-dampinghalf-001','half damping')]:
        path=a.output/name/'history.csv'
        if not path.exists():continue
        h=np.genfromtxt(path,names=True,delimiter=',')
        axes[0].plot(h['nominal_compression_MPa'],100*h['porosity'],label=label)
        axes[1].plot(h['nominal_compression_MPa'],100*h['axial_compression'],label=label)
    axes[0].set(xlabel='Nominal compression [MPa]',ylabel='Porosity [%]');axes[1].set(xlabel='Nominal compression [MPa]',ylabel='Axial compression [%]')
    for ax in axes:ax.grid(True);ax.legend(fontsize=8)
    fig.suptitle('4 mm verification, ramp 16 ms; contact regularization NOT converged')
    fig.tight_layout();fig.savefig(a.output/'contact-sensitivity.png',dpi=170);plt.close(fig)
    root=a.output/a.mesh_case;s=json.loads((root/'summary.json').read_text());c=json.loads((root/'config.json').read_text())
    with np.load(root/'last-state.npz') as data:X,x,t=data['reference_points'],data['points'],data['tets']
    edges=np.sort(np.concatenate([t[:,[i,j]] for i,j in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]]),axis=1);edges=np.unique(edges,axis=0)
    fig=plt.figure(figsize=(12,5))
    for k,(points,title) in enumerate([(X,'Initial mesh'),(x,'Actual last accepted mesh')]):
        ax=fig.add_subplot(1,2,k+1,projection='3d');ax.add_collection3d(Line3DCollection(points[edges],colors='steelblue',linewidths=.35,alpha=.6));ax.auto_scale_xyz(points[:,0],points[:,1],points[:,2]);ax.set(xlabel='x [mm]',ylabel='y [mm]',zlabel='z [mm]',title=title);ax.set_box_aspect(np.ptp(points,axis=0))
    fig.suptitle(f"4 mm verification cell, nominal pressure {c['pressure_mpa']} MPa; displacement scale 1\n{a.mesh_case}; mesh/contact convergence NOT established");fig.tight_layout();fig.savefig(a.output/'mesh-at-pressure.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    pore=(X[:,0]>=1-1e-9)&(X[:,0]<=3+1e-9)&np.isclose(X[:,1],1.)
    for points,label,style in [(X,'initial','--'),(x,'last accepted','-')]:
        center=np.mean(points[pore&(np.isclose(X[:,2],1.98)|np.isclose(X[:,2],2.02)),2])
        for level,color in [(1.98,'royalblue'),(2.02,'crimson')]:
            ids=np.flatnonzero(pore&np.isclose(X[:,2],level));ids=ids[np.argsort(X[ids,0])]
            ax.plot(points[ids,0],points[ids,2]-center,style,color=color,marker='o',label=f'{label}, reference z={level} mm')
    ax.set(xlabel='Actual x [mm]',ylabel='Actual z minus section center [mm]',title='Verification pore wall edges at reference y=1 mm; displacement scale 1')
    ax.grid(True);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(a.output/'pore-wall-section.png',dpi=180);plt.close(fig)
    logs=a.output/'test-logs';logs.mkdir()
    for f in sorted(Path('/tmp').glob('barrier-*-tests-*.log')):shutil.copy2(f,logs/f.name)
    for f in sorted(Path('/tmp').glob('barrier-tests-*.log')):shutil.copy2(f,logs/f.name)
    (a.output/'README.md').write_text('''# Periodic pressure-compaction development\n\nThe 4 mm verification cell starts at **0.25%** porosity. It is a manufactured thin-pore continuum fixture, not the requested sphere packing. Consult `comparison.csv` and each original `summary.json`: failed checks are retained and are not accepted physical solutions.\n\n`barrier-packing40-*` are **25 mm** RVE setup/cost/initialization trials only (actual initial porosity 39.8592%). They do not reach 0.1 MPa or 10% porosity. The requested 100 mm packing has not been solved.\n\nCSV units: mm, N, MPa, tonne, s. Porosity in history is a fraction; comparison uses percent. PNG meshes use actual coordinates and displacement scale 1. VTU and NPZ contain the actual last accepted mesh, including incomplete and deliberately interrupted trials. KeyboardInterrupt trials were stopped by the developer to replace expensive verified-equivalent search/derivative implementations; the isotropic mesh-refinement trial was stopped after its tiny stable timestep exposed the cost.\n\nThe code evolved during development: older records lack every-step audit, contact-energy variation and neighbor caching fields. Missing checks must not be interpreted as passed. Existing histories are preserved, not rewritten with newer check outcomes. All models use the stored input configuration.\n''')
    zpath=a.output.with_suffix('.zip')
    if zpath.exists():raise FileExistsError(zpath)
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
        for f in a.output.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(a.output.parent))
    with zipfile.ZipFile(zpath) as z:assert z.testzip() is None
    print(zpath)
if __name__=='__main__':main()
