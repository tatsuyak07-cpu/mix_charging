"""Export observed converged load-step endpoints, including incomplete ramps."""
import argparse,csv,json,shutil,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    summary=json.loads((a.source/'mode-summary.json').read_text());config=json.loads((a.source/'config.json').read_text())
    h=np.atleast_1d(np.genfromtxt(a.source/'mode-history.csv',delimiter=',',names=True))
    h=h[h['step_time']<=1+1e-8]
    if not len(h):raise ValueError('No completed compression endpoints')
    with np.load(a.source/'sintered-geometry.npz') as g:
        from packing_fem.sintered import hex_quadrature
        phi0=1-hex_quadrature(g['points'],g['elements']).sum()/config['length_mm']**3
    with (a.output/'pressure-porosity.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n')
        w.writerow(['prescribed_nominal_MPa','measured_nominal_MPa','measured_cauchy_MPa','axial_compression_percent','geometry_porosity_percent','material_porosity_percent','peak_peeq','min_material_J','macro_J','absolute_porosity_definition_difference','porosity_definition_check'])
        w.writerow([0,0,0,0,100*phi0,100*phi0,0,1,1,0,True])
        for row in h:w.writerow([config['target_mpa']*row['step_time'],row['nominal_compression_MPa'],row['cauchy_compression_MPa'],100*row['axial_compression_strain'],100*row['geometry_porosity'],100*row['material_porosity'],row['peak_peeq'],row['min_material_J'],row['macro_J'],abs(row['geometry_porosity']-row['material_porosity']),bool(abs(row['geometry_porosity']-row['material_porosity'])<1e-4)])
    for name in ['config.json','estimate.json','mode-metadata.json','mode-summary.json','mode-history.csv','mode-energy.csv','mode-last-frame.vtu','mode.log','plastic.inp','plastic.sta','provenance.json','sintered-geometry.npz','recovery.json','ramp-stop.json','analysis-error.json']:
        if (a.source/name).exists():shutil.copyfile(a.source/name,a.output/name)
    volume_ok=np.abs(h['geometry_porosity']-h['material_porosity'])<1e-4
    report={'first_volume_audit_failure_nominal_MPa':float(h['nominal_compression_MPa'][~volume_ok][0]) if np.any(~volume_ok) else None,
      'last_volume_audit_passing_nominal_MPa':float(h['nominal_compression_MPa'][volume_ok][-1]) if np.any(volume_ok) else None,
      'source':str(a.source),'requested_nominal_MPa':config['target_mpa'],
      'completed_compression_endpoints':len(h),'reached_nominal_MPa':float(h[-1]['nominal_compression_MPa']),
      'reached_cauchy_MPa':float(h[-1]['cauchy_compression_MPa']),
      'target_reached':bool(h[-1]['step_time']>=1-1e-8),'initial_porosity':phi0,
      'last_compression_geometry_porosity':float(h[-1]['geometry_porosity']),
      'last_compression_material_porosity':float(h[-1]['material_porosity']),
      'solver_checks':summary['checks'],'interpretation':'Exploratory converged step endpoints only; not a validated physical pressure/porosity curve. No self-contact, no mesh convergence, uncalibrated material; 25 mm cell.'}
    (a.output/'ramp-report.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    pressure=np.r_[0,h['nominal_compression_MPa']]
    for field,label,style in [('geometry_porosity','Corner geometry','o-'),('material_porosity','Full material gradients','x--')]:
        axes[0].plot(pressure,np.r_[phi0,h[field]]*100,style,label=label,markersize=3)
    if np.any(~volume_ok):
        axes[0].scatter(h['nominal_compression_MPa'][~volume_ok],100*h['geometry_porosity'][~volume_ok],color='red',marker='x',label='Volume audit failed',zorder=5)
    axes[0].set(xlabel='Nominal compression (MPa)',ylabel='Porosity (%)');axes[0].legend()
    axes[1].plot(pressure,np.r_[0,h['axial_compression_strain']]*100,'o-',markersize=3)
    axes[1].set(xlabel='Nominal compression (MPa)',ylabel='Axial compression (%)')
    axes[2].plot(pressure,np.r_[0,h['peak_peeq']],'o-',markersize=3)
    axes[2].set(xlabel='Nominal compression (MPa)',ylabel='Peak equivalent plastic strain')
    for ax in axes:ax.grid(alpha=.3)
    fig.suptitle(f"Exploratory 25 mm periodic cell, E=10 MPa | target {config['target_mpa']:g} MPa\nNo self-contact; material and mesh not validated")
    fig.savefig(a.output/'pressure-porosity.png',dpi=180);plt.close(fig)
    with zipfile.ZipFile(a.output.with_suffix('.zip'),'x',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(a.output.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(a.output.parent))
    with zipfile.ZipFile(a.output.with_suffix('.zip')) as z:assert z.testzip() is None
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
