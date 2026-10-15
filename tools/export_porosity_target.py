"""Export porosity-target trial endpoints, their audit status and exact mesh."""
import argparse,csv,json,shutil,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import meshio
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from packing_fem.sintered import hex_quadrature


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--phase-record',type=Path);p.add_argument('--failed-setup',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    c=json.loads((a.source/'config.json').read_text());s=json.loads((a.source/'mode-summary.json').read_text());target=c['target_final_porosity']
    h=np.atleast_1d(np.genfromtxt(a.source/'mode-history.csv',names=True,delimiter=','));h=h[h['step_time']<=1+1e-8]
    with np.load(a.source/'sintered-geometry.npz') as g:p0,e=g['points'],g['elements']
    initial=1-hex_quadrature(p0,e).sum()/c['length_mm']**3
    ok=np.abs(h['geometry_porosity']-h['material_porosity'])<1e-4
    with (a.output/'porosity-history.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['stage','axial_compression_percent','nominal_compression_MPa','cauchy_compression_MPa','geometry_porosity_percent','material_porosity_percent','peak_peeq','volume_definition_check','both_porosity_targets_reached'])
        w.writerow([0,0,0,0,100*initial,100*initial,0,True,False])
        for r,passing in zip(h,ok):w.writerow([r['step_time'],100*r['axial_compression_strain'],r['nominal_compression_MPa'],r['cauchy_compression_MPa'],100*r['geometry_porosity'],100*r['material_porosity'],r['peak_peeq'],bool(passing),bool(0<=min(r['geometry_porosity'],r['material_porosity']) and max(r['geometry_porosity'],r['material_porosity'])<=target)])
    for name in ['config.json','estimate.json','provenance.json','mode-metadata.json','mode-summary.json','mode-history.csv','mode-energy.csv','mode-last-frame.vtu','mode.log','plastic.inp','plastic.sta','sintered-geometry.npz','ramp-stop.json','recovery.json','analysis-error.json']:
        if (a.source/name).exists():shutil.copyfile(a.source/name,a.output/name)
    if a.phase_record:shutil.copyfile(a.phase_record,a.output/'phase-scan.json')
    if a.failed_setup:
        dst=a.output/'failed-envelope-setup';dst.mkdir()
        for name in ['config.json','estimate.json','analysis-error.json']:shutil.copyfile(a.failed_setup/name,dst/name)
    report={'requested_initial_porosity':c['porosity'],'actual_initial_porosity':initial,'target_final_porosity':target,
      'minimum_observed_geometry_porosity':float(h['geometry_porosity'].min()),'minimum_observed_material_porosity':float(h['material_porosity'].min()),
      'target_observed':bool(np.any((np.minimum(h['geometry_porosity'],h['material_porosity'])>=0)&(np.maximum(h['geometry_porosity'],h['material_porosity'])<=target))),
      'first_volume_audit_failure_compression_percent':float(100*h['axial_compression_strain'][~ok][0]) if np.any(~ok) else None,
      'complete_compression_endpoints':len(h),'status':s['status'],'checks':s['checks'],
      'limitations':['25 mm trial, original 100 mm size unverified','no self-contact or damage','uncalibrated material','mesh convergence unverified','40% is a requested nominal initial porosity; actual voxel value reported separately','no extrapolation to 10%']}
    (a.output/'target-report.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,ax=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
    strain=np.r_[0,100*h['axial_compression_strain']]
    ax[0].plot(strain,np.r_[initial,h['geometry_porosity']]*100,'o-',label='Corner geometry',markersize=3)
    ax[0].plot(strain,np.r_[initial,h['material_porosity']]*100,'x--',label='Full material gradients',markersize=3)
    ax[0].axhline(100*target,color='grey',linestyle=':',label=f'Requested target {100*target:g}%')
    if np.any(~ok):ax[0].scatter(100*h['axial_compression_strain'][~ok],100*h['geometry_porosity'][~ok],color='red',marker='x',label='Volume audit failed',zorder=5)
    ax[0].set(xlabel='Axial compression (%)',ylabel='Porosity (%)');ax[0].legend()
    ax[1].plot(strain,np.r_[0,h['cauchy_compression_MPa']],'o-',markersize=3);ax[1].set(xlabel='Axial compression (%)',ylabel='Cauchy compression (MPa)')
    for x in ax:x.grid(alpha=.3)
    fig.suptitle(f'Porosity target trial: actual initial {100*initial:.4f}% -> requested {100*target:g}%\n25 mm periodic cell, E=10 MPa; no self-contact; not physically validated')
    fig.savefig(a.output/'porosity-target.png',dpi=180);plt.close(fig)
    mesh=meshio.read(a.source/'mode-last-frame.vtu');points=mesh.points
    face_nodes=np.array([[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]])
    faces=e[:,face_nodes].reshape(-1,4);_,inverse,count=np.unique(np.sort(faces,axis=1),axis=0,return_inverse=True,return_counts=True)
    polygons=points[faces[count[inverse]==1]]
    fig=plt.figure(figsize=(8,8));ax=fig.add_subplot(projection='3d');ax.add_collection3d(Poly3DCollection(polygons,facecolors='#86b8d3',edgecolors='#263e4a',linewidths=.12,rasterized=True))
    for k,setlim in enumerate((ax.set_xlim,ax.set_ylim,ax.set_zlim)):setlim(points[:,k].min(),points[:,k].max())
    ax.set_box_aspect(np.ptp(points,axis=0));ax.view_init(elev=24,azim=-55)
    ax.set(xlabel='X (mm)',ylabel='Y (mm)',zlabel='Z (mm)',title=f'Last complete endpoint | true displacement scale 1x\nPorosity {100*h[-1]["geometry_porosity"]:.3f}% | exploratory diagnostics')
    fig.savefig(a.output/'last-complete-mesh.png',dpi=180,bbox_inches='tight');plt.close(fig)
    with zipfile.ZipFile(a.output.with_suffix('.zip'),'x',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(a.output.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(a.output.parent))
    with zipfile.ZipFile(a.output.with_suffix('.zip')) as z:assert z.testzip() is None
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
