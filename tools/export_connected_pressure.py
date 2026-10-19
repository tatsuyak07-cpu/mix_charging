"""Export complete connected-cell verification including all-step audit data."""
import argparse,csv,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--initial-refinement',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or a.output.with_suffix('.zip').exists():raise FileExistsError('Exclusive export required')
    for root in a.inputs:
        if not (root/'summary.json').is_file():raise ValueError(f'Incomplete run: {root}')
    refined=json.loads((a.initial_refinement/'summary.json').read_text())
    a.output.mkdir(parents=True);shutil.copytree(a.initial_refinement,a.output/'initial-refinement');rows=[];cases={};pairs=[]
    tangent_source=Path('exports/layered-contact-transient-001/connected-initial-stiffness/summary.json');tangent=json.loads(tangent_source.read_text());S=np.array(tangent['macro_stiffness']);shutil.copy2(tangent_source,a.output/'initial-tangent-reference.json')
    for root in a.inputs:
        shutil.copytree(root,a.output/root.name);s=json.loads((root/'summary.json').read_text());r=json.loads((root/'run/summary.json').read_text());e=json.loads((root/'endpoint-audit.json').read_text());c=json.loads((root/'run/config.json').read_text());b=np.loadtxt(root/'run/accepted-step-audit.csv.gz',delimiter=',',skiprows=1);b=np.atleast_2d(b);cases[root.name]=(s,r,e,c,b)
        loaded=b[:,3]>.01*b[:,3].max();contact=loaded&(b[:,10]>0);h=np.atleast_1d(np.genfromtxt(root/'run/history.csv',delimiter=',',names=True));total=abs(b[:,5]-b[0,5])+b[:,8]+b[:,9];ratio=b[:,4]/np.maximum(total,r['energy_scale_Nmm']*1e-8)
        integrity=len(b)==r['steps']+1 and abs(b[-1,2]-r['final_porosity'])<1e-12
        if np.any(loaded):integrity=integrity and np.isclose(r['whole_path_audit']['loaded']['peak_kinetic_Nmm'],np.max(b[loaded,4]),rtol=1e-12,atol=1e-20)
        reference=np.linalg.solve(S,np.array([0.,0.,-b[-1,3]*4**3,0.,0.,0.]));observed=np.array(e['macro_F'])-np.eye(3)
        rows.append(dict(case=root.name,pressure_MPa=s['pressure_mpa'],ramp_ms=1000*s['ramp_s'],cfl=s['cfl'],steps=r['steps'],final_time_ms=1000*b[-1,0],elapsed_s=s['elapsed_s'],record_integrity=bool(integrity),solver_endpoint_pass=s['solver_and_endpoint_checks_passed'],additional_1percent_window_KE_pass=bool(np.max(ratio[loaded])<c['kinetic_tolerance']) if np.any(loaded) else False,whole_loaded_KE_internal_contact_ratio=float(np.max(ratio[loaded])) if np.any(loaded) else None,whole_contact_KE_ratio=float(np.max(ratio[contact])) if np.any(contact) else None,peak_kinetic_Nmm=float(np.max(b[:,4])),peak_kinetic_over_energy_scale=float(np.max(b[:,4])/r['energy_scale_Nmm']),peak_contacts=int(np.max(b[:,10])),initial_porosity=r['initial_porosity'],final_porosity=r['final_porosity'],axial_compression=r['final_axial_compression'],peak_peeq=e['peak_peeq'],stress_relative_error=e['stress_relative_error'],energy_relative_error=r['energy_relative_error'],damping_work_Nmm=float(h['damping_Nmm'][-1]),damping_work_fraction=float(h['damping_Nmm'][-1]/h['external_work_Nmm'][-1]) if h['external_work_Nmm'][-1] else None,friction_work_Nmm=float(h['friction_Nmm'][-1]),late_KE_ratio=r['kinetic_internal_ratio_loaded'],linear_reference_applicable=e['peak_peeq']<1e-10,linear_axial_reference=-float(reference[2]),linear_axial_relative_difference=abs(observed[2,2]-reference[2])/abs(reference[2]),failure=s['failure']))
    for pressure in sorted({row['pressure_MPa'] for row in rows}):
        group=[row for row in rows if row['pressure_MPa']==pressure]
        for T in sorted({row['ramp_ms'] for row in group}):
            both=sorted([row for row in group if row['ramp_ms']==T],key=lambda row:row['cfl'],reverse=True)
            if len(both)!=2:continue
            coarse,fine=both;bc=cases[coarse['case']][-1];bf=cases[fine['case']][-1];times=np.linspace(0,min(bc[-1,0],bf[-1,0]),1001)
            pairs.append(dict(pressure_MPa=pressure,ramp_ms=T,kind='CFL_halving',coarse=coarse['case'],fine=fine['case'],endpoint_porosity_difference=abs(coarse['final_porosity']-fine['final_porosity']),endpoint_axial_difference=abs(coarse['axial_compression']-fine['axial_compression']),whole_step_interpolated_porosity_difference=float(np.max(abs(np.interp(times,bc[:,0],bc[:,2])-np.interp(times,bf[:,0],bf[:,2]))))))
    for pressure in sorted({row['pressure_MPa'] for row in rows}):
        for cfl in sorted({row['cfl'] for row in rows}):
            rates=sorted([row for row in rows if row['pressure_MPa']==pressure and row['cfl']==cfl],key=lambda row:row['ramp_ms'])
            if len(rates)==2:
                fast,slow=rates;bf=cases[fast['case']][-1];bs=cases[slow['case']][-1];phase=np.linspace(0,1,1001)
                pairs.append(dict(kind='ramp_duration_comparison',pressure_MPa=pressure,cfl=cfl,fast=fast['case'],slow=slow['case'],endpoint_porosity_difference=abs(fast['final_porosity']-slow['final_porosity']),endpoint_axial_difference=abs(fast['axial_compression']-slow['axial_compression']),same_phase_porosity_difference=float(np.max(abs(np.interp(phase,bf[:,0]/(fast['ramp_ms']/1000),bf[:,2])-np.interp(phase,bs[:,0]/(slow['ramp_ms']/1000),bs[:,2]))))))
    with (a.output/'comparison.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    ref_load=np.array([0.,0.,-.02*4**3,0.,0.,0.]);q_coarse=np.linalg.solve(S,ref_load);q_refined=np.linalg.solve(np.array(refined['macro_stiffness']),ref_load)
    initial_refinement=dict(checks=refined['checks'],coarse_nodes=tangent['nodes'],refined_nodes=refined['nodes'],coarse_tetrahedra=tangent['tetrahedra'],refined_tetrahedra=refined['tetrahedra'],porosity_difference=abs(tangent['initial_porosity']-refined['initial_porosity']),axial_compliance_difference_relative_to_refined=float(abs(q_coarse[2]-q_refined[2])/abs(q_refined[2])),interpretation='one initial elastic refinement comparison, not plastic pressure-path mesh convergence')
    summary=dict(initial_elastic_refinement=initial_refinement,scope='connected_4mm_initial_0.25percent_fixture_not_sphere_packing',cases=len(rows),solver_endpoint_pass_count=sum(row['solver_endpoint_pass'] for row in rows),additional_pressure_window_KE_pass_count=sum(row['additional_1percent_window_KE_pass'] for row in rows),all_step_record_integrity=all(row['record_integrity'] for row in rows),pressure_window='command > 1 percent of observed peak, instantaneous KE/(abs(internal change)+normal+tangent energy) < 1 percent',time_comparisons=pairs,mesh_convergence_verified=False,damping_sensitivity_verified=False,material_calibrated=False,unloading_verified=False,particle_40_to_10_completed=False)
    (a.output/'verification-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    for name,(s,r,e,c,b) in cases.items():
        label=f"P={s['pressure_mpa']:g}, T={1000*s['ramp_s']:g} ms, CFL={s['cfl']}";loaded=b[:,3]>.01*b[:,3].max();ratio=b[:,4]/np.maximum(abs(b[:,5]-b[0,5])+b[:,8]+b[:,9],r['energy_scale_Nmm']*1e-8)
        axes[0].plot(b[:,3],100*b[:,2],label=label);axes[1].plot(b[:,3],b[:,1],label=label);axes[2].semilogy(b[loaded,0]/s['ramp_s'],np.maximum(ratio[loaded],1e-12),label=label)
    axes[0].set(xlabel='Commanded Cauchy compression [MPa]',ylabel='Porosity [%]');axes[1].set(xlabel='Commanded Cauchy compression [MPa]',ylabel='Axial compression');axes[2].set(xlabel='Time / ramp duration',ylabel='KE / (internal change + contact energy)');axes[2].axhline(.01,color='black',linestyle=':',label='1% diagnostic')
    for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=6)
    fig.suptitle('Actual connected 4 mm FEM fixture, E=10 MPa, initial porosity 0.25% (not packing)');fig.tight_layout();fig.savefig(a.output/'connected-pressure-verification.png',dpi=180);plt.close(fig)
    chosen=max(rows,key=lambda row:(row['pressure_MPa'],row['solver_endpoint_pass'] and row['additional_1percent_window_KE_pass'],row['ramp_ms'],-row['cfl']));root=a.output/chosen['case']
    with np.load(root/'run/initial-mesh.npz') as z:X=z['points'];t0=z['tets']
    with np.load(root/'run/last-state.npz') as z:x=z['points'];t=z['tets']
    lo=np.minimum(X.min(0),x.min(0))-.15;hi=np.maximum(X.max(0),x.max(0))+.15;fig=plt.figure(figsize=(11,5))
    for panel,points,tets,title in ((1,X,t0,'Initial mesh'),(2,x,t,'Last accepted mesh')):
        edges=np.array(sorted({tuple(sorted((int(tet[i]),int(tet[j])))) for tet in tets for i in range(4) for j in range(i+1,4)}));ax=fig.add_subplot(1,2,panel,projection='3d');ax.add_collection3d(Line3DCollection(points[edges],colors='navy',linewidths=.5,alpha=.75));ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),xlabel='x [mm]',ylabel='y [mm]',zlabel='z [mm]',title=f'{title}: {len(tets)} tets');ax.set_box_aspect(hi-lo)
    b=cases[chosen['case']][-1];fig.suptitle(f"Connected verification mesh, scale 1; P={b[-1,3]:.6g} MPa, T={chosen['ramp_ms']:g} ms, CFL={chosen['cfl']}\nMaximum final PEEQ={chosen['peak_peeq']:.4g}; initial porosity 0.25%, not sphere packing");fig.tight_layout();fig.savefig(a.output/'connected-mesh.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4));sections=[]
    for points,tets in ((X,t0),(x,t)):
        edge_ids=np.array(sorted({tuple(sorted((int(tet[i]),int(tet[j])))) for tet in tets for i in range(4) for j in range(i+1,4)}))
        valid=np.isclose(X[:,1],2.)&(X[:,0]>=1.)&(X[:,0]<=3.)&(X[:,2]>=1.98-1e-10)&(X[:,2]<=2.02+1e-10)
        selected=edge_ids[np.all(valid[edge_ids],axis=1)];ids=np.unique(selected);center=points[ids].mean(0);sections.append((points-center,selected,ids))
    xspan=max(np.max(abs(points[ids,0])) for points,edges,ids in sections)*1.1;zspan=max(np.max(abs(points[ids,2])) for points,edges,ids in sections)*1.3
    for ax,(points,edges,ids),title in zip(axs,sections,('Initial pore edges','Final pore edges')):
        for edge in edges:ax.plot(points[edge,0],points[edge,2],'-o',color='navy',linewidth=.8,markersize=3)
        ax.set(xlim=(-xspan,xspan),ylim=(-zspan,zspan),xlabel='x from edge-node center [mm]',ylabel='z from edge-node center [mm]',title=title);ax.grid(alpha=.25)
    fig.suptitle('Reference y=2 mm material edges projected to x-z; recentered, axes independently scaled');fig.tight_layout();fig.savefig(a.output/'pore-midplane.png',dpi=180);plt.close(fig)
    logs=a.output/'logs';logs.mkdir()
    for path in sorted(Path('/tmp').glob('connected-pressure-*.log')):shutil.copy2(path,logs/path.name)
    shutil.copy2(__file__,a.output/'export_connected_pressure.py')
    (a.output/'README.md').write_text('''# 連結多孔質セルの圧縮検証\n\n4 mm角、初期空隙率0.25%、100節点・264初期四面体の検証用形状です。\n球充填体の40%→10%の結果ではありません。\ncomparison.csvとverification-summary.jsonで既存の後半・終点検査と追加KE監査を区別しています。\n全採用ステップのデータは各 run/accepted-step-audit.csv.gz です。\n追加KE窓は、圧力が観測最大値の1%を超える時点。初期の小さい分母にも注意してください。\n初期接線の一次予測は弾性の低圧比較のみで使い、塑性時に適用しません。\n連結セルの変形・接触・塑性を検証する試験で、初期弾性の1回の細分化比較を initial-refinement に保存しました。圧縮塑性のメッシュ収束ではありません。\n材料校正、除荷、一般辺接触、表面再メッシュ、メッシュ収束は未完了です。\n''')
    hashes={str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.output.rglob('*')) if f.is_file()};(a.output/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    with zipfile.ZipFile(a.output.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(a.output.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(a.output.parent))
    with zipfile.ZipFile(a.output.with_suffix('.zip')) as z:
        if z.testzip():raise RuntimeError('ZIP CRC failure')
    print(json.dumps(summary,indent=2));print('ZIP CRC passed')
if __name__=='__main__':main()
