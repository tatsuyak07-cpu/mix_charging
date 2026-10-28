"""Export a fixed-geometry component comparison, preserving original summaries."""
import argparse
import copy
import csv
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from packing_fem.neck_validation import top_surface_weights
from packing_fem.results import read_dat,find


def weighted_history(source,target):
    original=json.loads((source/'summary.json').read_text())
    with np.load(source/'geometry.npz') as g:p,e=g['points'],g['elements']
    weights=top_surface_weights(p,e)
    h=np.genfromtxt(source/'history.csv',delimiter=',',names=True)
    frames=0
    for t,s in read_dat(source/'neck.dat'):
        if frames>=len(h) or not np.isclose(t,h['step_time'][frames],atol=1e-8):raise ValueError('History frame mismatch')
        u=find(s,'displacements','PHYSICAL');u=u[np.argsort(u[:,0])][:,1:4]
        h['grip_compression_mm'][frames]=-np.dot(weights,u[:,2]);frames+=1
    if frames!=len(h):raise ValueError('Incomplete weighted history')
    result=copy.deepcopy(original)
    result['residual_grip_compression_mm']=float(h['grip_compression_mm'][-1])
    result['grip_mean_definition']='reference-area-weighted top surface'
    result['original_reported_residual_mm']=original['residual_grip_compression_mm']
    names=h.dtype.names
    np.savetxt(target/'history-area-weighted.csv',np.column_stack([h[n] for n in names]),delimiter=',',header=','.join(names),comments='')
    (target/'summary-area-weighted.json').write_text(json.dumps(result,indent=2)+'\n')
    return result,h


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    results={};histories={};rows=[];beams={}
    for kind in ('C3D8','C3D8I'):
        source=a.source/('beam_'+kind);target=a.output/('beam_'+kind);target.mkdir()
        for f in source.iterdir():
            if f.suffix in ('.json','.inp','.log','.dat'):shutil.copyfile(f,target/f.name)
        beams[kind]=json.loads((source/'summary.json').read_text())
        for level in (1,2,4):
            name=f'{kind}_r{level}';source=a.source/name;target=a.output/name;target.mkdir()
            for f in source.iterdir():
                if f.suffix in ('.json','.csv','.npz','.vtu','.inp','.log','.sta'):shutil.copyfile(f,target/f.name)
            result,h=weighted_history(source,target);results[name]=result;histories[name]=h
            rows.append([name,kind,level,.5/level,result['geometry']['elements'],result['geometry']['initial_solid_volume_mm3'],
                         result['peak_quarter_force_N'],result['peak_neck_nominal_stress_MPa'],
                         result['residual_grip_compression_mm'],result['peak_peeq'],
                         result['compression_work_relative_error'],result['cycle_work_relative_error'],all(result['checks'].values())])
    with (a.output/'comparison.csv').open('w') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['case','element_type','refinement_factor','neck_edge_mm','elements','solid_volume_mm3',
            'peak_quarter_force_N','peak_neck_nominal_stress_MPa','residual_area_mean_grip_compression_mm','peak_peeq',
            'compression_work_relative_error','cycle_work_relative_error','cycle_checks_passed']);w.writerows(rows)
    comparisons={}
    for kind in ('C3D8','C3D8I'):
        for x,y in [(1,2),(2,4)]:
            coarse=results[f'{kind}_r{x}'];fine=results[f'{kind}_r{y}']
            force=abs(coarse['peak_quarter_force_N']/fine['peak_quarter_force_N']-1)
            residual=abs(coarse['residual_grip_compression_mm']/fine['residual_grip_compression_mm']-1)
            comparisons[f'{kind}_{x}_to_{y}']={'peak_force_relative_change':force,'residual_displacement_relative_change':residual,
                'within_2_percent_response':force<.02 and residual<.02}
    report={'component_cases':results,'beam_cases':beams,'refinement_comparisons':comparisons,
            'response_criterion':.02,'relative_change_denominator':'finer case',
            'geometry_volume_preserved':all(abs(r['geometry']['initial_solid_volume_mm3']-238)<1e-10 for r in results.values()),
            'limitations':['fixed stepped isolated quarter component, not a periodic RVE',
                           'shoulder mesh remains coarse','pointwise plastic strain not a mesh-independent quantity',
                           'no self-contact, damage, or material calibration','100 mm / 0.1 MPa not run',
                           'C3D8I internal gradient audit not implemented; boundary work and solver ELSE compared']}
    (a.output/'verification-summary.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for kind in ('C3D8','C3D8I'):
        edge=[];force=[];residual=[]
        for level in (1,2,4):
            name=f'{kind}_r{level}';h=histories[name];r=results[name]
            axes[0,0].plot(h['grip_compression_mm'],h['quarter_force_N'],label=name)
            edge.append(.5/level);force.append(r['peak_quarter_force_N']);residual.append(r['residual_grip_compression_mm'])
        axes[0,1].plot(edge,force,'o-',label=kind)
        axes[1,0].plot(edge,residual,'o-',label=kind)
    axes[0,0].set(xlabel='Area mean grip compression [mm]',ylabel='Quarter-specimen force [N]',title='Compression and release');axes[0,0].legend(fontsize=8)
    axes[0,1].set(xlabel='Neck element edge [mm]',ylabel='Peak quarter force [N]',title='Same stepped geometry, volume = 238 mm^3');axes[0,1].invert_xaxis();axes[0,1].legend()
    axes[1,0].set(xlabel='Neck element edge [mm]',ylabel='Residual grip compression [mm]',title='Area-weighted residual displacement');axes[1,0].invert_xaxis();axes[1,0].legend()
    reference=beams['C3D8I']['timoshenko_reference_mm']
    axes[1,1].bar(['C3D8','C3D8I'],[beams[k]['tip_displacement_mm'] for k in ('C3D8','C3D8I')])
    axes[1,1].axhline(reference,color='black',ls='--',label='Timoshenko reference');axes[1,1].set(ylabel='Tip displacement [mm]',title='Independent slender beam');axes[1,1].legend()
    fig.suptitle('Fixed-shape neck component | uncalibrated plasticity | not a packing pressure prediction')
    fig.tight_layout();fig.savefig(a.output/'neck-comparison.png',dpi=180);plt.close(fig)
    from matplotlib.collections import PolyCollection
    fig,mesh_axes=plt.subplots(1,3,figsize=(10,8),sharey=True)
    for ax,level in zip(mesh_axes,(1,2,4)):
        with np.load(a.source/f'C3D8_r{level}'/'geometry.npz') as g:
            points=g['points'];elements=g['elements']
        surface=elements[:,[0,1,5,4]]
        surface=surface[np.all(points[surface,1]==0,axis=1)]
        polygons=points[surface][:,:,[0,2]]
        ax.add_collection(PolyCollection(polygons,facecolors='#b2d4e5',edgecolors='#284553',linewidths=.25))
        ax.set(xlim=(0,2.5),ylim=(9,15),aspect='equal',xlabel='X [mm]',title=f'Neck edge = {.5/level:g} mm')
    mesh_axes[0].set_ylabel('Z [mm]')
    fig.suptitle('Exact reference meshes at Y=0 | unchanged stepped neck boundary | cropped view')
    fig.tight_layout();fig.savefig(a.output/'neck-mesh-section.png',dpi=200);plt.close(fig)
    print(json.dumps(comparisons,indent=2))

if __name__=='__main__':main()
