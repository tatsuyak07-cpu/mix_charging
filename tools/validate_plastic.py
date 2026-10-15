"""Run independent nonlinear verification cases on fixed archived sphere centers."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np
from scipy.spatial import cKDTree
from packing_fem import sintered,plastic
from packing_fem.energy import audit


def prepare_case(source,out,c):
    out.mkdir(parents=True,exist_ok=False)
    original=json.loads((source/'config.json').read_text())
    with np.load(source/'sintered-geometry.npz') as g:
        centers=g['centers'];p=g['points'];e=g['elements'];mask=g['mask']
    estimate=json.loads((source/'estimate.json').read_text())
    if c['mesh_mm']==original['mesh_mm']:
        shutil.copyfile(source/'sintered-geometry.npz',out/'sintered-geometry.npz')
    else:
        L=c['length_mm'];h=c['mesh_mm'];n=int(round(L/h));mask=np.zeros((n,n,n),bool)
        for center in centers:sintered.draw(mask,center,center,c['diameter_mm']/2,h)
        core=int(mask.sum())
        _,neighbors=cKDTree(centers,boxsize=L).query(centers,k=c['neck_neighbors']+1)
        bonds={tuple(sorted((i,int(j)))) for i,row in enumerate(neighbors) for j in row if i!=j}
        for i,j in sorted(bonds):
            delta=centers[j]-centers[i];delta-=L*np.rint(delta/L)
            sintered.draw(mask,centers[i],centers[i]+delta,c['neck_diameter_mm']/2,h)
        p,e,_=sintered.mesh(mask,h);component,rank=sintered.connectivity(mask)
        estimate.update(grid_cells_per_axis=n,voxel_mm=h,solid_elements=len(e),initial_porosity=1-mask.mean(),
                        porosity_error=abs(1-mask.mean()-c['porosity']),sphere_voxel_volume_mm3=core*h**3,
                        added_neck_voxel_volume_mm3=(mask.sum()-core)*h**3,periodic_components=component,
                        winding_rank=rank,cells_across_neck=c['neck_diameter_mm']/h,
                        nodes=len(p)+6,upper_bound_dofs=3*len(p)+6)
        np.savez_compressed(out/'sintered-geometry.npz',centers=centers,points=p,elements=e,mask=mask)
    (out/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    (out/'estimate.json').write_text(json.dumps(estimate,indent=2,default=lambda x:x.item())+'\n')
    plastic.write_model(out,c,p,e,np.rint(p/c['mesh_mm']).astype(int),estimate,True)
    (out/'provenance.json').write_text(json.dumps({'source':str(source),'same_centers':True,
        'geometry_sha256':hashlib.sha256((out/'sintered-geometry.npz').read_bytes()).hexdigest(),
        'input_sha256':hashlib.sha256((out/'plastic.inp').read_bytes()).hexdigest()},indent=2)+'\n')


def worker(source,out,c,timeout):
    try:
        prepare_case(source,out,c)
        result=plastic.run(out,'/workspace/environment/calculix/usr/bin/ccx',timeout)
        energy=audit(out,out/'energy-audit')
        return {'case':out.name,'result':result,'energy':energy}
    except Exception as exc:
        failure={'case':out.name,'error':repr(exc)}
        if out.exists():(out/'verification-error.json').write_text(json.dumps(failure,indent=2)+'\n')
        return failure


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('results/plastic-packing-001'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--timeout',type=float,default=1800)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    c=json.loads((a.source/'config.json').read_text());os.environ['OMP_NUM_THREADS']='1'
    variants=[('baseline',{}),('half_increment',{'initial_increment':.025,'max_increment':.05}),
              ('quarter_increment',{'initial_increment':.0125,'max_increment':.025}),
              ('yield_half',{'yield_mpa':.025}),('hardening_double',{'hardening_mpa':.4}),
              ('mesh_075mm',{'mesh_mm':.75})]
    results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending=[pool.submit(worker,a.source,a.output/name,{**c,**updates},a.timeout) for name,updates in variants]
        for f in as_completed(pending):
            results.append(f.result())
            print(json.dumps({'completed':results[-1]['case'],'error':results[-1].get('error')},indent=2),flush=True)
    (a.output/'verification-results.json').write_text(json.dumps(results,indent=2)+'\n')
    print('Verification cases retained:',a.output,flush=True)

if __name__=='__main__':main()
