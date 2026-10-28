"""Measure voxel-volume variation under uniform periodic translations.

Particle radii, relative positions and periodic bond topology remain fixed.
This is an initial-geometry discretization choice, not a convergence remedy.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc
from packing_fem.sintered import draw


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,default=256);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    c=json.loads((a.source/'config.json').read_text())
    with np.load(a.source/'sintered-geometry.npz') as g:x=g['centers']
    L=c['length_mm'];n=int(np.ceil(L/c['mesh_mm']));h=L/n;target=int(round((1-c['porosity'])*n**3))
    _,neighbors=cKDTree(x,boxsize=L).query(x,k=min(len(x),c['neck_neighbors']+1))
    bonds=sorted({tuple(sorted((i,int(j)))) for i,row in enumerate(neighbors) for j in row if i!=j})
    records=[];best=None
    for phase in np.r_[np.zeros((1,3)),qmc.Halton(3,scramble=False).random(a.samples)]*h:
        y=(x+phase)%L;mask=np.zeros((n,n,n),bool)
        for center in y:draw(mask,center,center,c['diameter_mm']/2,h)
        for i,j in bonds:
            delta=y[j]-y[i];delta-=L*np.rint(delta/L);draw(mask,y[i],y[i]+delta,c['neck_diameter_mm']/2,h)
        count=int(mask.sum());records.append({'phase_mm':phase.tolist(),'solid_elements':count})
        if best is None or abs(count-target)<best[0]:best=(abs(count-target),phase.copy(),mask.copy(),y.copy())
        if count==target:break
    report={'purpose':__doc__,'source':str(a.source),'target_solid_voxels':target,'selected_phase_mm':best[1].tolist(),'actual_porosity':1-float(best[2].mean()),'trials':records}
    (a.output/'phase-scan.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(a.output/'selected-phase-geometry.npz',mask=best[2],centers=best[3],phase_mm=best[1])
    print(json.dumps({k:v for k,v in report.items() if k!='trials'},indent=2))

if __name__=='__main__':main()
