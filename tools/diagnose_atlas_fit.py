"""Diagnose positive-fit feasibility; never advances or accepts a FEM state."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix,hstack,vstack
from packing_fem.contact_atlas import density,scales
from packing_fem.surface_patch import integrate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    feature=json.loads(a.summary.read_text())['failure_feature'];poses=feature['poses']+feature['controls'];area=feature['reference_area_mm2'];tol=feature['tolerance'];rows=[]
    for factor in (16,64):
        refs=[];pools=[]
        for pose in poses:
            r=integrate(np.asarray(pose['source']),np.asarray(pose['targets']),area,kn=feature['kn'],thickness=feature['thickness'],tolerance=tol/factor,vertex_map=np.asarray(pose['mapping']),groups=pose['groups'],pointwise=True,error_budget='normalized',absolute_energy_budget=feature['energy_budget']/factor,absolute_force_budget=feature['force_budget']/factor)
            refs.append(r['value']);pools.append(r['bary'])
        for augmented in (False,True):
            bary=np.concatenate(pools)
            if augmented:
                vertices=[]
                for i in range(3):
                    for epsilon in (1e-10,1e-8,1e-6,1e-4,.01):
                        for ratio in (.25,.75):
                            point=np.zeros(3);point[i]=1-epsilon;point[(i+1)%3]=epsilon*ratio;point[(i+2)%3]=epsilon*(1-ratio);vertices.append(point)
                bary=np.r_[bary,vertices]
            bary=np.unique(bary,axis=0);C=np.array([np.ones(len(bary)),bary[:,0],bary[:,1],bary[:,0]**2,bary[:,0]*bary[:,1],bary[:,1]**2]);moments=np.array([1,1/3,1/3,1/6,1/12,1/6]);M=[];target=[]
            for pose,value in zip(poses,refs):
                scale=scales(value,feature['energy_budget'],feature['force_budget'],tol);M.extend((area*density(bary,pose,feature['kn'],feature['thickness'])/scale).T);target.extend(value/scale)
            M=np.asarray(M);target=np.asarray(target);size=len(bary);response=csr_matrix(M);slack=csr_matrix(-np.ones((len(M),1)))
            result=linprog(np.r_[np.zeros(size),1.],A_eq=hstack([csr_matrix(C),csr_matrix((6,1))]),b_eq=moments,A_ub=vstack([hstack([response,slack]),hstack([-response,slack])]),b_ub=np.r_[target,-target],bounds=(0,None),options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10,'time_limit':60.})
            row=dict(reference_factor=factor,augmented=augmented,candidates=size,success=result.success,message=result.message,minimum_normalized_residual=float(result.x[-1]) if result.success else None,unchanged_fit_acceptance=tol/8)
            rows.append(row);print(json.dumps(row),flush=True)
    (a.output/'summary.json').write_text(json.dumps(dict(scope=__doc__,cases=rows),indent=2)+'\n')


if __name__=='__main__':main()
