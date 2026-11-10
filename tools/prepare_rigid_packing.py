"""Geometry-only placement of a central cylinder in an existing random pore.

A periodic change of origin preserves every sphere and short sinter bond.
Cell length is calibrated before loading to40% matrix-region CAD porosity.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution,brentq,minimize_scalar
from packing_fem.random_packing import neck_added_volume


def cylinder_distance(delta,r,h):
    z=np.stack([np.linalg.norm(delta[...,:2],axis=-1)-r,abs(delta[...,2])-h/2],axis=-1)
    return np.linalg.norm(np.maximum(z,0),axis=-1)+np.minimum(z.max(axis=-1),0)


def prepare(config,source,out):
    out=Path(out);out.mkdir(exist_ok=False,parents=True);c=json.loads(Path(config).read_text());p=dict(np.load(source));f=p['fractional_centers'];bonds=p['bonds'];images=p['bond_images'];D=c['diameter_mm'];R=D/2;rneck=c['neck_diameter_mm']/2;r=c['rigid_diameter_mm']/2;h=c['rigid_height_mm'];vr=np.pi*r*r*h
    lengths=np.linalg.norm(f[bonds[:,1]]+images-f[bonds[:,0]],axis=1);sphere=len(f)*4*np.pi*R**3/3
    L=brentq(lambda L:.6*(L**3-vr)-sphere-neck_added_volume(L*lengths,R,rneck).sum(),float(p['length_mm']),float(p['length_mm'])*1.02)
    samples=np.linspace(0,1,41);axes=f[bonds[:,0],None,:]+samples[None,:,None]*(f[bonds[:,1]]+images-f[bonds[:,0]])[:,None,:]
    def clearance(center):
        delta=f-center;delta-=np.rint(delta);s=float((cylinder_distance(delta*L,r,h)-R).min())
        delta=axes-center;delta-=np.rint(delta);neck=float((cylinder_distance(delta*L,r,h)-rneck).min())
        return min(s,neck)
    result=differential_evolution(lambda v:-clearance(v),[(0,1)]*3,seed=c['rigid_placement_seed'],maxiter=400,popsize=12,tol=1e-7,polish=True)
    center=result.x;minimum_sphere=float((cylinder_distance((f-center-np.rint(f-center))*L,r,h)-R).min());minimum_neck=np.inf
    from itertools import product
    for (i,j),shift in zip(bonds,images):
        a=(f[i]-center)*L;v=(f[j]+shift-f[i])*L;near=-np.rint((a+v/2)/L)
        for extra in product([-1,0,1],repeat=3):
            A=a+(near+extra)*L
            if float(cylinder_distance(A+v/2,r,h))-np.linalg.norm(v)/2>rneck+minimum_sphere:continue
            fit=minimize_scalar(lambda t:float(cylinder_distance(A+t*v,r,h)),bounds=(0,1),method='bounded',options=dict(xatol=1e-12))
            minimum_neck=min(minimum_neck,float(fit.fun-rneck),float(cylinder_distance(A,r,h)-rneck),float(cylinder_distance(A+v,r,h)-rneck))
    report=dict(source=str(source),length_mm=L,rigid_diameter_mm=2*r,rigid_height_mm=h,placement_fractional=center.tolist(),minimum_sphere_clearance_mm=minimum_sphere,minimum_neck_clearance_mm=float(minimum_neck),matrix_cad_porosity=.4,geometry_only_selection=True,statistical_isotropy_verified=False)
    (out/'placement.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    if min(minimum_sphere,minimum_neck)<c.get('rigid_initial_clearance_mm',.025):raise ValueError('No cylinder clearance within the predeclared geometry search')
    moved=np.mod(f-center+.5,1.);vectors=f[bonds[:,1]]+images-f[bonds[:,0]];newimages=np.rint(vectors-(moved[bonds[:,1]]-moved[bonds[:,0]])).astype(int)
    np.savez_compressed(out/'packing.npz',fractional_centers=moved,bonds=bonds,bond_images=newimages,length_mm=L)
    solid=sphere+neck_added_volume(L*lengths,R,rneck).sum();c.update(length_mm=L,rigid_center_mm=[L/2]*3,requested_matrix_porosity=.4,requested_initial_porosity=float(1-solid/L**3))
    (out/'config.json').write_text(json.dumps(c,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--source',required=True);p.add_argument('--out',required=True);a=p.parse_args();prepare(a.config,a.source,a.out)
