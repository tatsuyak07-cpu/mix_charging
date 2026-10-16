"""Verification operators only; NOT enabled in the compression solver.

Periodic solid-side classification uses tetrahedral containment, not winding
number on an open cut surface. Probe is an independent point, not a mesh node;
self-contact neighbourhood exclusion and edge-edge contact are not provided.
"""
from itertools import product
from math import prod
import numpy as np
from .contact_geometry import closest_triangle
from .periodic_contact import Contact
from .explicit_periodic import Periodic


class PeriodicSolidProbe:
    def __init__(self, reference, tets, length, stiffness, skin):
        if stiffness<=0 or skin<=0:raise ValueError('Positive stiffness and search skin required')
        self.X=np.asarray(reference,float);self.t=np.asarray(tets,int)
        self.L=length;self.k=stiffness;self.skin=skin
        # Reuse exact opposite-face matching; paired cell caps are excluded.
        self.faces=Contact(self.X,self.t,Periodic(self.X,length),1.,0.,skin).faces

    def evaluate(self,point,x,H):
        point=np.asarray(point,float);x=np.asarray(x,float);H=np.asarray(H,float)
        if not np.all(np.isfinite(H)) or np.linalg.det(H)<=0:raise ValueError('Positive finite periodic cell required')
        inv=np.linalg.inv(H);frac=x@inv.T;pf=inv@point
        pad=self.skin*np.linalg.norm(inv,axis=1)
        lo=np.ceil(pf-frac.max(axis=0)-pad).astype(int)
        hi=np.floor(pf-frac.min(axis=0)+pad).astype(int)
        count=prod(int(v) for v in np.maximum(hi-lo+1,0))
        if count>729:raise ValueError('Periodic image budget exceeded; no truncation')
        shifts=list(product(*[range(a,b+1) for a,b in zip(lo,hi)]))
        tet=x[self.t];D=np.stack([tet[:,i]-tet[:,0] for i in (1,2,3)],axis=-1)
        if np.any(np.linalg.det(D)<=0):raise ValueError('Nonpositive tetrahedron')
        inverse=np.linalg.inv(D);inside=False
        for shift in shifts:
            uv=np.einsum('eij,ej->ei',inverse,point-H@shift-tet[:,0])
            weights=np.c_[1-uv.sum(axis=1),uv]
            if np.any(np.all(weights>=-1e-12,axis=1)):inside=True;break
        result=dict(inside=inside,energy=0.,force=np.zeros(3),master_force=np.zeros_like(x),
                    image_force_tensor=np.zeros((3,3)),image_count=count,depth_mm=0.)
        if not inside:return result
        choices=[]
        for shift in shifts:
            for face,ids in enumerate(self.faces):
                q,w=closest_triangle(point,x[ids]+H@shift)
                choices.append((float(np.sum((q-point)**2)),face,shift,q,w))
        if not choices:raise ValueError('Inside solid but no physical free surface')
        d2,face,shift,q,w=min(choices,key=lambda item:(item[0],item[1],item[2]))
        if np.sqrt(d2)>self.skin:raise ValueError('Inside solid beyond probe search band')
        force=self.k*(q-point);master=np.zeros_like(x)
        np.add.at(master,self.faces[face],-w[:,None]*force)
        result.update(energy=.5*self.k*d2,force=force,master_force=master,
            image_force_tensor=np.outer(-force,np.asarray(shift)*self.L),
            depth_mm=float(np.sqrt(d2)),face=face,shift=shift,weights=w,closest=q)
        return result


def friction_return(spring,old_normal,normal,increment,normal_force,stiffness,mu,rotation=None):
    """Transport a known contact patch's spring and apply Coulomb return.

    Coordinates mm, forces N, stiffness N/mm, energies N mm. Caller must supply
    material rotation when surface spin matters; normal alignment alone cannot
    recover spin about the normal. Automatic patch identification is NOT solved.
    """
    if not np.isfinite([normal_force,stiffness,mu]).all() or normal_force<0 or stiffness<=0 or mu<0:
        raise ValueError('Invalid friction parameters')
    n0=np.asarray(old_normal,float);n=np.asarray(normal,float);spring=np.asarray(spring,float)
    du=np.asarray(increment,float)
    if not np.isfinite(np.r_[n0,n,spring,du]).all():raise ValueError('Nonfinite friction state')
    if not np.isclose(n0@n0,1.,atol=1e-10,rtol=0) or not np.isclose(n@n,1.,atol=1e-10,rtol=0):raise ValueError('Unit normals required')
    if abs(spring@n0)>1e-10*max(np.linalg.norm(spring),1.):raise ValueError('Spring must be tangential')
    if rotation is None:
        c=float(np.clip(n0@n,-1,1));v=np.cross(n0,n)
        if c<=-1+1e-10:raise ValueError('Antipodal normals need explicit material rotation')
        W=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        rotation=np.eye(3)+W+W@W/(1+c)
    R=np.asarray(rotation,float)
    if not np.isfinite(R).all() or not np.allclose(R.T@R,np.eye(3),atol=1e-10,rtol=0) or not np.isclose(np.linalg.det(R),1.,atol=1e-10,rtol=0) or not np.allclose(R@n0,n,atol=1e-10,rtol=0):
        raise ValueError('Rotation must be proper and map old to new normal')
    old=R@spring;du=du-(du@n)*n;trial=old+du
    cap=mu*normal_force/stiffness;size=np.linalg.norm(trial)
    new=trial*min(1.,cap/max(size,1e-300));plastic=trial-new
    force=-stiffness*new
    # Backward-Euler work = elastic increment + friction + algorithmic loss.
    elastic_change=.5*stiffness*(new@new-old@old)
    dissipation=float(stiffness*new@plastic)
    algorithmic=.5*stiffness*float((new-old)@(new-old))
    work=float(-force@du)
    trapezoidal_work=.5*stiffness*float((old+new)@du)
    # Velocity Verlet uses endpoint forces, not backward-Euler work.
    # Keep the signed quadrature correction separate from friction dissipation.
    trapezoidal_correction=trapezoidal_work-elastic_change-dissipation
    return dict(spring=new,force=force,energy=.5*stiffness*float(new@new),
                elastic_change=float(elastic_change),friction_loss=dissipation,
                algorithmic_loss=algorithmic,work=work,
                balance_residual=float(work-elastic_change-dissipation-algorithmic),
                trapezoidal_work=trapezoidal_work,trapezoidal_correction=trapezoidal_correction)
