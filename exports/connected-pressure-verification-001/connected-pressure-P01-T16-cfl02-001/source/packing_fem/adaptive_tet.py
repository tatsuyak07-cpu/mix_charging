"""Boundary-preserving local 2-to-3 tetrahedron reconnection.

Conservative cavity reference-volume transfer; uniform state is reproduced.
Nonuniform state is projected in elastic logarithmic strain and equivalent
plastic strain. Energy error is checked before acceptance. This is not a
surface regenerator or a guarantee of recovery from arbitrary mesh distortion.
"""
from itertools import combinations
import numpy as np
from .explicit_material import geometry,oriented,response,spectral_log

FACES=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])


def boundary(t):
    faces=t[:,FACES].reshape(-1,3);keys=np.sort(faces,axis=1)
    _,inv,count=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
    if np.any(count>2):raise ValueError('Nonmanifold tetrahedron faces')
    return faces[count[inv]==1]


def closed_surface_volume(x,t):
    triangles=x[boundary(t)]
    return float(np.einsum('ni,ni->n',triangles[:,0],np.cross(triangles[:,1],triangles[:,2])).sum()/6)


def quality(x,t):
    v,_,_=geometry(x,t);length2=sum(np.sum((x[t[:,i]]-x[t[:,j]])**2,axis=1) for i,j in combinations(range(4),2))
    return 12*(3*v)**(2/3)/length2


def propose(x,t,b,ep,V0,c,threshold=.35,energy_tolerance=1e-3):
    q=quality(x,t);faces=t[:,FACES].reshape(-1,3);keys=np.sort(faces,axis=1)
    _,inv,count=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
    owners={}
    for k,key in enumerate(inv):
        if count[key]==2:owners.setdefault(int(key),[]).append(k//4)
    rejected=0
    for key,pair in sorted(owners.items(),key=lambda item:min(q[item[1]])):
        if min(q[pair])>=threshold:break
        a,d=pair;shared=np.intersect1d(t[a],t[d]);u=np.setdiff1d(t[a],shared)[0];v=np.setdiff1d(t[d],shared)[0]
        if any(u in row and v in row for row in t):continue
        trial=np.array([[u,v,shared[0],shared[1]],[u,v,shared[1],shared[2]],[u,v,shared[2],shared[0]]])
        try:trial=oriented(x,trial);vol,_,_=geometry(x,trial)
        except ValueError:continue
        oldvol,_,_=geometry(x,t[pair])
        if not np.isclose(vol.sum(),oldvol.sum(),rtol=1e-10,atol=1e-12):continue
        if set(map(tuple,np.sort(boundary(t[pair]),axis=1)))!=set(map(tuple,np.sort(boundary(trial),axis=1))):continue
        if quality(x,trial).min()<=min(q[pair])*1.05:continue
        total=V0[pair].sum();weights=V0[pair]/total
        e,Q=spectral_log(b[pair]);logs=np.einsum('eik,ek,ejk->eij',Q,e,Q);mean=np.einsum('e,eij->ij',weights,logs)
        target_logJ=np.log(vol.sum()/total);mean+=np.eye(3)*(target_logJ-np.trace(mean))/3
        values,vectors=np.linalg.eigh(mean);mapped=vectors@np.diag(np.exp(2*values))@vectors.T
        newb=np.repeat(mapped[None],3,axis=0);newep=np.full(3,float(weights@ep[pair]));newV0=total*vol/vol.sum()
        oldenergy=float(V0[pair]@response(b[pair],ep[pair],c)[1]);newenergy=float(newV0@response(newb,newep,c)[1])
        scale=max(abs(oldenergy),c['young_mpa']*total*1e-12)
        error=abs(newenergy-oldenergy)/scale
        if error>energy_tolerance:rejected+=1;continue
        keep=np.ones(len(t),bool);keep[pair]=False
        report={'operation':'2_to_3','removed_elements':pair,'quality_before':float(min(q[pair])),
                'quality_after':float(quality(x,trial).min()),'reference_volume_relative_error':float(abs(newV0.sum()-total)/total),
                'current_volume_relative_error':float(abs(vol.sum()-oldvol.sum())/oldvol.sum()),'material_energy_relative_error':error,
                'previous_rejected_state_transfers':rejected,'boundary_unchanged':True}
        return np.r_[t[keep],trial],np.r_[b[keep],newb],np.r_[ep[keep],newep],np.r_[V0[keep],newV0],report
    return None
