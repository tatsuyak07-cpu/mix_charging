"""Boundary-preserving 3-to-2 / 4-to-4 cavity repairs with state audits.

No surface nodes or material contact samples are moved or recreated. These
operations supplement, rather than replace, general surface remeshing.
"""
from itertools import combinations
import numpy as np
from .adaptive_tet import boundary,quality,propose
from .explicit_material import geometry,oriented,spectral_log,response


def oriented_surface(t):
    return {min(tuple(np.roll(f,k)) for k in range(3)) for f in boundary(t)}


def project(x,t,b,ep,V0,owners,trial,c,tolerance):
    old=t[owners];trial=oriented(x,np.asarray(trial,int))
    if oriented_surface(old)!=oriented_surface(trial):return None
    oldvol=geometry(x,old)[0];vol=geometry(x,trial)[0]
    if not np.isclose(oldvol.sum(),vol.sum(),rtol=1e-11,atol=1e-13):return None
    before=float(quality(x,old).min());after=float(quality(x,trial).min())
    if after<=before*1.05:return None
    total=float(V0[owners].sum());weights=V0[owners]/total
    values,Q=spectral_log(b[owners]);logs=np.einsum('eik,ek,ejk->eij',Q,values,Q)
    mean=np.einsum('e,eij->ij',weights,logs)
    mean+=np.eye(3)*(np.log(vol.sum()/total)-np.trace(mean))/3
    eigen,vectors=np.linalg.eigh(mean);mapped=vectors@np.diag(np.exp(2*eigen))@vectors.T
    newb=np.repeat(mapped[None],len(trial),axis=0);newep=np.full(len(trial),float(weights@ep[owners]))
    newV0=total*vol/vol.sum()
    oldenergy=float(V0[owners]@response(b[owners],ep[owners],c)[1]);newenergy=float(newV0@response(newb,newep,c)[1])
    error=abs(newenergy-oldenergy)/max(abs(oldenergy),c['young_mpa']*total*1e-12)
    if error>tolerance:return None
    keep=np.ones(len(t),bool);keep[owners]=False
    combined=np.r_[t[keep],trial]
    boundary(combined)  # Reject a nonmanifold cavity attachment.
    if len(np.unique(np.sort(combined,axis=1),axis=0))!=len(combined):return None
    report=dict(operation=f'{len(owners)}_to_{len(trial)}',removed_elements=list(map(int,owners)),
                quality_before=before,quality_after=after,material_energy_relative_error=error,
                reference_volume_relative_error=abs(newV0.sum()-total)/total,
                current_volume_relative_error=abs(vol.sum()-oldvol.sum())/oldvol.sum(),boundary_unchanged=True)
    return combined,np.r_[b[keep],newb],np.r_[ep[keep],newep],np.r_[V0[keep],newV0],report


def propose_reconnection(x,t,b,ep,V0,c,threshold=.35,tolerance=.001):
    q=quality(x,t)
    if float(q.min())>=threshold:return None
    first=propose(x,t,b,ep,V0,c,threshold,tolerance)
    if first is not None:return first
    owners={}
    for i,tet in enumerate(t):
        for edge in combinations(sorted(map(int,tet)),2):owners.setdefault(edge,[]).append(i)
    for (u,v),ids in sorted(owners.items(),key=lambda row:min(q[row[1]])):
        if min(q[ids])>=threshold:break
        if len(ids) not in (3,4):continue
        link=[list(map(int,np.setdiff1d(t[i],[u,v]))) for i in ids]
        vertices=sorted(set(k for edge in link for k in edge))
        if len(vertices)!=len(ids):continue
        if len(ids)==3:
            a,d,e=vertices;trials=[[[a,d,e,u],[a,e,d,v]]]
        else:
            adjacent={k:[] for k in vertices}
            for a,d in link:adjacent[a].append(d);adjacent[d].append(a)
            if any(len(vs)!=2 for vs in adjacent.values()):continue
            a=vertices[0];d=min(adjacent[a]);e=next(k for k in adjacent[d] if k!=a);f=next(k for k in adjacent[e] if k!=d)
            if f not in adjacent[a]:continue
            trials=[]
            for p,r,s,w in ((a,e,d,f),(d,f,e,a)):
                trials.append([[p,r,u,s],[p,r,s,v],[p,r,v,w],[p,r,w,u]])
        for trial in trials:
            try:result=project(x,t,b,ep,V0,ids,trial,c,tolerance)
            except ValueError:continue
            if result is not None:return result
    return None
