"""Persistent material-node quadrature during conforming surface refinement.

Geometry vertices are not automatically new integration points. Retained
material points keep their reference-area weights and friction states. This
adapter supports retained-node bisection, not arbitrary node removal/ALE.
"""
from copy import deepcopy
import numpy as np


def remap_anchor(old,new,state,x,H):
    shift=np.asarray(state['anchor_shift']);original=old.X[state['anchor_ids']]+old.L*shift
    reference=state['bary']@original
    current=state['bary']@(x[state['anchor_ids']]+H@shift)
    triangles=new.X[new.faces]+new.L*shift
    E=np.column_stack((original[1]-original[0],original[2]-original[0]))
    uv=(triangles-original[0])@np.linalg.pinv(E).T
    projected=original[0]+uv@E.T
    contained=(np.max(abs(projected-triangles),axis=(1,2))<old.L*1e-9)&np.all(uv>=-1e-9,axis=(1,2))&np.all(uv.sum(2)<=1+1e-9,axis=1)
    u=triangles[:,1]-triangles[:,0];v=triangles[:,2]-triangles[:,0];r=reference-triangles[:,0]
    uu=np.einsum('ij,ij->i',u,u);vv=np.einsum('ij,ij->i',v,v);uvv=np.einsum('ij,ij->i',u,v)
    ru=np.einsum('ij,ij->i',r,u);rv=np.einsum('ij,ij->i',r,v);det=uu*vv-uvv*uvv
    a=(vv*ru-uvv*rv)/det;b=(uu*rv-uvv*ru)/det;weights=np.c_[1-a-b,a,b]
    candidates=np.flatnonzero(contained&np.all(weights>=-1e-9,axis=1))
    for face in candidates:
        bary=weights[face];ids=new.faces[face]
        if np.max(abs(bary@triangles[face]-reference))>old.L*1e-9:continue
        if np.max(abs(bary@(x[ids]+H@shift)-current))>old.L*1e-9:continue
        return ids.copy(),shift.copy(),bary.copy()
    raise ValueError('Material friction anchor has no unchanged reference/current representation')


def transfer(old,new,proposal,H):
    n=len(old.X);supports=proposal['supports']
    if old.quadrature_mode!='material' or new.quadrature_mode!='material':raise ValueError('Material quadrature mode must agree across transfer')
    if len(new.X)<n or not np.array_equal(supports[:n],np.repeat(np.arange(n)[:,None],2,axis=1)) or not np.array_equal(new.X[:n],old.X):
        raise ValueError('Material quadrature requires retained material source nodes')
    new.quadrature_weights=np.r_[old.integration_area0.copy(),np.zeros(len(new.X)-n)]
    x=proposal['x'];new.capture_audit=False
    fresh=new.evaluate(x,x*0,H,H*0,0.)[2]
    committed=map_history(old,new,old.history,fresh,x,H)
    force_history=old.accepted_force_history if old.accepted_force_history is not None else old.trial_force_history
    if not force_history:force_history=old.history
    applied=map_history(old,new,force_history,fresh,x,H)
    new.history=committed;new.accepted_force_history=applied
    return new.replay_force_state(x,x*0,H)


def map_history(old,new,history,fresh,x,H):
    mapped={};covered=set();n=len(old.X)
    for key,state in fresh.items():
        node=key[1]
        if node>=n:raise ValueError('Geometry-only node became a material integration point')
        matches=[]
        for oldkey,value in history.items():
            if oldkey[1]!=node:continue
            oldtri=old.X[old.faces[value['face']]]+old.L*np.asarray(value['shift'])
            newtri=new.X[new.faces[state['face']]]+new.L*np.asarray(state['shift'])
            E=np.column_stack((oldtri[1]-oldtri[0],oldtri[2]-oldtri[0]));uv=np.linalg.lstsq(E,(newtri-oldtri[0]).T,rcond=None)[0].T
            coplanar=np.max(abs(oldtri[0]+uv@E.T-newtri))<old.L*1e-9
            overlaps=(np.all(uv>=-1e-9) and np.all(uv.sum(1)<=1+1e-9)) or np.min(np.linalg.norm(oldtri[:,None]-newtri[None,:],axis=2))<old.L*1e-9
            if coplanar and overlaps:matches.append((oldkey,value))
        if len(matches)!=1:raise ValueError('Material contact patch correspondence is not unique')
        oldkey,value=matches[0]
        if oldkey in covered:raise ValueError('Material contact patch split requires projection')
        kept=deepcopy(value);ids,shift,bary=remap_anchor(old,new,kept,x,H)
        kept.update(face=state['face'],shift=state['shift'],anchor_ids=ids,anchor_shift=shift,bary=bary)
        mapped[oldkey]=kept;covered.add(oldkey)
    if covered!=set(history):raise ValueError('Material contact history creation/loss requires explicit integration refinement')
    return mapped
