"""Conforming eight-way T4 refinement preserving the faceted boundary exactly."""
import numpy as np
from .explicit_material import oriented


def refine(points,tets):
    pairs=np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
    edges,inverse=np.unique(np.sort(tets[:,pairs].reshape(-1,2),axis=1),axis=0,return_inverse=True)
    mid=inverse.reshape(-1,6)+len(points)
    nodes=np.vstack([points,points[edges].mean(axis=1)])
    a,b,c,d,e,f=mid.T;v0,v1,v2,v3=tets.T
    children=np.stack([np.c_[v0,a,b,c],np.c_[v1,a,d,e],np.c_[v2,b,d,f],np.c_[v3,c,e,f],
                       np.c_[a,f,b,c],np.c_[a,f,c,e],np.c_[a,f,e,d],np.c_[a,f,d,b]],axis=1).reshape(-1,4)
    return nodes,oriented(nodes,children),edges
