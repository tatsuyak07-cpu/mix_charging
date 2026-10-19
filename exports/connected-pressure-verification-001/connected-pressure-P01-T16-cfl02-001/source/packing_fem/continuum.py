"""C3D8 quadrature in CalculiX IP order (x changes fastest, then y, then z).

Source: CalculiX manual C3D8 Figure 50 and src/gauss.f gauss3d2.
"""
from itertools import product
import numpy as np

SIGNS=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
GAUSS=np.array([q[::-1] for q in product((-1/np.sqrt(3),1/np.sqrt(3)),repeat=3)])
GRADIENTS=np.empty((8,8,3))
for k,q in enumerate(GAUSS):
    for axis in range(3):
        rest=[j for j in range(3) if j!=axis]
        GRADIENTS[k,:,axis]=SIGNS[:,axis]*np.prod(1+SIGNS[:,rest]*q[rest],axis=1)/8


def jacobians(points,elements):
    return np.einsum('eni,qnj->eqij',points[elements],GRADIENTS)


def deformation_gradients(reference,current,elements):
    a=jacobians(reference,elements)
    return jacobians(current,elements)@np.linalg.inv(a)


def stress_tensors(values):
    """CalculiX S11,S22,S33,S12,S13,S23 to symmetric tensor."""
    v=np.asarray(values);s=np.zeros(v.shape[:-1]+(3,3))
    for k,(i,j) in enumerate([(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]):
        s[...,i,j]=v[...,k];s[...,j,i]=v[...,k]
    return s
