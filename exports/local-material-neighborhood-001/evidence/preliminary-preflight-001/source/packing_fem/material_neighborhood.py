"""Conservative short material-bridge exclusion for area contact.

A fixed reference predicate identifies short primitive pairs connected through
solid, not void. Every active trial must still have a material-covered closest
segment. Loss of that witness is unsupported local folding: stop, never silently
exclude it. This is not a general collision/injectivity certificate.
"""
from itertools import product
import numpy as np
from scipy.spatial import cKDTree
from .barrier_contact import feature


def segment_covered(a,b,vertices,tolerance=1e-10):
    """Exact affine barycentric interval union, up to dimensionless roundoff."""
    if not len(vertices):return False
    D=np.stack([vertices[:,k]-vertices[:,0] for k in (1,2,3)],axis=2)
    try:inv=np.linalg.inv(D)
    except np.linalg.LinAlgError:return False
    values=[]
    for p in (a,b):
        v=np.einsum('nij,nj->ni',inv,p-vertices[:,0]);values.append(np.column_stack((1-v.sum(1),v)))
    w,dw=values[0],values[1]-values[0]
    lower=np.zeros(len(vertices));upper=np.ones(len(vertices));valid=np.ones(len(vertices),bool)
    for j in range(4):
        slope=dw[:,j];moving=abs(slope)>1e-14
        crossing=np.divide(-tolerance-w[:,j],slope,out=np.zeros(len(vertices)),where=moving)
        lower=np.where(moving&(slope>0),np.maximum(lower,crossing),lower)
        upper=np.where(moving&(slope<0),np.minimum(upper,crossing),upper)
        valid&=moving|(w[:,j]>=-tolerance)
    valid&=lower<=upper
    intervals=sorted(zip(lower[valid],upper[valid]));end=0.
    for lo,hi in intervals:
        if lo>end+1e-10:return False
        end=max(end,hi)
        if end>=1-1e-10:return True
    return False


class MaterialNeighborhood:
    def __init__(self,X,t,L,radius):
        self.X=X;self.t=np.array(t,copy=True);self.L=L;self.radius=radius;self.cache={}
        v=X[t];self.centers=v.mean(1);self.tree=cKDTree(self.centers);self.bound=float(np.linalg.norm(v-self.centers[:,None],axis=2).max())
        self.minimum=X.min(0);self.maximum=X.max(0)

    @staticmethod
    def endpoints(points,w):
        return w[w>0]@points[w>0],-w[w<0]@points[w<0]

    def witness(self,key,ids,kind,shift):
        if key in self.cache:return self.cache[key]
        start=1 if kind=='pt' else 2;points=self.X[ids].copy();points[start:]+=self.L*np.asarray(shift)
        _,w=feature(points,kind);a,b=self.endpoints(points,w)
        if np.linalg.norm(a-b)>self.radius:
            self.cache[key]=None;return None
        lo=np.ceil((np.minimum(a,b)-self.maximum-1e-10)/self.L).astype(int);hi=np.floor((np.maximum(a,b)-self.minimum+1e-10)/self.L).astype(int)
        indices=[];images=[]
        for image in product(*(range(i,j+1) for i,j in zip(lo,hi))):
            offset=self.L*np.array(image);middle=(a+b)/2-offset
            candidates=self.tree.query_ball_point(middle,self.bound+np.linalg.norm(a-b)/2+1e-10)
            for k in candidates:
                vertices=self.X[self.t[k]]+offset
                if np.any(vertices.max(0)<np.minimum(a,b)-1e-10) or np.any(vertices.min(0)>np.maximum(a,b)+1e-10):continue
                indices.append(k);images.append(image)
        vertices=self.X[self.t[indices]]+self.L*np.asarray(images).reshape(-1,1,3)
        result=(np.asarray(indices,int),np.asarray(images).reshape(-1,3)) if segment_covered(a,b,vertices) else None
        self.cache[key]=result;return result

    def local(self,key,ids,kind,shift,points,w,x,H):
        witness=self.witness(key,ids,kind,shift)
        if witness is None:return False
        indices,images=witness;a,b=self.endpoints(points,w)
        vertices=x[self.t[indices]]+(images@H.T)[:,None,:]
        if not segment_covered(a,b,vertices):
            from .barrier_contact import ContactResolutionError
            raise ContactResolutionError('Local material bridge lost under deformation; local folding needs refined contact treatment')
        return True
