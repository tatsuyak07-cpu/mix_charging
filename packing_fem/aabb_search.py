"""Conservative primitive AABB hierarchy with bounded per-query allocation."""
import numpy as np


class AABBTree:
    def __init__(self,lower,upper,leaf_size=12):
        self.lower=np.asarray(lower,float);self.upper=np.asarray(upper,float)
        if self.lower.ndim!=2 or self.lower.shape[1]!=3 or self.upper.shape!=self.lower.shape or not len(self.lower):raise ValueError('Nonempty 3D boxes required')
        if not np.isfinite(self.lower).all() or not np.isfinite(self.upper).all() or np.any(self.lower>self.upper):raise ValueError('Invalid boxes')
        self.nodes=[]
        def build(ids):
            low=self.lower[ids].min(0);high=self.upper[ids].max(0);index=len(self.nodes);self.nodes.append(None)
            if len(ids)<=leaf_size:self.nodes[index]=(low,high,ids,None,None)
            else:
                centers=(self.lower[ids]+self.upper[ids])/2;axis=int(np.argmax(np.ptp(centers,axis=0)));order=ids[np.argsort(centers[:,axis],kind='stable')];half=len(order)//2
                left=build(order[:half]);right=build(order[half:]);self.nodes[index]=(low,high,None,left,right)
            return index
        build(np.arange(len(self.lower)))
        self.padding=128*np.finfo(float).eps*max(1.,float(abs(self.lower).max()),float(abs(self.upper).max()))

    def query(self,lower,upper,distance,max_results=None):
        """Return all boxes at Euclidean box distance <= distance + roundoff.

        Sorting retains the legacy image/primitive enumeration. The caller
        processes one query at a time; no global point-pair lists are built.
        """
        low=np.asarray(lower);high=np.asarray(upper);limit=(distance+self.padding)**2;stack=[0];found=[]
        while stack:
            index=stack.pop();a,b,ids,left,right=self.nodes[index]
            d=np.maximum(np.maximum(a-high,low-b),0.)
            if d@d>limit:continue
            if ids is None:stack.extend((right,left));continue
            delta=np.maximum(np.maximum(self.lower[ids]-high,low-self.upper[ids]),0.)
            hits=ids[np.einsum('ij,ij->i',delta,delta)<=limit]
            if max_results is not None and len(found)+len(hits)>max_results:
                raise ValueError('Contact AABB candidate budget exceeded during bounded query; no truncation')
            found.extend(hits)
        return sorted(map(int,found))
