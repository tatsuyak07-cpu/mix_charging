"""Experimental EE surface-stationarity screening, not a contact force law.

A closest pair of edge primitives can fail to be a local closest pair of the
incident surface triangles. Positive distance-decreasing directional derivatives
provide a geometric witness. Passing this test does not prove global separation.
"""
import numpy as np
from .barrier_contact import feature


class SurfaceStationarity:
    def __init__(self,X,faces,classes,L):
        self.X=X;self.faces=faces;self.classes=classes;self.L=L;self.incident={};self.cache={}
        for face in faces:
            for node in face:self.incident.setdefault(int(classes[node]),[]).append((int(node),face))

    def star(self,ids,image):
        key=(*map(int,ids),*map(int,image))
        if key in self.cache:return self.cache[key]
        first=int(ids[0]);rows=[]
        for anchor,face in self.incident.get(int(self.classes[first]),[]):
            delta=(self.X[first]-self.X[anchor])/self.L
            lift=np.rint(delta).astype(int)
            if not np.allclose(delta,lift,atol=1e-10,rtol=0):continue
            if anchor!=first and not np.any(lift):continue
            if len(ids)==2 and not any(np.linalg.norm(self.X[j]+self.L*lift-self.X[ids[1]])<self.L*1e-10 for j in face):continue
            rows.append((face,np.asarray(image)+lift))
        self.cache[key]=rows;return rows

    def classify(self,ids,shift,x,H,tolerance=1e-10):
        points=x[ids].copy();points[2:]+=H@shift
        _,w=feature(points,'ee');a=w[:2]@points[:2];b=-w[2:]@points[2:];distance=float(np.linalg.norm(a-b))
        result=dict(distance_mm=distance,dominated=False,certified=False,maximum_descent_cosine=None,witness=None)
        if distance<=self.L*1e-12:return result
        scores=[];complete=True
        for side,(part,weights,q,toward,image) in enumerate(((ids[:2],w[:2],a,b-a,np.zeros(3,int)),(ids[2:],-w[2:],b,a-b,np.asarray(shift)))):
            support=part[weights>1e-12];star=self.star(support,image)
            if not star:complete=False;continue
            for face,lift in star:
                vertices=x[face]+H@lift;v=vertices-q;length=np.linalg.norm(v,axis=1)
                for k in np.flatnonzero(length>self.L*1e-12):
                    score=float(toward@v[k]/(distance*length[k]));scores.append(score)
                    if score>tolerance and (result['witness'] is None or score>result['witness']['descent_cosine']):
                        result['witness']=dict(side=side,face=face.tolist(),image=lift.tolist(),vertex=int(face[k]),point=q.tolist(),direction=v[k].tolist(),toward=toward.tolist(),descent_cosine=score)
        result.update(certified=complete,maximum_descent_cosine=max(scores) if scores else None)
        result['dominated']=complete and result['witness'] is not None
        return result
