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
            if any(not any(np.linalg.norm(self.X[j]+self.L*lift-self.X[node])<self.L*1e-10 for j in face) for node in ids[1:]):continue
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
                    # The pair already minimizes distance over each original
                    # segment. Motion toward either of its endpoints cannot
                    # be a descent witness, including an endpoint minimizer.
                    # Near zero separation, subtracting world coordinates can
                    # manufacture a positive cosine along this exact tangent.
                    ref=self.X[face[k]]+self.L*lift
                    if np.any(np.linalg.norm(self.X[part]+self.L*image-ref,axis=1)<self.L*1e-10):continue
                    score=float(toward@v[k]/(distance*length[k]));scores.append(score)
                    if score>tolerance and (result['witness'] is None or score>result['witness']['descent_cosine']):
                        result['witness']=dict(side=side,face=face.tolist(),image=lift.tolist(),vertex=int(face[k]),point=q.tolist(),direction=v[k].tolist(),toward=toward.tolist(),descent_cosine=score)
        result.update(certified=complete,maximum_descent_cosine=max(scores) if scores else None)
        result['dominated']=complete and result['witness'] is not None
        return result


    def classify_point(self,ids,shift,x,H,tolerance=1e-10):
        """Diagnostic PT derivatives; source motion NEVER authorizes exclusion.

        The source is an integration node, not an unknown closest-surface point.
        A valid distributed pressure may have a source distance-decreasing
        direction. Keep the two sides separate and leave the force law alone.
        """
        points=x[ids].copy();points[1:]+=H@shift
        _,w=feature(points,'pt');a=points[0];b=-w[1:]@points[1:];distance=float(np.linalg.norm(a-b))
        result=dict(distance_mm=distance,source_is_quadrature_node=True,candidate_exclusion_authorized=False)
        for name,part,weights,q,toward,image in (
            ('source',ids[:1],np.ones(1),a,b-a,np.zeros(3,int)),
            ('target',ids[1:],-w[1:],b,a-b,np.asarray(shift))):
            side=dict(complete_star=False,has_descent=False,maximum_descent_cosine=None,witness=None)
            if distance>self.L*1e-12:
                star=self.star(part[weights>1e-12],image);scores=[]
                for face,lift in star:
                    v=x[face]+H@lift-q;length=np.linalg.norm(v,axis=1)
                    for k in np.flatnonzero(length>self.L*1e-12):
                        score=float(toward@v[k]/(distance*length[k]));scores.append(score)
                        if score>tolerance and (side['witness'] is None or score>side['witness']['descent_cosine']):
                            side['witness']=dict(face=face.tolist(),image=lift.tolist(),vertex=int(face[k]),point=q.tolist(),direction=v[k].tolist(),toward=toward.tolist(),descent_cosine=score)
                side.update(complete_star=bool(star),has_descent=bool(star) and side['witness'] is not None,maximum_descent_cosine=max(scores) if scores else None)
            result[name]=side
        return result
