"""Conservative witnesses that selected point quadrature resolves an EE pair.

An endpoint contact elsewhere is insufficient. Accept a closest-endpoint
witness on the opposing feature, or a common connected coplanar patch with
the same gap along the source edge. This adds no force or contact energy.
"""
from collections import deque
import numpy as np
from .barrier_contact import feature
from .surface_stationarity import SurfaceStationarity


class EdgeCoverage:
    def __init__(self,contact):
        self.contact=contact;self.L=contact.L;self.tol=self.L*1e-10
        self.star=SurfaceStationarity(contact.X,contact.faces,contact.classes,self.L)
        self.face_ids={tuple(sorted(map(int,f))):i for i,f in enumerate(contact.faces)}
        self.cache={}

    def component(self,face,image,x,H,allowed):
        key=(int(face),*map(int,image))
        if key in self.cache:return self.cache[key]
        c=self.contact;triangle=x[c.faces[face]]+H@image
        normal=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]);length=np.linalg.norm(normal)
        if length==0:return set()
        normal/=length;queue=deque([(int(face),tuple(image))]);visited=set();result=set()
        while queue:
            f,lift=queue.popleft();item=(f,*lift)
            if item in visited:continue
            visited.add(item)
            if lift not in allowed:continue
            vertices=x[c.faces[f]]+H@np.array(lift)
            if np.max(abs((vertices-triangle[0])@normal))>self.tol:continue
            result.add(item);keys=c.target_vertices(f,lift)
            for node in c.faces[f]:
                for other,other_lift in self.star.star(np.array([node]),np.array(lift)):
                    j=self.face_ids[tuple(sorted(map(int,other)))];other_image=tuple(map(int,other_lift))
                    if other_image in allowed and len(keys.intersection(c.target_vertices(j,other_image)))>=2:
                        queue.append((j,other_image))
        self.cache[key]=result
        return result

    def resolved(self,ids,shift,points,distance,point_candidates,x,H,shifts):
        c=self.contact;_,w=feature(points,'ee')
        # A decreasing direction inside an incident triangle proves the
        # primitive EE pair is not a closest surface pair. This is the same
        # geometric witness used by the optional stationarity screening.
        if self.star.classify(ids,shift,x,H)['dominated']:return 'surface_descent'
        for side in (0,1):
            part=ids[2*side:2*side+2];other=ids[2*(1-side):2*(1-side)+2]
            weights=w[:2] if side==0 else -w[2:]
            other_weights=-w[2:] if side==0 else w[:2]
            source_image=np.zeros(3,int) if side==0 else np.asarray(shift)
            target_image=np.asarray(shift) if side==0 else np.zeros(3,int)
            other_ref=c.X[other]+self.L*target_image
            for endpoint,node in enumerate(part):
                for candidate in point_candidates.get(int(node),[]):
                    d,key,ptids,ptshift,p,branch,pw,r=candidate
                    # Equal gaps are essential: a distant endpoint contact
                    # cannot stand in for a near-zero edge-interior approach.
                    if d>distance+self.tol:continue
                    image=np.asarray(ptshift)+source_image
                    ref=c.X[ptids[1:]]+self.L*image
                    closest_other=other_ref[other_weights>1e-12]
                    contains=all(np.any(np.linalg.norm(ref-v,axis=1)<self.tol) for v in closest_other)
                    if weights[endpoint]>1-1e-12 and contains:return 'closest_endpoint'
                    if abs(d-distance)>self.tol:continue
                    triangle=x[ptids[1:]]+H@image
                    normal=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]);norm=np.linalg.norm(normal)
                    if norm==0:continue
                    normal/=norm
                    source=points[2*side:2*side+2];target=points[2*(1-side):2*(1-side)+2]
                    heights=(source-triangle[0])@normal
                    if max(abs(abs(heights)-d))>self.tol or abs(heights[1]-heights[0])>self.tol:continue
                    if np.max(abs((target-triangle[0])@normal))>self.tol:continue
                    goals={(self.face_ids[tuple(sorted(map(int,f)))],*map(int,lift)) for f,lift in self.star.star(other,target_image)}
                    allowed={tuple(map(int,s)) for s in shifts}
                    allowed.update(tuple(map(int,np.asarray(s)+source_image)) for s in shifts)
                    allowed.add(tuple(map(int,image)))
                    component=self.component(key[2],image,x,H,allowed)
                    if component.intersection(goals):return 'connected_coplanar_patch'
        return None
