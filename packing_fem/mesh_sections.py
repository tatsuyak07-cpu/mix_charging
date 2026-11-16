"""Area of a T4 solid cut by a bounded plane; no voxel approximation."""
import itertools
import numpy as np


def section_area(points,tets,center,normal,halfwidth):
    normal=np.asarray(normal,dtype=float);normal=normal/np.linalg.norm(normal)
    seed=np.eye(3)[np.argmin(abs(normal))];u=np.cross(normal,seed);u/=np.linalg.norm(u);v=np.cross(normal,u)
    relative=np.asarray(points)-center;distance=relative@normal
    uv=np.column_stack((relative@u,relative@v));signed=distance[tets];coordinates=uv[tets]
    selected=(signed.min(axis=1)<=0)&(signed.max(axis=1)>=0)
    selected&=np.all((coordinates.min(axis=1)<=halfwidth)&(coordinates.max(axis=1)>=-halfwidth),axis=1)
    area=0.;polygons=0
    for ds,xy in zip(signed[selected],coordinates[selected]):
        cut=[]
        for i,j in itertools.combinations(range(4),2):
            if ds[i]==0:cut.append(xy[i])
            if ds[j]==0:cut.append(xy[j])
            if ds[i]*ds[j]<0:cut.append((ds[j]*xy[i]-ds[i]*xy[j])/(ds[j]-ds[i]))
        if len(cut)<3:continue
        cut=np.unique(cut,axis=0)
        if len(cut)<3:continue
        midpoint=cut.mean(axis=0);delta=cut-midpoint
        polygon=list(cut[np.argsort(np.arctan2(delta[:,1],delta[:,0]))])
        for axis,sign in itertools.product(range(2),[-1,1]):
            clipped=[]
            if not polygon:break
            for a,b in zip(polygon,polygon[1:]+polygon[:1]):
                da=sign*a[axis]-halfwidth;db=sign*b[axis]-halfwidth
                if da<=0:clipped.append(a)
                if (da<0 and db>0) or (da>0 and db<0):clipped.append((db*a-da*b)/(db-da))
            polygon=clipped
        if len(polygon)>=3:
            polygon=np.array(polygon);next_point=np.roll(polygon,-1,axis=0)
            area+=.5*abs(np.sum(polygon[:,0]*next_point[:,1]-polygon[:,1]*next_point[:,0]));polygons+=1
    # Exactly coincident internal faces would be counted twice. The caller
    # must use a noncoincident section; reject instead of silently doubling it.
    faces_on_plane=np.sum(signed==0,axis=1)>=3
    if np.any(faces_on_plane):raise ValueError('Section coincides with a tetrahedron face; choose a noncoincident plane')
    return float(area),polygons
