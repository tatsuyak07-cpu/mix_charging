"""Conservative history matching on the lifted reference surface.

Transfers only across an identical triangle or a shared reference edge in the
same periodic covering. Vertex-only transitions are deliberately rejected.
"""
import numpy as np


def adjacent_patch(reference,faces,length,old_face,old_shift,face,shift):
    tolerance=length*1e-10
    def keys(f,s):
        return set(map(tuple,np.rint((reference[faces[f]]+np.asarray(s)*length)/tolerance).astype(np.int64)))
    return len(keys(old_face,old_shift)&keys(face,shift))>=2


def normal_rotation(a,b):
    a=np.asarray(a);b=np.asarray(b);c=float(np.clip(a@b,-1,1));v=np.cross(a,b)
    if c<=-1+1e-10:raise ValueError('Ambiguous antipodal contact rotation')
    W=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+W+W@W/(1+c)


def material_rotation(old_triangle,new_triangle,old_normal,normal):
    def basis(tri):
        e0=tri[1]-tri[0];e1=tri[2]-tri[0];cross=np.cross(e0,e1)
        if np.linalg.norm(cross)<1e-15:raise ValueError('Degenerate history triangle')
        return np.stack([e0,e1,cross/np.linalg.norm(cross)],axis=1)
    D=basis(new_triangle)@np.linalg.inv(basis(old_triangle))
    u,_,vt=np.linalg.svd(D);R=u@vt
    if np.linalg.det(R)<=0:raise ValueError('Improper material transport')
    return normal_rotation(R@old_normal,normal)@R


def closest_segments(a,b,c,d):
    """Constrained closest points on two segments; handles parallel segments."""
    a,b,c,d=map(lambda v:np.asarray(v,float),(a,b,c,d));u=b-a;v=d-c
    if u@u<=0 or v@v<=0:raise ValueError('Zero-length contact edge')
    choices=[]
    def add(s,t):
        p=a+s*u;q=c+t*v;choices.append((float((p-q)@(p-q)),s,t,p,q))
    for s in (0.,1.):add(s,float(np.clip((a+s*u-c)@v/(v@v),0,1)))
    for t in (0.,1.):add(float(np.clip((c+t*v-a)@u/(u@u),0,1)),t)
    gram=np.array([[u@u,-u@v],[-u@v,v@v]])
    if np.linalg.det(gram)>1e-14*(u@u)*(v@v):
        s,t=np.linalg.solve(gram,np.array([-u@(a-c),v@(a-c)]))
        if 0<=s<=1 and 0<=t<=1:add(float(s),float(t))
    distance2,s,t,p,q=min(choices,key=lambda item:item[0])
    return dict(distance=float(np.sqrt(distance2)),s=s,t=t,point_a=p,point_b=q)


def swept_edge_event(old,new,tolerance=1e-9):
    """Linear endpoint trajectories: exact coplanarity roots, fail closed.

    Coplanar moving swept boxes can be ambiguous: return an event rather than
    claim a clear trajectory. This is a stopping guard, NOT an edge contact law.
    """
    old=np.asarray(old,float);new=np.asarray(new,float)
    lo0=np.minimum(old[:2],new[:2]).min(axis=0);hi0=np.maximum(old[:2],new[:2]).max(axis=0)
    lo1=np.minimum(old[2:],new[2:]).min(axis=0);hi1=np.maximum(old[2:],new[2:]).max(axis=0)
    if np.any(hi0<lo1-tolerance) or np.any(hi1<lo0-tolerance):return None
    delta=new-old
    relative=delta-delta[0]
    # Distance between moving segments is Lipschitz in endpoint motions.
    # This certifies separation even when their axis-aligned boxes overlap.
    initial=closest_segments(*old)['distance']
    motion=np.max(np.linalg.norm(relative[:2],axis=1))+np.max(np.linalg.norm(relative[2:],axis=1))
    if initial>motion+tolerance:return None
    r=old[2]-old[0];dr=delta[2]-delta[0]
    u=old[1]-old[0];du=delta[1]-delta[0];v=old[3]-old[2];dv=delta[3]-delta[2]
    cross0=np.cross(u,v);cross1=np.cross(du,v)+np.cross(u,dv);cross2=np.cross(du,dv)
    coeff=np.array([r@cross0,dr@cross0+r@cross1,dr@cross1+r@cross2,dr@cross2])
    scale=max(np.linalg.norm(u),np.linalg.norm(v),np.linalg.norm(r),tolerance)
    if np.max(abs(coeff))<=tolerance*scale**2:
        if np.max(abs(delta-delta[0]))<=tolerance:
            hit=closest_segments(*old)
            return dict(time_fraction=0.,reason='static edge intersection') if hit['distance']<=tolerance else None
        return dict(time_fraction=0.,reason='ambiguous coplanar swept edge boxes')
    coeff=np.trim_zeros(coeff,'b');roots=np.polynomial.polynomial.polyroots(coeff)
    for root in sorted((float(z.real) for z in roots if abs(z.imag)<1e-8 and -1e-10<=z.real<=1+1e-10)):
        tau=float(np.clip(root,0,1));points=old+tau*delta;hit=closest_segments(*points)
        if hit['distance']<=tolerance:return dict(time_fraction=tau,reason='swept edge intersection')
    return None
