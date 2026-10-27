"""Experimental stateless single-triangle-pair reference-area quadrature.

Friction history, periodic image search and bulk solver coupling are absent.
Unresolved lower bounds or integration errors raise; no force is extrapolated.
"""
import numpy as np
from .contact_geometry import closest_triangle
from .barrier_contact import barrier

RULE=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])


def density(bary,source,target,kn,thickness):
    """Potential and six nodal reactions per unit reference area."""
    point=bary@source;closest,weights=closest_triangle(point,target)
    offset=point-closest
    if np.min(weights)>0:
        # Face-interior distance avoids spurious tangential forces from the
        # roundoff in reconstructing the projected Cartesian point.
        normal=np.cross(target[1]-target[0],target[2]-target[0]);normal/=np.linalg.norm(normal)
        offset=normal*float((point-target[0])@normal)
    gap=float(np.linalg.norm(offset))
    energy,normal,_=barrier(gap,kn,thickness)
    force=normal*offset/gap if normal else np.zeros(3)
    return np.r_[energy,(bary[:,None]*force).ravel(),(-weights[:,None]*force).ravel()],gap


def lower_bound(triangle,target):
    """AABB distance with a roundoff allowance; may be very conservative."""
    pad=64*np.finfo(float).eps*max(1.,float(np.max(abs(triangle))),float(np.max(abs(target))))
    separation=np.maximum(np.maximum(triangle.min(0)-target.max(0),target.min(0)-triangle.max(0)),0.)
    return max(0.,float(np.linalg.norm(separation))-pad)


def split(vertices,values):
    """Cut a parametric triangle by a linear interpolant, never prune by it."""
    if np.ptp(values)<1e-12*max(1.,float(max(abs(values)))):
        # Equal vertex samples do not prove absence of an interior minimum.
        lengths=[np.linalg.norm(vertices[(i+1)%3]-vertices[i]) for i in range(3)]
        i=int(np.argmax(lengths));j=(i+1)%3;k=(i+2)%3;mid=(vertices[i]+vertices[j])/2
        return [np.array([vertices[i],mid,vertices[k]]),np.array([mid,vertices[j],vertices[k]])]
    level=(min(values)+max(values))/2;children=[]
    for sign in (1.,-1.):
        polygon=[]
        for i in range(3):
            j=(i+1)%3;a=vertices[i];b=vertices[j];va=sign*(values[i]-level);vb=sign*(values[j]-level)
            if va>=0:polygon.append(a)
            if (va>0 and vb<0) or (va<0 and vb>0):polygon.append(a+(b-a)*va/(va-vb))
        for i in range(1,len(polygon)-1):children.append(np.array([polygon[0],polygon[i],polygon[i+1]]))
    return children


def integrate(source,target,reference_area,kn=20.,thickness=.02,relative_tolerance=1e-3,max_depth=28):
    source=np.asarray(source,float);target=np.asarray(target,float)
    if source.shape!=(3,3) or target.shape!=(3,3):raise ValueError('Expected two 3-D triangles')
    if not reference_area>0 or not relative_tolerance>0:raise ValueError('Positive reference area and tolerance required')
    absolute=np.r_[1e-18,np.full(18,1e-16)]
    stats=dict(cells=0,pruned=0,maximum_depth=0,evaluations=0,pruned_area=0.)
    points=[];weights=[]
    def area(vertices):return reference_area*abs(float(np.linalg.det(vertices)))
    def rule(vertices):
        bary=RULE@vertices;values=[];gaps=[]
        for point in bary:
            value,gap=density(point,source,target,kn,thickness);values.append(value);gaps.append(gap)
        stats['evaluations']+=3
        return area(vertices)*np.mean(values,axis=0),bary,gaps
    def visit(vertices,depth):
        stats['cells']+=1;stats['maximum_depth']=max(depth,stats['maximum_depth'])
        bound=lower_bound(vertices@source,target);a=area(vertices)
        if bound>=thickness:
            stats['pruned']+=1;stats['pruned_area']+=a;return np.zeros(19)
        coarse,_,cg=rule(vertices)
        vertex_gaps=np.array([np.linalg.norm(p-closest_triangle(p,target)[0]) for p in vertices@source])
        children=split(vertices,vertex_gaps)
        fine=np.zeros(19);samples=[];minimum=min(cg)
        for child in children:
            value,bary,gaps=rule(child);fine+=value;samples.append((child,bary));minimum=min(minimum,min(gaps))
        # A zero estimate never certifies an empty contact patch. Also require
        # samples close to the lower bound so an unseen peak is not accepted.
        resolved=bound>0 and minimum<=2*bound
        error=abs(fine-coarse)
        if resolved and np.all(error<=relative_tolerance*abs(fine)+absolute*a/reference_area):
            for child,bary in samples:
                points.extend(bary);weights.extend([area(child)/3]*3)
            return fine
        if depth>=max_depth:raise ValueError('Triangle contact integration unresolved at maximum depth')
        return sum((visit(child,depth+1) for child in children),np.zeros(19))
    response=visit(np.eye(3),0)
    return dict(energy=float(response[0]),force=response[1:].reshape(6,3),
                barycentric_points=np.asarray(points).reshape(-1,3),area_weights=np.asarray(weights),stats=stats)
