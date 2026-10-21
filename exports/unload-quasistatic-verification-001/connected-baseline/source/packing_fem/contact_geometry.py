"""Experimental closest-feature geometry, not enabled in the explicit solver.

Closed-surface probe uses winding number for the solid side. It is a verification
operator, NOT periodic self-contact: open/cell-cut surfaces must not use it.
"""
import numpy as np


def closest_triangle(point, triangle):
    """Return closest point and convex barycentric weights, including edges."""
    triangle=np.asarray(triangle,float);point=np.asarray(point,float)
    a,b,c=triangle
    e0=b-a;e1=c-a
    gram=np.array([[e0@e0,e0@e1],[e0@e1,e1@e1]])
    if np.linalg.det(gram)<=np.finfo(float).eps*max(float(np.trace(gram))**2,1e-300):
        raise ValueError('Degenerate closest-point triangle')
    uv=np.linalg.solve(gram,np.array([e0@(point-a),e1@(point-a)]))
    weights=np.r_[1-uv.sum(),uv]
    if weights.min()>=0:return weights@triangle,weights
    candidates=[]
    for i,j in ((0,1),(1,2),(2,0)):
        edge=triangle[j]-triangle[i]
        s=np.clip((point-triangle[i])@edge/(edge@edge),0.,1.)
        w=np.zeros(3);w[i]=1-s;w[j]=s
        q=w@triangle;candidates.append((float((point-q)@(point-q)),q,w))
    _,q,w=min(candidates,key=lambda item:item[0])
    return q,w


def closed_probe(point, vertices, faces, stiffness):
    """Frictionless point vs one closed oriented surface, unit tributary area.

    stiffness: N/mm, coordinates: mm, energy: N mm. Inactive outside the
    solid. Inside: 0.5*k*distance**2. Force is the negative derivative away
    from equal-distance medial-axis ties. Returns master nodal forces too.
    """
    if stiffness<=0:raise ValueError('Positive stiffness required')
    vertices=np.asarray(vertices,float);faces=np.asarray(faces,int)
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    _,counts=np.unique(edges,axis=0,return_counts=True)
    if np.any(counts!=2):raise ValueError('Probe requires a closed edge-manifold surface')
    oriented=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    signs=np.where(oriented[:,0]<oriented[:,1],1,-1)
    _,inverse=np.unique(np.sort(oriented,axis=1),axis=0,return_inverse=True)
    balance=np.bincount(inverse,weights=signs)
    if np.any(balance):raise ValueError('Inconsistently oriented closed surface')
    tri=vertices[faces];r=tri-np.asarray(point,float)
    lengths=np.linalg.norm(r,axis=2)
    numerator=np.einsum('ij,ij->i',r[:,0],np.cross(r[:,1],r[:,2]))
    denominator=np.prod(lengths,axis=1)
    denominator+=np.einsum('ij,ij->i',r[:,0],r[:,1])*lengths[:,2]
    denominator+=np.einsum('ij,ij->i',r[:,1],r[:,2])*lengths[:,0]
    denominator+=np.einsum('ij,ij->i',r[:,2],r[:,0])*lengths[:,1]
    winding=float(np.sum(2*np.arctan2(numerator,denominator))/(4*np.pi))
    choices=[]
    for i,triangle in enumerate(tri):
        q,w=closest_triangle(point,triangle)
        choices.append((float(np.sum((q-point)**2)),i,q,w))
    distance2,face,q,weights=min(choices,key=lambda item:(item[0],item[1]))
    inside=abs(winding)>.5
    force=stiffness*(q-point) if inside else np.zeros(3)
    master=np.zeros_like(vertices)
    np.add.at(master,faces[face],-weights[:,None]*force)
    return dict(energy=.5*stiffness*distance2 if inside else 0.,force=force,
                master_force=master,closest=q,weights=weights,face=face,
                winding=winding,inside=inside)
