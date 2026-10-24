"""Geometry extension on a conforming bisection of an existing certified grid.

Refine void geometry too, without assigning void mass or material. This adapter
supports the exact edge-bisection lineage only, not arbitrary remeshing.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from .explicit_material import geometry,oriented
from .structured_extension import StructuredExtension


def refine_extension(old,proposal):
    obj=object.__new__(StructuredExtension);X=proposal['X'];t=proposal['t'];reference=old.reference.copy();full=old.tets.copy();ids=list(map(int,old.ids))
    for source in proposal['supports'][len(old.X):]:
        a,b=old.ids[source];point=(old.reference[a]+old.reference[b])/2
        if np.any(np.linalg.norm(reference-point,axis=1)<1e-12):raise ValueError('Extension midpoint already present')
        midpoint=len(reference);reference=np.r_[reference,point[None]];ids.append(midpoint)
        owners=np.flatnonzero(np.sum(np.isin(full,[a,b]),axis=1)==2)
        if not len(owners):raise ValueError('Extension edge has no parent')
        keep=np.ones(len(full),bool);keep[owners]=False;children=[]
        for owner in owners:
            rest=[int(i) for i in full[owner] if i not in (a,b)]
            children.extend([[a,midpoint,*rest],[midpoint,b,*rest]])
        full=oriented(reference,np.r_[full[keep],children])
    obj.X=X;obj.physical_tets=t;obj.reference=reference;obj.tets=full;obj.ids=np.array(ids);obj.lengths=old.lengths.copy()
    obj.known=np.zeros(len(reference),bool);obj.known[obj.ids]=True;obj.ghost=np.flatnonzero(~obj.known)
    obj.material_mask=np.zeros(len(full),bool);lookup={tuple(sorted(row)):i for i,row in enumerate(full)}
    for row in obj.ids[t]:
        key=tuple(sorted(row))
        if key not in lookup:raise ValueError('Physical refinement inconsistent with full geometry extension')
        obj.material_mask[lookup[key]]=True
    obj.pairs=[]
    for axis,L in enumerate(obj.lengths):
        lower=reference[:,axis].min();lo=np.flatnonzero(abs(reference[:,axis]-lower)<1e-10);hi=np.flatnonzero(abs(reference[:,axis]-lower-L)<1e-10)
        other=[k for k in range(3) if k!=axis];keys=lambda nodes:[tuple(np.rint(reference[i,other]/(max(obj.lengths)*1e-10)).astype(int)) for i in nodes]
        match=dict(zip(keys(hi),hi));lowkeys=keys(lo)
        if set(lowkeys)!=set(match):raise ValueError('Extension bisection violates periodic surface matching')
        obj.pairs.append((lo,np.array([match[k] for k in lowkeys])))
    V,g,D=geometry(reference,full);obj.invD=np.linalg.inv(D);obj.reference_volume=V
    blocks=np.einsum('n,nai,nbi->nab',V,g,g);rows=np.repeat(full,4,axis=1).ravel();cols=np.tile(full,(1,4)).ravel()
    K=coo_matrix((blocks.ravel(),(rows,cols)),shape=(len(reference),len(reference))).tocsr()
    obj.Kgg=K[obj.ghost][:,obj.ghost].tocsc();obj.Kgp=K[obj.ghost][:,obj.ids];obj.factor=splu(obj.Kgg) if len(obj.ghost) else None
    return obj
