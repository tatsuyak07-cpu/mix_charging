"""Geometry-only injectivity certificate for a canonical tensor-grid P1 map.

Missing nodes in voids receive a harmonic GEOMETRIC extension, not material,
mass or constitutive stiffness. Physical tetrahedra must be a subset of the
canonical full-box subdivision. This adapter is not a general packing/remesh
certificate and never authorizes contact-force removal.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from .sintered import mesh
from .explicit_material import geometry,hex_to_tet


class StructuredExtension:
    def __init__(self,X,t,max_nodes=200000):
        self.X=np.asarray(X,float);self.physical_tets=np.asarray(t,int)
        if self.X.ndim!=2 or self.X.shape[1]!=3 or not len(t) or not np.all(np.isfinite(X)):raise ValueError('Finite nonempty physical mesh required')
        geometry(self.X,self.physical_tets)
        self.axes=[np.unique(self.X[:,k]) for k in range(3)];shape=tuple(len(v) for v in self.axes)
        if min(shape)<2 or np.prod(shape)>max_nodes:raise ValueError('Unsupported tensor-grid extent or node budget')
        _,hexes,ijk=mesh(np.ones(tuple(n-1 for n in shape),bool),1.)
        self.reference=np.column_stack([self.axes[k][ijk[:,k]] for k in range(3)])
        self.tets=hex_to_tet(self.reference,hexes);self.shape=shape
        self.ids=np.ravel_multi_index(tuple(np.searchsorted(self.axes[k],self.X[:,k]) for k in range(3)),shape)
        if len(np.unique(self.ids))!=len(X):raise ValueError('Duplicate reference coordinates unsupported')
        self.known=np.zeros(len(self.reference),bool);self.known[self.ids]=True;self.ghost=np.flatnonzero(~self.known)
        boundary=np.any((ijk==0)|(ijk==np.array(shape)-1),axis=1)
        if not np.all(self.known[boundary]):raise ValueError('Physical constraints required on complete box boundary')
        canonical={tuple(sorted(tet)):i for i,tet in enumerate(self.tets)};self.material_mask=np.zeros(len(self.tets),bool)
        for tet in self.ids[self.physical_tets]:
            key=tuple(sorted(tet))
            if key not in canonical:raise ValueError('Physical tetrahedron is not in canonical background subdivision')
            index=canonical[key]
            if self.material_mask[index]:raise ValueError('Duplicate physical tetrahedron')
            self.material_mask[index]=True
        self.lengths=np.array([v[-1]-v[0] for v in self.axes]);self.grid=np.arange(len(self.reference)).reshape(shape)
        V,g,D=geometry(self.reference,self.tets);self.invD=np.linalg.inv(D);self.reference_volume=V
        blocks=np.einsum('n,nai,nbi->nab',V,g,g);rows=np.repeat(self.tets,4,axis=1).ravel();cols=np.tile(self.tets,(1,4)).ravel()
        matrix=coo_matrix((blocks.ravel(),(rows,cols)),shape=(len(self.reference),len(self.reference))).tocsr()
        self.Kgg=matrix[self.ghost][:,self.ghost].tocsc();self.Kgp=matrix[self.ghost][:,self.ids];self.factor=splu(self.Kgg) if len(self.ghost) else None

    def evaluate(self,x,H,margin_tolerance=1e-10,periodic_tolerance=1e-10):
        x=np.asarray(x,float);H=np.asarray(H,float)
        if x.shape!=self.X.shape or H.shape!=(3,3) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(H)) or np.linalg.det(H)<=0:raise ValueError('Finite geometry and positive cell determinant required')
        A=H/self.lengths;u=np.zeros_like(self.reference);u[self.ids]=x-self.X@A.T
        residual=0.
        if self.factor is not None:
            rhs=-(self.Kgp@u[self.ids]);u[self.ghost]=self.factor.solve(rhs)
            residual=float(np.linalg.norm(self.Kgg@u[self.ghost]-rhs)/max(np.linalg.norm(rhs),1e-30))
        y=self.reference@A.T+u;y[self.ids]=x
        periodic_error=0.
        for axis in range(3):
            lo=np.take(self.grid,0,axis=axis).ravel();hi=np.take(self.grid,-1,axis=axis).ravel()
            periodic_error=max(periodic_error,float(np.max(abs(y[hi]-y[lo]-H[:,axis]))))
        D=(y[self.tets][:,1:]-y[self.tets][:,:1]).swapaxes(1,2);F=D@self.invD
        normalized=np.linalg.inv(A)@F;symmetric=(normalized+normalized.swapaxes(1,2))/2
        margins=np.linalg.eigvalsh(symmetric)[:,0];J=np.linalg.det(F);minimum=float(margins.min());scale=max(float(np.linalg.norm(H)),1.)
        checks=dict(physical_positions_preserved=bool(np.array_equal(y[self.ids],x)),harmonic_residual_small=residual<1e-10,periodic_boundary_matches=periodic_error<periodic_tolerance*scale,all_extension_jacobians_positive=bool(J.min()>0),strong_monotonicity_margin=minimum>margin_tolerance)
        summary=dict(scope=__doc__,certificate_passed=all(checks.values()),checks=checks,normalized_margin=minimum,minimum_extension_J=float(J.min()),minimum_material_J=float(J[self.material_mask].min()),minimum_void_extension_J=float(J[~self.material_mask].min()) if np.any(~self.material_mask) else None,ghost_nodes=len(self.ghost),physical_nodes=len(x),extension_nodes=len(y),physical_tetrahedra=int(self.material_mask.sum()),extension_tetrahedra=len(self.tets),harmonic_relative_residual=residual,periodic_error_mm=periodic_error,macro_A=A.tolist(),sigma_min_A=float(np.linalg.svd(A,compute_uv=False).min()),distance_lower_bound_factor=float(np.linalg.svd(A,compute_uv=False).min()*minimum) if minimum>0 else None,candidate_exclusion_authorized=False,physical_void_filled=False,contact_force_computed=False)
        data=dict(reference_points=self.reference,points=y,tets=self.tets,physical_node_indices=self.ids,material_tet_mask=self.material_mask,normalized_margin_per_tet=margins,current_J=J,cell=H)
        return summary,data
