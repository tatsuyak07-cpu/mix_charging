import numpy as np
import pytest
from packing_fem.uniform_tet import refine
from packing_fem.explicit_material import geometry
from packing_fem.adaptive_tet import boundary


def test_refinement_preserves_parent_volume_and_affine_geometry():
    X=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    t=np.array([[0,1,2,3]]);Y,s,edges=refine(X,t)
    np.testing.assert_allclose(geometry(Y,s)[0],np.full(8,1/48),rtol=1e-14)
    F=np.array([[1.1,.2,0.],[0.,.9,.1],[.1,0.,.8]])
    np.testing.assert_allclose(geometry(Y@F.T,s)[0].sum(),geometry(X@F.T,t)[0].sum(),rtol=1e-14)
    np.testing.assert_array_equal(Y[len(X):],X[edges].mean(axis=1))


def test_shared_face_conforms_and_boundary_area_is_preserved():
    X=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[0.,0.,-1.]])
    t=np.array([[0,1,2,3],[0,2,1,4]]);Y,s,_=refine(X,t)
    def area(points,tets):
        f=boundary(tets);a=points[f[:,1]]-points[f[:,0]];b=points[f[:,2]]-points[f[:,0]]
        return np.linalg.norm(np.cross(a,b),axis=1).sum()/2
    assert len(boundary(s))==4*len(boundary(t))
    assert area(Y,s)==pytest.approx(area(X,t),rel=1e-14)
