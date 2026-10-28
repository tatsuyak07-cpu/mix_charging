import numpy as np
import pytest
from scipy.spatial import cKDTree
from packing_fem.geometry import make_packing, candidate_pairs, sphere_volume, tet_volumes, sphere_mesh,tetra_quadrature


def test_integer_count_and_periodic_nonoverlap():
    x=make_packing(15,5,.6,42)
    assert len(x)==21
    assert cKDTree(x,boxsize=15).query_pairs(5*(1-1e-7))==set()
    assert abs(1-len(x)*sphere_volume(5)/15**3-.6)<=sphere_volume(5)/15**3/2
    np.testing.assert_array_equal(x,make_packing(15,5,.6,42))


def test_across_boundary_and_translation_invariance():
    x=np.array([[.2,4,4],[9.8,4,4]])
    assert candidate_pairs(x,[10]*3,1)==[(0,1,(-1,0,0))]
    y=(x+np.array([3,0,0]))%10
    assert candidate_pairs(y,[10]*3,1)==[(0,1,(0,0,0))]


def test_periodic_self_contact_is_counted_once():
    x=np.array([[3.,3,3]])
    p=candidate_pairs(x,[6]*3,6.01)
    assert len(p)==3
    assert {s for _,_,s in p}=={(1,0,0),(0,1,0),(0,0,1)}


def test_solid_and_box_volume_under_known_deformation():
    p=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    t=np.array([[0,1,2,3]])
    F=np.diag([1.1,1.1,.8])
    assert tet_volumes(p@F.T,t)[0]==pytest.approx(np.linalg.det(F)/6)
    # Same affine deformation of solid and box leaves porosity unchanged.
    assert 1-tet_volumes(p@F.T,t)[0]/np.linalg.det(F)==pytest.approx(5/6)


def test_surface_faces_are_boundary_and_positive_elements():
    p,t,s=sphere_mesh(5,1.5)
    assert np.all(tet_volumes(p,t,signed=True)>0)
    face_ids=((0,1,2),(0,3,1),(1,3,2),(2,3,0))
    edges={}
    for e,f in s:
        tri=t[e,list(face_ids[f-1])]
        for a,b in zip(tri,np.roll(tri,-1)):
            key=tuple(sorted((a,b)));edges[key]=edges.get(key,0)+1
    assert set(edges.values())=={2}  # watertight closed sphere surface
    assert tet_volumes(p,t).sum()<sphere_volume(5)  # explicitly faceted


def test_quadratic_ordering_and_independent_affine_volume():
    corners=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    mids=np.array([(corners[i]+corners[j])/2 for i,j in
                  [(0,1),(1,2),(0,2),(0,3),(1,3),(2,3)]])
    p=np.vstack((corners,mids));t=np.arange(10).reshape(1,10)
    np.testing.assert_allclose(tetra_quadrature(p,t),np.full((1,4),1/24))
    assert tet_volumes(p,t)[0]==pytest.approx(1/6)
    F=np.array([[1,.1,0],[0,.8,0],[0,0,1.2]])
    assert tet_volumes(p@F.T+100,t)[0]==pytest.approx(np.linalg.det(F)/6)


def test_curved_sphere_volume_refinement():
    errors=[]
    for size in (1.5,1.):
        p,t,_=sphere_mesh(5,size,2)
        errors.append(abs(tet_volumes(p,t).sum()/sphere_volume(5)-1))
    assert errors[1]<errors[0]<.002
