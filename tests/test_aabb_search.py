import numpy as np
import pytest
from packing_fem.aabb_search import AABBTree
from packing_fem.barrier_contact import BarrierContact


def test_hierarchy_matches_exhaustive_box_distances():
    rng=np.random.default_rng(749);low=rng.normal(size=(1000,3));high=low+rng.random((1000,3))*.3;tree=AABBTree(low,high)
    for _ in range(40):
        a=rng.normal(size=3);b=a+rng.random(3)*.2;d=.13
        delta=np.maximum(np.maximum(low-b,a-high),0.)
        expected=set(np.flatnonzero((delta*delta).sum(1)<d*d))
        assert set(tree.query(a,b,d))==expected


@pytest.mark.parametrize('periodic',[False,True])
def test_periodic_contact_pairs_and_forces_match_legacy(periodic):
    from test_explicit_backend import contact_fixture
    X,_,t,p=contact_fixture(periodic);x=X.copy();x[:4,0]-=.19;H=np.eye(3)
    contacts=[BarrierContact(X,t,p,100.,.3,.1,thickness=.02,search_method=method) for method in ('aabb','center_sphere')]
    result=[c.evaluate(x,x*0,H,H*0,0.) for c in contacts]
    assert {r[0] for r in contacts[0].neighbors}=={r[0] for r in contacts[1].neighbors}
    for i in (0,1):np.testing.assert_allclose(result[0][i],result[1][i],rtol=1e-13,atol=1e-14)
    for key in ('normal_energy','tangent_energy','contacts'):assert result[0][-1][key]==pytest.approx(result[1][-1][key])
    assert contacts[0].search_counts['max_query_candidates']<contacts[0].max_candidates


def test_deformed_periodic_images_match_exhaustive_primitive_search():
    from itertools import product
    from test_explicit_backend import contact_fixture
    from packing_fem.contact_geometry import closest_triangle
    from packing_fem.barrier_contact import segment_distances
    X,_,t,p=contact_fixture(True);c=BarrierContact(X,t,p,100.,.3,.1,thickness=.02)
    F=np.array([[1.,.13,.04],[.02,.95,.07],[.03,0.,1.02]]);x=X.copy();x[:4,0]-=.19;x=x@F.T
    shifts=[np.asarray(s) for s in product((-1,0,1),repeat=3)];actual=c.build_neighbors(x,F,shifts);expected=set()
    for node in c.nodes:
        for face,ids in enumerate(c.faces):
            for shift in shifts:
                if c.adjacent(node,face,shift):continue
                tri=x[ids]+F@shift;closest,_=closest_triangle(x[node],tri)
                if np.linalg.norm(x[node]-closest)<c.skin:expected.add(('pt',int(node),face,*shift))
    for i,edge in enumerate(c.edges):
        for j,other in enumerate(c.edges):
            for shift in shifts:
                if (i,j,*shift)>(j,i,*(-shift)) or c.local_edge(i,j,shift):continue
                points=np.r_[x[edge],x[other]+F@shift]
                if segment_distances(points[None])[0]<c.skin:expected.add(('ee',i,j,*shift))
    assert {job[0] for job in actual}==expected


def test_periodic_force_comparison_includes_image_macro_work():
    from tools.verify_plane_contact import fixture
    from packing_fem.explicit_periodic import affine_basis,Periodic
    X=fixture(1,.015,.25)[0];p=Periodic(X,1.)
    pair=next((i,j) for i in range(len(X)) for j in range(i+1,len(X)) if p.classes[i]==p.classes[j] and np.linalg.norm(X[i]-X[j])>.5)
    a,b=pair;direction=np.array([.3,-.2,.1]);fa=np.zeros_like(X);fb=fa.copy();fa[a]=direction;fb[b]=direction
    macro_a=np.zeros(6);macro_b=-(affine_basis(X[b:b+1])[0]-affine_basis(X[a:a+1])[0]).T@direction
    assert np.max(abs(fa-fb))>.1
    assert np.linalg.norm(p.force(fa,macro_a)-p.force(fb,macro_a))>1e-3
    np.testing.assert_allclose(p.force(fa,macro_a),p.force(fb,macro_b),atol=1e-14)


def test_single_dense_query_rejects_before_accumulating_excess_pairs():
    tree=AABBTree(np.zeros((1000,3)),np.ones((1000,3)))
    with pytest.raises(ValueError,match='bounded query; no truncation'):
        tree.query(np.zeros(3),np.zeros(3),.1,max_results=10)
    assert tree.query(np.full(3,3.),np.full(3,3.),.1,max_results=0)==[]
