import numpy as np
from packing_fem.implicit_remesh import intersection_volume,overlap


def test_exact_overlap_preserves_partition_and_history_mass():
    x=np.array([[0.,0,0],[1.,0,0],[0,1.,0],[0,0,1.],[.25,.25,.25]])
    old=np.array([[0,1,2,3]])
    new=np.array([[4,1,2,3],[0,4,2,3],[0,1,4,3],[0,1,2,4]])
    W=overlap(x,old,x,new)
    np.testing.assert_allclose(W.toarray(),np.full((4,1),1/24),atol=1e-10)
    np.testing.assert_allclose(np.asarray(W.sum(axis=0)).ravel(),[1/6],atol=1e-10)
    assert intersection_volume(x[:4],x[:4]+2)==0


def test_periodic_edge_bisection_preserves_geometry_and_constitutive_subcells():
    from packing_fem.implicit_remesh import periodic_bisection
    from packing_fem.explicit_material import geometry
    from packing_fem.adaptive_tet import closed_surface_volume
    one=np.array([[0.,0,0],[1.,0,0],[0,1.,0],[0,0,1.]])
    X=np.r_[one,one+[2.,0,0]];t=np.array([[0,1,2,3],[4,5,6,7]])
    x=X@np.diag([.8,1.1,1.05])
    result=periodic_bisection(X,x,t,[0,1],2.)
    assert result['report']['periodic_edges']==2
    assert len(result['t'])==4
    oldv=geometry(x,t)[0];newv=geometry(result['x'],result['t'])[0]
    np.testing.assert_allclose(np.bincount(result['parents'],weights=newv),oldv,atol=1e-14)
    np.testing.assert_allclose(closed_surface_volume(result['x'],result['t']),closed_surface_volume(x,t),atol=1e-14)


def test_edge_search_covers_corner_images_and_reversed_endpoint_order():
    from itertools import product
    from packing_fem.implicit_remesh import periodic_bisection
    from packing_fem.explicit_material import geometry,oriented
    one=np.array([[0.,0,0],[1.,0,0],[0,1.,0],[0,0,1.]])
    blocks=[]
    for i,shift in enumerate(product([0.,2.],repeat=3)):
        block=one.copy()
        if i%2:block[[0,1]]=block[[1,0]]
        blocks.append(block+shift)
    # A disconnected, nonperiodic near-image must remain untouched.
    blocks.append(one+[4.+2e-8,0,0])
    X=np.concatenate(blocks);t=oriented(X,np.arange(len(X)).reshape(-1,4))
    x=X@np.diag([.8,1.1,1.05])
    result=periodic_bisection(X,x,t,[0,1],2.)
    assert result['report']['periodic_edges']==8
    assert result['report']['new_nodes']==8
    np.testing.assert_array_equal(np.bincount(result['parents']),[2]*8+[1])
    np.testing.assert_allclose(np.bincount(result['parents'],weights=geometry(result['x'],result['t'])[0]),geometry(x,t)[0],atol=1e-14)
    np.testing.assert_array_equal(result['t'][result['parents']==8],t[8:9])
