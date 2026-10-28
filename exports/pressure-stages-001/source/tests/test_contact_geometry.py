import numpy as np
import pytest
from packing_fem.contact_geometry import closest_triangle,closed_probe
from packing_fem.adaptive_tet import boundary
from packing_fem.explicit_material import oriented


def tetra():
    x=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    return x,boundary(oriented(x,np.arange(4)[None]))


@pytest.mark.parametrize('p,w',[
    ([.2,.3,.1],[.5,.2,.3]),([.3,-.2,.1],[.7,.3,0]),
    ([-.2,-.3,.1],[1,0,0]),([.8,.8,.1],[0,.5,.5])])
def test_closest_feature_analytic(p,w):
    tri=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    q,weights=closest_triangle(np.array(p),tri)
    np.testing.assert_allclose(weights,w,atol=1e-14)
    np.testing.assert_allclose(q,np.asarray(w)@tri,atol=1e-14)


def test_closed_probe_force_is_energy_derivative_for_slave_and_master():
    x,f=tetra();p=np.array([.25,.23,.07]);k=13.;r=closed_probe(p,x,f,k);eps=1e-7
    assert r['inside']
    for axis in range(3):
        dp=np.eye(3)[axis]*eps
        grad=(closed_probe(p+dp,x,f,k)['energy']-closed_probe(p-dp,x,f,k)['energy'])/(2*eps)
        assert -grad==pytest.approx(r['force'][axis],abs=1e-8)
    for node in range(len(x)):
        for axis in range(3):
            dx=np.zeros_like(x);dx[node,axis]=eps
            grad=(closed_probe(p,x+dx,f,k)['energy']-closed_probe(p,x-dx,f,k)['energy'])/(2*eps)
            assert -grad==pytest.approx(r['master_force'][node,axis],abs=1e-8)
    np.testing.assert_allclose(r['force']+r['master_force'].sum(axis=0),0,atol=1e-14)
    np.testing.assert_allclose(np.cross(p,r['force'])+np.cross(x,r['master_force']).sum(axis=0),0,atol=1e-14)


def test_periodic_image_work_and_objectivity_of_closed_probe():
    x,f=tetra();p=np.array([1.25,.23,.07]);H=np.eye(3);shift=np.array([1,0,0]);k=13.
    r=closed_probe(p,x+H@shift,f,k);eps=1e-7
    # The image is translated by H*s; macro work is sum(master force) dot dH*s.
    for i in range(3):
        for j in range(3):
            dH=np.zeros((3,3));dH[i,j]=eps
            grad=(closed_probe(p,x+(H+dH)@shift,f,k)['energy']-closed_probe(p,x+(H-dH)@shift,f,k)['energy'])/(2*eps)
            assert -grad==pytest.approx(r['master_force'].sum(axis=0)[i]*shift[j],abs=1e-8)
    theta=.61;R=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1.]])
    translation=np.array([3.,-2.,.5]);rr=closed_probe(R@p+translation,(x+H@shift)@R.T+translation,f,k)
    assert rr['energy']==pytest.approx(r['energy'],abs=1e-14)
    np.testing.assert_allclose(rr['force'],R@r['force'],atol=1e-13)


def test_no_ghost_contact_outside_face_edge_or_vertex():
    x,f=tetra()
    for p in ([.2,.3,-.01],[.3,-.01,.02],[-.01,-.01,.02],[.5,.5,.1]):
        r=closed_probe(np.array(p),x,f,13.)
        assert not r['inside'] and r['energy']==0
        np.testing.assert_array_equal(r['force'],0)
    with pytest.raises(ValueError,match='closed'):closed_probe(np.array([.2,.3,.1]),x,f[:-1],13.)


@pytest.mark.parametrize('feature',['edge','vertex'])
def test_concave_feature_potential_gradient(feature):
    from packing_fem import sintered
    from packing_fem.explicit_material import hex_to_tet
    if feature=='edge':
        mask=np.ones((2,2,1),bool);mask[1,1,0]=False;p=np.array([.91,.94,.43])
    else:
        mask=np.ones((3,3,3),bool);mask[1,1,1]=False;p=np.array([.94,.93,.92])
    x,h,_=sintered.mesh(mask,1.);f=boundary(hex_to_tet(x,h));k=13.;eps=1e-7
    r=closed_probe(p,x,f,k)
    assert r['inside']
    assert np.count_nonzero(r['weights']>1e-10)==(2 if feature=='edge' else 1)
    for axis in range(3):
        dp=np.eye(3)[axis]*eps
        grad=(closed_probe(p+dp,x,f,k)['energy']-closed_probe(p-dp,x,f,k)['energy'])/(2*eps)
        assert -grad==pytest.approx(r['force'][axis],abs=1e-8)
    # Perturb vertices of the selected closest feature, including both sides.
    for node in f[r['face']][r['weights']>1e-10]:
        for axis in range(3):
            dx=np.zeros_like(x);dx[node,axis]=eps
            grad=(closed_probe(p,x+dx,f,k)['energy']-closed_probe(p,x-dx,f,k)['energy'])/(2*eps)
            assert -grad==pytest.approx(r['master_force'][node,axis],abs=1e-8)
    np.testing.assert_allclose(r['force']+r['master_force'].sum(axis=0),0,atol=1e-14)
    np.testing.assert_allclose(np.cross(p,r['force'])+np.cross(x,r['master_force']).sum(axis=0),0,atol=1e-13)
