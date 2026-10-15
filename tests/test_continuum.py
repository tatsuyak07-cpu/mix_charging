"""Independent non-affine C3D8 mapping checks, including solver IP numbering."""
import numpy as np
from packing_fem.continuum import GAUSS,deformation_gradients
from packing_fem.sintered import mesh,hex_quadrature


def test_calculix_gauss_numbering_for_nonaffine_element():
    p,e,_=mesh(np.ones((1,1,1),bool),1.)
    moved=p.copy();moved[:,0]*=1+.4*p[:,2]
    # Mapping x=X(1+0.4Z), y=Y, z=Z has determinant 1+0.4Z.
    expected=(1+.4*(GAUSS[:,2]+1)/2)/8
    np.testing.assert_allclose(hex_quadrature(moved,e)[0],expected,atol=1e-15)
    assert np.all(GAUSS[:4,2]<0) and np.all(GAUSS[4:,2]>0)
    assert np.all(GAUSS[::2,0]<0) and np.all(GAUSS[1::2,0]>0)
    f=deformation_gradients(p,moved,e)[0]
    np.testing.assert_allclose(np.linalg.det(f),8*expected,atol=1e-14)


def test_deformation_gradient_for_sheared_rotated_affine_cube():
    p,e,_=mesh(np.ones((1,1,1),bool),1.)
    f=np.array([[0.,-1.1,.2],[.9,.3,0.],[0.,.1,.8]])
    moved=p@f.T+np.array([12.,-3.,4.])
    computed=deformation_gradients(p,moved,e)
    np.testing.assert_allclose(computed,np.broadcast_to(f,(1,8,3,3)),atol=1e-14)
    np.testing.assert_allclose(hex_quadrature(moved,e),np.linalg.det(f)/8,atol=1e-14)
