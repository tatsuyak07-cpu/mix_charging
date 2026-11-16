import numpy as np
import pytest
from packing_fem.mesh_sections import section_area


def test_single_tetrahedron_section_and_window_clipping():
    points=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    tets=np.array([[0,1,2,3]])
    area,count=section_area(points,tets,[0.,0.,.25],[0.,0.,1.],2.)
    assert area==pytest.approx(.75**2/2);assert count==1
    area,_=section_area(points,tets,[0.,0.,.25],[0.,0.,1.],.25)
    assert area==pytest.approx(.25**2)


def test_slice_is_invariant_to_rigid_rotation_translation_and_normal_sign():
    points=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    angle=.61;Q=np.array([[np.cos(angle),0.,np.sin(angle)],[0.,1.,0.],[-np.sin(angle),0.,np.cos(angle)]])
    translation=np.array([6.,-2.,4.]);center=np.array([0.,0.,.25]);normal=np.array([0.,0.,1.])
    for sign in [-1,1]:
        area,_=section_area(points@Q.T+translation,np.array([[0,1,2,3]]),Q@center+translation,sign*Q@normal,2.)
        assert area==pytest.approx(.75**2/2)


def test_coincident_face_is_rejected():
    points=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    with pytest.raises(ValueError,match='coincides'):
        section_area(points,np.array([[0,1,2,3]]),[0.,0.,0.],[0.,0.,1.],2.)


def test_multi_tetrahedron_cube_section_is_additive_and_clipped():
    points=np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.],
                     [0.,0.,1.],[1.,0.,1.],[1.,1.,1.],[0.,1.,1.]])
    tets=np.array([[0,1,2,6],[0,2,3,6],[0,3,7,6],
                   [0,7,4,6],[0,4,5,6],[0,5,1,6]])
    area,count=section_area(points,tets,[.5,.5,.37],[0.,0.,1.],2.)
    assert area==pytest.approx(1.);assert count==6
    area,_=section_area(points,tets,[.5,.5,.37],[0.,0.,1.],.2)
    assert area==pytest.approx(.16)
