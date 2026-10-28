import numpy as np
import pytest
from packing_fem.surface_patch import integrate,field,separation_bound,components
from packing_fem.barrier_contact import barrier
from packing_fem.explicit_periodic import affine_basis


def square(gap=.015):
    source=np.array([[0.,0.,gap],[1.,0.,gap],[0.,1.,gap]])
    nodes=np.array([[-1.,-1.,0.],[2.,-1.,0.],[2.,2.,0.],[-1.,2.,0.]])
    faces=np.array([[0,1,2],[0,2,3]])
    return source,nodes,faces


@pytest.mark.parametrize('angle',[0.,.37,1.1])
def test_support_bound_and_multiple_facets_match_one_area_spring(angle):
    source,nodes,faces=square();R=np.array([[1.,0.,0.],[0.,np.cos(angle),-np.sin(angle)],[0.,np.sin(angle),np.cos(angle)]])
    source=source@R.T;nodes=nodes@R.T
    assert 0<separation_bound(source,nodes[faces[0]])<=.015
    result=integrate(source,nodes[faces],.5,vertex_map=faces)
    E,F,_=barrier(.015,20.,.02)
    assert result['value'][0]==pytest.approx(.5*E,rel=1e-10)
    force=result['value'][1:10].reshape(3,3)
    np.testing.assert_allclose(force.sum(0),.5*F*(R@np.array([0.,0.,1.])),atol=1e-10)
    assert result['weights'].sum()==pytest.approx(.5,abs=1e-12)


def test_duplicate_facets_do_not_double_potential_or_reaction():
    source,nodes,faces=square();bary=np.array([[.2,.4,.4],[.7,.1,.2]])
    a=field(bary,source,nodes[faces],20.,.02)[0]
    b=field(bary,source,np.r_[nodes[faces],nodes[faces]],20.,.02)[0]
    np.testing.assert_allclose(a[:,:10],b[:,:10],atol=1e-14)
    np.testing.assert_allclose(a[:,10:].reshape(2,-1,3).sum(1),b[:,10:].reshape(2,-1,3).sum(1),atol=1e-14)


def test_periodic_image_macro_work_includes_image_reaction():
    source,nodes,faces=square();source[:,2]=.99;nodes[:,2]=.005;shift=np.array([0.,0.,1.])
    target=nodes[faces]+shift
    result=integrate(source,target,.5,vertex_map=faces);forces=result['value'][1:].reshape(-1,3)
    image_coordinates=np.r_[source,nodes+shift]
    generalized=np.einsum('aij,ai->j',affine_basis(image_coordinates),forces)
    bary,w=result['bary'],result['weights']
    for k in (2,4,5):
        q=np.zeros(6);q[k]=1.;displacement=affine_basis(image_coordinates)@q
        eps=1e-7
        def energy(sign):
            moved=image_coordinates+sign*eps*displacement
            values=field(bary,moved[:3],moved[3:][faces],20.,.02,faces)[0]
            return w@values[:,0]
        assert -(energy(1)-energy(-1))/(2*eps)==pytest.approx(generalized[k],rel=2e-6,abs=1e-10)


def test_crossing_triangles_never_get_positive_separation_certificate():
    source=np.array([[-1.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    target=np.array([[0.,.2,-1.],[0.,.2,1.],[0.,-.2,0.]])
    assert separation_bound(source,target)==0


def test_reference_image_connectivity_joins_seam_and_separates_disjoint_patches():
    X=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,0.,1.],[0.,1.,1.],[3.,3.,3.],[4.,3.,3.],[3.,4.,3.]])
    faces=np.array([[0,1,2],[3,4,5],[6,7,8]])
    groups=components(faces,[(0,0,0),(0,0,-1),(0,0,0)],X,1.)
    assert groups==[[0,1],[2]]


def test_batched_patch_density_matches_scalar_all_closest_features():
    from packing_fem.surface_patch import density_fast
    from packing_fem.triangle_quadrature import density
    source=np.array([[-.2,-.2,.01],[1.4,.1,.025],[.1,1.4,.015]])
    target=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    bary=np.r_[np.eye(3),[[.2,.3,.5],[.5,.4,.1],[.9,.05,.05]]]
    values,gaps,_=density_fast(bary,source,target,20.,.02)
    scalar=[density(b,source,target,20.,.02) for b in bary]
    np.testing.assert_allclose(values,np.array([v[0] for v in scalar]),rtol=1e-12,atol=1e-13)
    np.testing.assert_allclose(gaps,np.array([v[1] for v in scalar]),rtol=1e-12,atol=1e-14)


def test_inactive_bridge_does_not_merge_two_active_wall_contacts():
    # One connected U-shaped target, but two separate active contact patches.
    from packing_fem.surface_patch import grouped_field
    source=np.array([[0.,0.,0.],[.2,0.,0.],[0.,.2,0.]])
    nodes=np.array([[-1.,-1.,-.015],[1.,-1.,-.015],[1.,1.,-.015],[-1.,1.,-.015],[-1.,-1.,.017],[1.,-1.,.017],[1.,1.,.017],[-1.,1.,.017]])
    faces=np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7],[1,5,6],[1,6,2]])
    groups=[list(range(len(faces)))];targets=nodes[faces];B=np.array([[1/3]*3])
    values,_,_=grouped_field(B,source,targets,groups,20.,.02,faces,pointwise=True)
    E1,F1,_=barrier(.015,20.,.02);E2,F2,_=barrier(.017,20.,.02)
    assert values[0,0]==pytest.approx(E1+E2,rel=1e-12)
    assert values[0,1:10].reshape(3,3).sum(0)[2]==pytest.approx(F1-F2,rel=1e-12)
    result=integrate(source,targets,.02,vertex_map=faces,groups=groups,pointwise=True)
    assert result['value'][0]==pytest.approx(.02*(E1+E2),rel=1e-4)
def test_resultant_error_budget_matches_finer_component_rule():
    source=np.array([[.1,.1,.019],[.8,.1,.021],[.1,.8,.019]])
    targets=np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],[[1.,1.,0.],[0.,1.,0.],[1.,0.,0.]]])
    mapping=np.array([[0,1,2],[3,2,1]])
    from tools.experimental_surface_budget import integrate,grouped_field
    exact=integrate(source,targets,.245,vertex_map=mapping,tolerance=1e-5)['value']
    result=integrate(source,targets,.245,vertex_map=mapping,tolerance=.00025,force_error_norm='resultant')
    applied=result['weights']@grouped_field(result['bary'],source,targets,[[0,1]],20.,.02,mapping)[0]
    force_scale=max(np.linalg.norm(exact[1:10].reshape(3,3).sum(0)),np.linalg.norm(exact[10:].reshape(-1,3),axis=1).sum())
    assert abs(applied[0]-exact[0])/abs(exact[0])<.001
    assert abs(applied[1:]-exact[1:]).max()/force_scale<.001
    np.testing.assert_allclose(applied[1:].reshape(-1,3).sum(0),0,atol=1e-14)
