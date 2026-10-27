import numpy as np
import pytest
from packing_fem.triangle_quadrature import integrate,density,lower_bound,split
from packing_fem.barrier_contact import barrier


def planes(gap=.015):
    return np.array([[0.,0.,gap],[1.,0.,gap],[0.,1.,gap]]),np.array([[-2.,-2.,0.],[4.,-2.,0.],[-2.,4.,0.]])


def test_constant_interface_matches_area_spring_and_balances_force_and_moment():
    source,target=planes();result=integrate(source,target,.5)
    energy,force,_=barrier(.015,20.,.02)
    assert result['energy']==pytest.approx(.5*energy,rel=1e-12)
    np.testing.assert_allclose(result['force'][:3,2],np.full(3,force/6),atol=1e-13)
    np.testing.assert_allclose(result['force'].sum(0),0,atol=1e-13)
    np.testing.assert_allclose(np.cross(np.r_[source,target],result['force']).sum(0),0,atol=1e-13)
    assert result['area_weights'].sum()==pytest.approx(.5,abs=1e-14)


def test_separated_triangles_prune_without_fictitious_reactions():
    source,target=planes(.03);result=integrate(source,target,.5)
    assert result['energy']==0 and np.count_nonzero(result['force'])==0
    assert len(result['area_weights'])==0
    assert result['stats']['pruned_area']==pytest.approx(.5)
    assert result['stats']['evaluations']==0


def test_zero_aabb_bound_does_not_silently_accept_nonzero_samples():
    source,target=planes();theta=.6
    R=np.array([[1.,0.,0.],[0.,np.cos(theta),-np.sin(theta)],[0.,np.sin(theta),np.cos(theta)]])
    assert lower_bound(source@R.T,target@R.T)==0
    with pytest.raises(ValueError,match='unresolved'):
        integrate(source@R.T,target@R.T,.5,max_depth=0)


def test_general_nodal_force_is_gradient_of_single_material_point_potential():
    source,target=planes();source[1,2]=.018
    bary=np.array([.2,.3,.5]);value,_=density(bary,source,target,20.,.02)
    direction=np.array([[.1,.2,.3],[-.2,.4,.1],[.3,-.4,-.1],[-.1,.2,-.2],[.1,.1,.4],[.2,-.2,-.3]])
    step=1e-7
    plus=density(bary,source+step*direction[:3],target+step*direction[3:],20.,.02)[0][0]
    minus=density(bary,source-step*direction[:3],target-step*direction[3:],20.,.02)[0][0]
    assert -(plus-minus)/(2*step)==pytest.approx(value[1:]@direction.ravel(),rel=1e-7,abs=1e-11)


def test_reference_area_and_source_vertex_order_preserve_virtual_work():
    source,target=planes();first=integrate(source,target,.5);order=[2,0,1]
    other=integrate(source[order],target,1.)
    assert other['energy']==pytest.approx(2*first['energy'])
    np.testing.assert_allclose(other['force'][:3],2*first['force'][order],atol=1e-12)
    np.testing.assert_allclose(other['force'][3:],2*first['force'][3:],atol=1e-12)


@pytest.mark.parametrize('values',[[0.,1.,0.],[0.,1.,1.],[0.,.5,1.],[1.,1.,1.]])
def test_cut_partitions_reference_area_without_negative_weights(values):
    children=split(np.eye(3),np.array(values))
    assert sum(abs(np.linalg.det(c)) for c in children)==pytest.approx(1.,abs=1e-14)
    assert all(np.min(c)>=0 and np.max(abs(c.sum(1)-1))<1e-14 for c in children)


def test_invalid_reference_measure_and_tolerance_rejected():
    source,target=planes()
    with pytest.raises(ValueError):integrate(source,target,0.)
    with pytest.raises(ValueError):integrate(source,target,.5,relative_tolerance=0.)


@pytest.mark.parametrize('theta',[0.,.6])
def test_batch_density_matches_scalar_faces_edges_vertices_and_inactive_points(theta):
    from packing_fem.triangle_quadrature import density_batch
    source=np.array([[-.2,-.2,.01],[1.4,.1,.025],[.1,1.4,.015]])
    target=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    R=np.array([[1.,0.,0.],[0.,np.cos(theta),-np.sin(theta)],[0.,np.sin(theta),np.cos(theta)]])
    source=source@R.T;target=target@R.T
    bary=np.r_[np.eye(3),[[.2,.3,.5],[.5,.4,.1],[.9,.05,.05]]]
    values,gaps=density_batch(bary,source,target,20.,.02)
    scalar=[density(b,source,target,20.,.02) for b in bary]
    np.testing.assert_allclose(values,np.array([v[0] for v in scalar]),rtol=1e-12,atol=1e-13)
    np.testing.assert_allclose(gaps,np.array([v[1] for v in scalar]),rtol=1e-12,atol=1e-14)


def test_thin_edge_contact_has_nonzero_force_and_bounded_omissions():
    from tools.verify_triangle_integration import fixture,reference
    source,target=fixture('edge',.002,.005)
    result=integrate(source,target,.5);exact=reference('edge',.002,.005)
    assert result['energy']==pytest.approx(exact[0],rel=2e-3)
    assert result['force'][:3,2].sum()==pytest.approx(exact[1],rel=2e-3)
    assert result['stats']['negligible_cells']>0
    assert result['stats']['omitted_energy_upper']<=1e-18
    assert result['stats']['omitted_force_upper']<=1e-16
    assert result['area_weights'].sum()+result['stats']['pruned_area']==pytest.approx(.5,abs=1e-12)
