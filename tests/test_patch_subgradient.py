import numpy as np
import pytest
from packing_fem.surface_patch import field,IntegrationRefinement
from packing_fem.barrier_contact import barrier


@pytest.mark.parametrize('height',[.005,.01,.015])
def test_symmetric_v_patch_matches_analytic_uniform_clarke_gradient(height):
    source=np.array([[-.001,-.001,height],[.001,-.001,height],[0.,.002,height]])
    targets=np.array([[[0,-1,0],[0,1,0],[-1,0,1]],[[0,1,0],[0,-1,0],[1,0,1]]],float)
    mapping=np.array([[0,1,2],[1,0,3]]);B=np.array([[1/3]*3])
    with pytest.raises(IntegrationRefinement,match='subgradient'):field(B,source,targets,20.,.02,mapping)
    v,d,_=field(B,source,targets,20.,.02,mapping,tie_rule='clarke');E,F,_=barrier(height/np.sqrt(2),20.,.02)
    assert v[0,0]==pytest.approx(E,rel=1e-12)
    force=v[0,1:10].reshape(3,3).sum(0)
    np.testing.assert_allclose(force,[0,0,F/np.sqrt(2)],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(force+v[0,10:].reshape(-1,3).sum(0),0,atol=1e-12)
    repeated,_,_=field(B,source,targets[[1,0,0]],20.,.02,mapping[[1,0,0]],tie_rule='clarke')
    np.testing.assert_allclose(v,repeated,rtol=1e-12,atol=1e-12)
    eps=1e-8
    def energy(delta):return field(B,source+[0,0,delta],targets,20.,.02,mapping,tie_rule='clarke')[0][0,0]
    assert -(energy(eps)-energy(-eps))/(2*eps)==pytest.approx(force[2],rel=1e-7)


def test_clarke_convention_is_retained_through_positive_pose_atlas():
    from packing_fem.contact_atlas import build,density,scales
    from packing_fem.surface_patch import integrate
    source=np.array([[-.001,-.001,.015],[.001,-.001,.015],[0.,.002,.015]])
    targets=np.array([[[0,-1,0],[0,1,0],[-1,0,1]],[[0,1,0],[0,-1,0],[1,0,1]]],float)
    mapping=np.array([[0,1,2],[1,0,3]]);area=np.linalg.norm(np.cross(source[1]-source[0],source[2]-source[0]))/2
    pose=dict(source=source.tolist(),targets=targets.tolist(),mapping=mapping.tolist(),groups=[[0,1]],tie_rule='clarke')
    fitted=build([pose],area,10.,.02)
    reference=integrate(source,targets,area,kn=10.,thickness=.02,tolerance=1e-5,vertex_map=mapping,groups=[[0,1]],pointwise=True,error_budget='normalized',tie_rule='clarke')['value']
    applied=fitted['weights']@density(fitted['bary'],pose,10.,.02)
    assert np.all(fitted['weights']>0)
    assert np.max(abs(applied-reference)/scales(reference))<.001
    assert fitted['atlas'][0]['tie_rule']=='clarke'


def test_distance_bisector_cut_covers_parent_and_resolves_branch_boundary():
    from packing_fem.surface_patch import feature_children,validate_partition
    source=np.array([[-.001,-.001,.015],[.001,-.001,.015],[0.,.002,.015]])
    targets=np.array([[[0,-1,0],[0,1,0],[-1,0,1]],[[0,1,0],[0,-1,0],[1,0,1]]],float)
    children=feature_children(np.eye(3),source,targets,np.array([0,1]),np.ones(3))
    validate_partition(children)
    for B in children:
        x=(B@source)[:,0]
        assert min(x)>=-1e-14 or max(x)<=1e-14
