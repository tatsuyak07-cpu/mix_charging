import numpy as np
import pytest
from packing_fem.area_contact import AreaContact,SurfaceResolutionError
from packing_fem.explicit_periodic import Periodic
from tools.verify_plane_contact import fixture,measure


@pytest.mark.parametrize('n,phase',[(1,0.),(1,.01),(2,.25),(4,.5)])
def test_area_plane_matches_uniform_integral_without_mesh_phase_gain(n,phase):
    row=measure(n,.015,phase,backend='area_guarded')
    assert row['force_gain']==pytest.approx(1.,abs=1e-11)
    assert row['energy_gain']==pytest.approx(1.,abs=1e-11)
    assert row['tangent_resultant_N']<1e-12
    assert row['gap_gradient_relative']<1e-7 and row['macro_work_relative']<1e-7


def sticking_fixture():
    X,t,x,H,upper=fixture(1,.015,.25)
    c=AreaContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02)
    r=c.evaluate(x,x*0,H,H*0,0.);c.history=r[2]
    x=x.copy();H=H.copy();x[upper,0]+=.0001;H[0,2]+=.0001
    return X,t,x,H,upper,c


def test_sticking_area_force_is_full_potential_gradient():
    X,t,x,H,upper,c=sticking_fixture();f,extra,hist,r=c.evaluate(x,x*0,H,H*0,.001)
    assert f[upper,0].sum()==pytest.approx(-.001,abs=1e-11)
    eps=1e-7
    for node in range(len(x)):
        for axis in range(3):
            delta=np.zeros_like(x);delta[node,axis]=eps
            def E(z):
                report=c.evaluate(z,z*0,H,H*0,.001)[-1]
                return report['normal_energy']+report['tangent_energy']
            assert -(E(x+delta)-E(x-delta))/(2*eps)==pytest.approx(f[node,axis],abs=1e-8)
    assert r['max_coulomb_ratio']<=.3


def test_area_friction_history_is_objective_and_coulomb_bounded():
    _,_,x,H,upper,c=sticking_fixture();f,extra,hist,r=c.evaluate(x,x*0,H,H*0,.001)
    theta=.4;R=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    y=x@R.T+np.array([.3,-.2,.1]);rot=c.evaluate(y,y*0,R@H,H*0,.001)
    np.testing.assert_allclose(rot[0],f@R.T,atol=1e-11)
    assert rot[-1]['tangent_energy']==pytest.approx(r['tangent_energy'],abs=1e-13)
    c.history=hist;x=x.copy();x[upper,0]+=.01;H=H.copy();H[0,2]+=.01
    sliding=c.evaluate(x,x*0,H,H*0,.001)
    assert sliding[-1]['friction_increment']>0
    assert sliding[-1]['max_coulomb_ratio']<=.3+1e-12
    assert abs(sliding[0][upper,0].sum())==pytest.approx(.3*sliding[0][upper,2].sum(),rel=1e-10)


def test_guard_rejects_uncaptured_edge_approach():
    # Two long, thin tetrahedra whose crossed edge interiors approach while
    # every vertex stays farther than the barrier thickness from the other.
    from packing_fem.explicit_material import oriented
    X=np.array([[-1.,0,0],[1,0,0],[-1,-.1,-.1],[-1,.1,-.1],
                [0,-1,.1],[0,1,.1],[-.1,-1,.2],[.1,-1,.2]])+np.array([2,2,2])
    t=oriented(X,np.array([[0,1,2,3],[4,5,6,7]]));x=X.copy();x[4:,2]-=.085
    c=AreaContact(X,t,Periodic(X,5.),20.,.3,.2,thickness=.02)
    with pytest.raises(SurfaceResolutionError,match='Edge-only'):c.evaluate(x,x*0,np.eye(3)*5,np.zeros((3,3)),0.)


def test_sticking_friction_has_no_nodal_force_jump_at_shared_triangle_edge():
    X,t,x,H,upper=fixture(1,.015,.1);x[upper,1]+=.06305;H[1,2]+=.06305
    c=AreaContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02)
    c.history=c.evaluate(x,x*0,H,H*0,0.)[2]
    for eps in (1e-6,1e-7,1e-8):
        forces=[]
        for slip in (.00005-eps,.00005+eps):
            y=x.copy();G=H.copy();y[upper,0]+=slip;G[0,2]+=slip
            r=c.evaluate(y,y*0,G,G*0,.001);forces.append(r[0])
            assert r[0][upper,0].sum()==pytest.approx(-10*slip,abs=1e-10)
        assert np.linalg.norm(forces[1]-forces[0])<100*(2*eps)


@pytest.mark.parametrize('angle',[.01,.5,2.])
def test_minimal_normal_rotation_preserves_spring_energy(angle):
    from packing_fem.area_contact import transported_spring
    a=np.array([0.,0,1.]);b=np.array([np.sin(angle),0,np.cos(angle)])
    value=transported_spring(a,b,np.array([1.,0,0]))
    np.testing.assert_allclose(value,[np.cos(angle),0,-np.sin(angle)],atol=1e-14)
    assert value@value==pytest.approx(1.,abs=1e-14)
    assert value@b==pytest.approx(0.,abs=1e-14)


def test_convex_surface_face_edge_normal_force_is_continuous():
    from packing_fem.explicit_material import oriented
    target=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,-1]])
    point=np.array([.3,-.01,.015]);source=point+np.array([[0,0,0],[.3,0,.3],[-.3,0,.3],[0,-.3,.3]])
    X=np.r_[target,source]+2;t=oriented(X,np.array([[0,1,2,3],[4,5,6,7]]))
    c=AreaContact(X,t,Periodic(X,5.),20.,0.,.1,thickness=.02)
    for eps in (1e-5,1e-6,1e-7):
        forces=[]
        for sign in (-1,1):
            x=X.copy();x[4:,1]+=.01+sign*eps
            forces.append(c.evaluate(x,x*0,np.eye(3)*5,np.zeros((3,3)),0.)[0])
        assert np.linalg.norm(forces[1]-forces[0])<100*eps


def test_area_backend_pressure_solver_elastic_limit(tmp_path):
    import json
    from pathlib import Path
    from scipy.optimize import brentq
    from packing_fem import sintered
    from packing_fem.explicit_material import hex_to_tet
    from packing_fem.explicit_solver import solve
    c=json.loads(Path('configs/explicit-prototype.json').read_text());c.update(contact_backend='area_guarded',loading_mode='nominal_pressure',pressure_mpa=.01,length_mm=1.,yield_mpa=1e6,ramp_s=.0006,hold_s=.0003,damping_per_s=20000.)
    X,h,_=sintered.mesh(np.ones((1,1,1),bool),1.);r=solve(tmp_path/'pressure',X,hex_to_tet(X,h),c)
    expected=brentq(lambda lam:-c['young_mpa']*np.log(lam)/lam-.01,.99,1.)
    assert all(r['checks'].values()),r
    assert 1-r['final_axial_compression']==pytest.approx(expected,abs=1e-7)
    assert not r['edge_edge_response_implemented']
    assert r['edge_response_scope']=='collision_guard_only_no_edge_spring'


@pytest.mark.parametrize('height',[.012,.015])
def test_distinct_simultaneous_closest_patches_stop_instead_of_dropping_one(height):
    from packing_fem.explicit_material import oriented
    lower=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,-1.]])
    upper=np.array([[0.,0,.03],[0,1,.03],[1,0,.03],[0,0,1.]])
    source=np.array([.3,.3,height])+np.array([[0,0,0],[.2,0,0],[0,.2,0],[0,0,.003]])
    X=np.r_[lower,upper,source]+2;t=oriented(X,np.arange(12).reshape(3,4))
    c=AreaContact(X,t,Periodic(X,5.),20.,.3,.1,thickness=.02)
    with pytest.raises(SurfaceResolutionError,match='Multiple distinct'):c.evaluate(X,X*0,np.eye(3)*5,np.zeros((3,3)),0.)


def test_area_history_survives_boundary_preserving_reconnection():
    from test_contact_integration import fixture as contact_fixture
    from tools.verify_explicit_remesh import seed_bipyramid
    from packing_fem.adaptive_tet import propose
    from packing_fem.explicit_material import geometry
    import json
    from pathlib import Path
    X,_,t,_,_=contact_fixture();count=len(X);X,t=seed_bipyramid(X,t,0,.04);x=X.copy();x[count-4:count,2]-=.19
    c=AreaContact(X,t,Periodic(X,3.),100.,.3,.1,thickness=.02);H=np.eye(3)*3
    r=c.evaluate(x,x*0,H,H*0,0.);c.history=r[2];c.accept_geometry(x,H)
    x[count-4:count,0]+=.0001;r=c.evaluate(x,x*0,H,H*0,.001);c.history=r[2];c.accept_geometry(x,H)
    assert r[-1]['tangent_energy']>0
    cfg=json.loads(Path('configs/explicit-prototype.json').read_text());V0=geometry(X,t)[0]
    candidate=propose(x,t,np.repeat(np.eye(3)[None],len(t),axis=0),np.full(len(t),.2),V0,cfg,1.,.001)
    assert candidate is not None
    baseline=c.evaluate(x,x*0,H,H*0,0.)
    c.update_topology(candidate[0]);new=c.evaluate(x,x*0,H,H*0,0.)
    np.testing.assert_allclose(new[0],baseline[0],atol=1e-12)
    assert new[-1]['tangent_energy']==pytest.approx(r[-1]['tangent_energy'],abs=1e-14)


def test_finite_periodic_sliding_transfers_anchor_and_matches_coulomb_work():
    from tools.verify_area_sliding import run
    rows,report=run()
    assert all(report['checks'].values()),report
    assert rows[-1]['slip_mm']==pytest.approx(.8)


def test_sticking_friction_periodic_macro_work_includes_image_anchor_terms():
    from packing_fem.explicit_periodic import affine_basis
    X,_,x,H,_,c=sticking_fixture();p=Periodic(X,1.)
    f,extra,_,_=c.evaluate(x,x*0,H,H*0,.001);generalized=p.force(f,extra);basis=affine_basis(X);eps=1e-7
    for k in range(6):
        v=np.zeros(6);v[k]=1.;macro=np.einsum('aij,j->ai',affine_basis(np.eye(3)),v).T
        def E(z,G):
            r=c.evaluate(z,z*0,G,G*0,.001)[-1]
            return r['normal_energy']+r['tangent_energy']
        derivative=-(E(x+eps*basis[:,:,k],H+eps*macro)-E(x-eps*basis[:,:,k],H-eps*macro))/(2*eps)
        assert derivative==pytest.approx(generalized[-6+k],abs=1e-8)
