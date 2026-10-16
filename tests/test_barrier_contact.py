import numpy as np
import pytest
from packing_fem.barrier_contact import barrier,constraint,feature,relative,tangent_frame,mollifier,BarrierContact,ContactResolutionError
from packing_fem.explicit_periodic import Periodic
from packing_fem.explicit_material import oriented


@pytest.mark.parametrize('kind',['pt','ee'])
def test_barrier_normal_force_gradient_and_balance(kind):
    x=np.array([[.2,.2,.01],[0,0,0],[1,0,0],[0,1,0]],float) if kind=='pt' else np.array([[-1,0,0],[1,0,0],[0,-1,.01],[0,1,.01]],float)
    r=constraint(x,kind,10.,.02,0.);eps=1e-7
    for i in range(4):
        for axis in range(3):
            delta=np.zeros_like(x);delta[i,axis]=eps
            def E(z):
                branch,w=feature(z,kind);d=np.linalg.norm(relative(z,kind,branch));return barrier(d,10.,.02)[0]
            assert -(E(x+delta)-E(x-delta))/(2*eps)==pytest.approx(r['force'][i,axis],abs=1e-8)
    np.testing.assert_allclose(r['force'].sum(axis=0),0,atol=1e-14)
    np.testing.assert_allclose(np.cross(x,r['force']).sum(axis=0),0,atol=1e-14)


def test_point_triangle_edge_transition_has_continuous_energy_and_force():
    tri=np.array([[0.,0,0],[1,0,0],[0,1,0]])
    vals=[]
    for y in (-1e-8,0,1e-8):
        x=np.r_[[[.3,y,.01]],tri];r=constraint(x,'pt',10.,.02,0.);vals.append(r)
    assert max(v['normal_energy'] for v in vals)-min(v['normal_energy'] for v in vals)<1e-13
    np.testing.assert_allclose(vals[0]['force'],vals[2]['force'],atol=2e-6)
    assert barrier(.02,10.,.02)==(0.,0.,0.)


def test_friction_gradient_moment_objectivity_and_history():
    x=np.array([[-1.,0,0],[1.,0,0],[0,-1,.01],[0,1,.01]])
    r=constraint(x,'ee',10.,.02,.3);state=r['state'];x[2:,0]+=.001
    r=constraint(x,'ee',10.,.02,.3,state);assert r['tangent_energy']>0
    np.testing.assert_allclose(r['force'].sum(axis=0),0,atol=1e-14)
    np.testing.assert_allclose(np.cross(x,r['force']).sum(axis=0),0,atol=1e-14)
    angle=.7;R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
    rr=constraint(x@R.T+[2,3,4],'ee',10.,.02,.3,state)
    np.testing.assert_allclose(rr['force'],r['force']@R.T,atol=1e-13)
    assert r['coulomb_ratio']<=.3+1e-12


def test_periodic_image_work_and_linear_step_certificate():
    from test_explicit_backend import contact_fixture
    X,x,t,p=contact_fixture(True)
    # Separated precontact geometry, no penetration.
    x=X.copy();x[:4,0]-=.19
    c=BarrierContact(X,t,p,100.,.3,.1,thickness=.02);H=np.eye(3)
    f,extra,hist,r=c.evaluate(x,np.zeros_like(x),H,H*0,0.)
    assert r['contacts']>0
    np.testing.assert_allclose(f.sum(axis=0),0,atol=1e-13)
    eps=1e-7;plus=H.copy();minus=H.copy();plus[0,0]+=eps;minus[0,0]-=eps
    grad=(c.evaluate(x,np.zeros_like(x),plus,H*0,0.)[-1]['normal_energy']-c.evaluate(x,np.zeros_like(x),minus,H*0,0.)[-1]['normal_energy'])/(2*eps)
    assert -grad==pytest.approx(extra[0],rel=1e-6)
    c.accept_geometry(x,H);cross=x.copy();cross[:4,0]-=.03
    with pytest.raises(ContactResolutionError,match='certificate'):c.evaluate(cross,np.zeros_like(x),H,H*0,.001)


def test_barrier_parallel_edge_mollification_is_zero():
    x=np.array([[0.,0,0],[1,0,0],[0,0,.01],[1,0,.01]])
    assert mollifier(x,1e-3)==0
    assert constraint(x,'ee',10.,.02,.3,threshold=1e-3) is None


@pytest.mark.parametrize('method',['element_wave','assembled_mass_bound'])
def test_pressure_control_homogeneous_cell_matches_hencky_analytic(tmp_path,method):
    import json
    from pathlib import Path
    from scipy.optimize import brentq
    from packing_fem import sintered
    from packing_fem.explicit_material import hex_to_tet
    from packing_fem.explicit_solver import solve
    c=json.loads(Path('configs/explicit-prototype.json').read_text())
    c.update(timestep_method=method,contact_backend='distance_barrier',loading_mode='nominal_pressure',pressure_mpa=.01,length_mm=1.,yield_mpa=1e6,ramp_s=.0006,hold_s=.0003,damping_per_s=20000.)
    X,h,_=sintered.mesh(np.ones((1,1,1),bool),1.)
    r=solve(tmp_path/'pressure',X,hex_to_tet(X,h),c)
    assert all(r['checks'].values()),r
    expected=brentq(lambda stretch:-c['young_mpa']*np.log(stretch)/stretch-.01,.99,1.)
    assert 1-r['final_axial_compression']==pytest.approx(expected,abs=1e-7)


def test_vectorized_segment_search_matches_independent_scalar_search():
    from packing_fem.barrier_contact import segment_distances
    from packing_fem.contact_tracking import closest_segments
    rng=np.random.default_rng(314159);points=rng.normal(size=(100,4,3))
    points[0]=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]])
    np.testing.assert_allclose(segment_distances(points),[closest_segments(*p)['distance'] for p in points],atol=1e-13)


@pytest.mark.parametrize('kind',['pt','ee'])
def test_batched_tangent_derivative_matches_scalar_complex_reference(kind):
    from packing_fem.barrier_contact import tangent_gradient_batch
    points=np.array([[.2,.2,.01],[0,0,0],[1,0,0],[0,1,0]],float) if kind=='pt' else np.array([[-1,0,0],[1,0,0],[0,-1,.01],[0,1,.01]],float)
    old=constraint(points,kind,10.,.02,.3)['state']
    if kind=='pt':points[0,0]+=.001
    else:points[2:,0]+=.001
    branch,_=feature(points,kind);plastic=np.array([.0001,0.]);reference=np.zeros((4,3))
    for i in range(4):
        for j in range(3):
            z=points.astype(complex);z[i,j]+=1j*1e-30
            T=tangent_frame(z,kind,branch,old['aux']);spring=T.T@(old['anchor']@z)-plastic
            reference[i,j]=-np.imag(.25*10.*(spring@spring))/1e-30
    np.testing.assert_allclose(tangent_gradient_batch(points,kind,branch,old,plastic,10.,None),reference,atol=1e-14)


def test_assembled_mass_frequency_bound_exceeds_exact_elastic_spectrum():
    from packing_fem.explicit_timestep import assembled_frequency_bound
    from packing_fem.explicit_material import geometry,internal_force,advance
    from packing_fem.explicit_solver import lumped_mass
    import json
    from pathlib import Path
    c=json.loads(Path('configs/explicit-prototype.json').read_text());c['yield_mpa']=1e6
    X=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1.]])
    F=np.array([[1.01,.01,0],[0,.99,0],[0,0,1.]]);x=X@F.T;t=np.arange(4)[None]
    V0=geometry(X,t)[0];mass=lumped_mass(t,V0,1e-9,4)
    b,ep=advance(np.eye(3)[None],np.zeros(1),F[None],c);sigma=internal_force(x,t,b,ep,c)[-1];D=geometry(x,t)[2]
    stiffness=np.zeros((12,12));d=1e-7
    for j in range(12):
        delta=np.zeros_like(x);delta.flat[j]=d;forces=[]
        for sign in (-1,1):
            z=x+sign*delta;Dz=geometry(z,t)[2];bz,ez=advance(b,ep,Dz@np.linalg.inv(D),c)
            forces.append(internal_force(z,t,bz,ez,c)[0].ravel())
        stiffness[:,j]=-(forces[1]-forces[0])/(2*d)
    M=np.repeat(mass,3);weighted=stiffness/np.sqrt(M[:,None]*M[None,:])
    exact=np.linalg.eigvalsh(.5*(weighted+weighted.T)).max()
    assert assembled_frequency_bound(x,t,mass,c,sigma,b)>=exact


@pytest.mark.parametrize('kind,points',[
    ('pt',[[.3,-.005,.01],[0,0,0],[1,0,0],[0,1,0]]),
    ('pt',[[-.004,-.005,.01],[0,0,0],[1,0,0],[0,1,0]]),
    ('ee',[[0,0,0],[1,0,0],[1.005,-1,.01],[1.005,1,.01]])])
def test_friction_at_edge_vertex_and_segment_endpoint(kind,points):
    x=np.array(points,float);old=constraint(x,kind,10.,.02,.3)['state']
    if kind=='pt':x[0,0]+=.0001
    else:x[2:,1]+=.0001
    r=constraint(x,kind,10.,.02,.3,old);plastic=r['state']['plastic'];eps=1e-7
    def E(z):
        branch,_=feature(z,kind);d=np.linalg.norm(relative(z,kind,branch))
        T=tangent_frame(z,kind,branch,old['aux']);s=T.T@(old['anchor']@z)-plastic
        return barrier(d,10.,.02)[0]+.25*10.*(s@s)
    for i in range(4):
        for j in range(3):
            dz=np.zeros_like(x);dz[i,j]=eps
            assert -(E(x+dz)-E(x-dz))/(2*eps)==pytest.approx(r['force'][i,j],abs=1e-8)
    np.testing.assert_allclose(np.cross(x,r['force']).sum(axis=0),0,atol=1e-13)


def test_candidate_budget_is_not_silently_truncated():
    from test_explicit_backend import contact_fixture
    X,x,t,p=contact_fixture(False);c=BarrierContact(X,t,p,100.,.3,.1,thickness=.02,max_candidates=1)
    with pytest.raises(ValueError,match='candidate budget'):c.evaluate(X,np.zeros_like(X),np.eye(3),np.zeros((3,3)),0.)


def test_solver_contact_retry_retains_accepted_state(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    from packing_fem import sintered
    from packing_fem.explicit_material import hex_to_tet
    from packing_fem.explicit_solver import solve
    original=BarrierContact.evaluate;raised=[]
    def once(self,x,v,H,Hdot,dt):
        if dt>0 and not raised:
            raised.append(True);raise ContactResolutionError('injected rollback test')
        return original(self,x,v,H,Hdot,dt)
    monkeypatch.setattr(BarrierContact,'evaluate',once)
    c=json.loads(Path('configs/explicit-prototype.json').read_text())
    c.update(contact_backend='distance_barrier',length_mm=1.,ramp_s=.0006,hold_s=.0003,compression_fraction=.001,yield_mpa=1e6,damping_per_s=20000.)
    X,h,_=sintered.mesh(np.ones((1,1,1),bool),1.);r=solve(tmp_path/'retry',X,hex_to_tet(X,h),c)
    assert all(r['checks'].values()),r
    assert r['contact_retries']==1
    assert len(json.loads((tmp_path/'retry/contact-retries.json').read_text()))==1


def test_verlet_neighbor_cache_rebuilds_and_matches_uncached_forces():
    from test_explicit_backend import contact_fixture
    X,x,t,p=contact_fixture(True);H=np.eye(3);zero=np.zeros_like(X)
    c=BarrierContact(X,t,p,100.,0.,.1,thickness=.02)
    c.evaluate(X,zero,H,H*0,0.);assert c.neighbor_builds==1
    c.evaluate(X,zero,H,H*0,0.);assert c.neighbor_builds==1
    near=X.copy();near[:4,0]-=.19
    result=c.evaluate(near,zero,H,H*0,0.);assert c.neighbor_builds==2
    fresh=BarrierContact(X,t,p,100.,0.,.1,thickness=.02).evaluate(near,zero,H,H*0,0.)
    np.testing.assert_allclose(result[0],fresh[0],atol=1e-13)
    assert result[-1]['normal_energy']==pytest.approx(fresh[-1]['normal_energy'])


def test_pressure_unloading_keeps_permanent_deformation_and_peak_pressure(tmp_path):
    import json
    from pathlib import Path
    from packing_fem import sintered
    from packing_fem.explicit_material import hex_to_tet
    from packing_fem.explicit_solver import solve
    c=json.loads(Path('configs/explicit-prototype.json').read_text())
    c.update(contact_backend='distance_barrier',loading_mode='nominal_pressure',pressure_mpa=.08,length_mm=1.,ramp_s=.016,hold_s=.002,unload_s=.016,recovery_hold_s=.006,max_steps=100000,timestep_method='assembled_mass_bound')
    X,h,_=sintered.mesh(np.ones((1,1,1),bool),1.);r=solve(tmp_path/'unload',X,hex_to_tet(X,h),c)
    history=np.genfromtxt(tmp_path/'unload/history.csv',names=True,delimiter=',')
    assert all(r['checks'].values()),r
    assert r['checks']['path_completed'] and r['checks']['unload_completed'] and r['checks']['requested_pressure_reached']
    assert history[-1]['nominal_compression_MPa']==0
    assert r['final_axial_compression']>.001
    last=np.load(tmp_path/'unload/last-state.npz');assert last['peeq'].max()>0


def test_cached_candidate_forces_match_fresh_search_over_small_deformations():
    from test_explicit_backend import contact_fixture
    X,_,t,p=contact_fixture(True);x=X.copy();x[:4,0]-=.19;H=np.eye(3)
    c=BarrierContact(X,t,p,100.,0.,.1,thickness=.02);rng=np.random.default_rng(66)
    for step in range(10):
        x=x+rng.normal(size=x.shape)*1e-5
        cforce,_,_,cr=c.evaluate(x,np.zeros_like(x),H,H*0,0.)
        fresh=BarrierContact(X,t,p,100.,0.,.1,thickness=.02).evaluate(x,np.zeros_like(x),H,H*0,0.)
        np.testing.assert_allclose(cforce,fresh[0],atol=1e-12)
        assert cr['normal_energy']==pytest.approx(fresh[-1]['normal_energy'],abs=1e-14)


@pytest.mark.parametrize('mu',[0.,.3])
def test_near_parallel_mollified_potential_gradient_and_moment(mu):
    x=np.array([[-1.,0,0],[1,0,0],[-.5,-.005,.01],[.5,.005,.01]])
    threshold=.004;old=constraint(x,'ee',10.,.02,mu,threshold=threshold)['state']
    x[2:,0]+=.0001;r=constraint(x,'ee',10.,.02,mu,old,threshold)
    eps=1e-7;plastic=r['state']['plastic']
    def E(z):
        branch,_=feature(z,'ee');d=np.linalg.norm(relative(z,'ee',branch));normal=barrier(d,10.,.02)[0]
        tang=0.
        if mu:
            T=tangent_frame(z,'ee',branch,old['aux']);v=T.T@(old['anchor']@z)-plastic;tang=.25*10.*(v@v)
        return mollifier(z,threshold)*(normal+tang)
    for i in range(4):
        for j in range(3):
            dz=np.zeros_like(x);dz[i,j]=eps
            assert -(E(x+dz)-E(x-dz))/(2*eps)==pytest.approx(r['force'][i,j],abs=1e-8)
    np.testing.assert_allclose(np.cross(x,r['force']).sum(axis=0),0,atol=1e-13)


def test_porosity_target_controller_holds_for_equilibrium(tmp_path):
    import json
    from pathlib import Path
    from packing_fem import sintered
    from packing_fem.explicit_material import hex_to_tet
    from packing_fem.explicit_solver import solve
    c=json.loads(Path('configs/explicit-prototype.json').read_text())
    mask=np.ones((3,3,3),bool);mask[1,1,1]=False;X,h,_=sintered.mesh(mask,1.)
    c.update(contact_backend='distance_barrier',loading_mode='porosity_target',pressure_mpa=.02,target_final_porosity=1/27-5e-6,
        ramp_s=.004,hold_s=.001,timestep_method='assembled_mass_bound',max_steps=50000)
    r=solve(tmp_path/'goal',X,hex_to_tet(X,h),c)
    assert all(r['checks'].values()),r
    assert r['failure'] is None,r
    assert r['porosity_goal_equilibrated'] and r['checks']['target_porosity_at_final']
    assert r['final_porosity']<=c['target_final_porosity'] and r['peak_nominal_pressure_MPa']<c['pressure_mpa']
    assert r['checks']['static_equilibrium']


def test_active_barrier_history_survives_interior_reconnection():
    from test_contact_integration import fixture
    from tools.verify_explicit_remesh import seed_bipyramid
    from packing_fem.adaptive_tet import propose
    from packing_fem.explicit_material import geometry
    import json
    from pathlib import Path
    X,_,t,_,_=fixture();original_count=len(X);X,t=seed_bipyramid(X,t,0,.04)
    x=X.copy();x[original_count-4:original_count,2]-=.19
    c=BarrierContact(X,t,Periodic(X,3.),100.,.3,.1,thickness=.02);H=np.eye(3)*3;zero=np.zeros_like(X)
    r=c.evaluate(x,zero,H,H*0,0.);c.history=r[2];c.accept_geometry(x,H)
    x[original_count-4:original_count,0]+=.001;r=c.evaluate(x,zero,H,H*0,.001);c.history=r[2];c.accept_geometry(x,H)
    assert r[-1]['tangent_energy']>0
    cfg=json.loads(Path('configs/explicit-prototype.json').read_text());V0=geometry(X,t)[0]
    candidate=propose(x,t,np.repeat(np.eye(3)[None],len(t),axis=0),np.full(len(t),.2),V0,cfg,1.,.001)
    assert candidate is not None
    c.update_topology(candidate[0]);new=c.evaluate(x,zero,H,H*0,0.)
    assert set(new[2])==set(r[2]);np.testing.assert_allclose(new[0],r[0],atol=1e-13)
    assert new[-1]['tangent_energy']==pytest.approx(r[-1]['tangent_energy'],abs=1e-15)


def test_unreachable_porosity_goal_is_not_reported_as_solution(tmp_path):
    import json
    from pathlib import Path
    from packing_fem import sintered
    from packing_fem.explicit_material import hex_to_tet
    from packing_fem.explicit_solver import solve
    cfg=json.loads(Path('configs/explicit-prototype.json').read_text());mask=np.ones((3,3,3),bool);mask[1,1,1]=False
    X,h,_=sintered.mesh(mask,1.)
    cfg.update(contact_backend='distance_barrier',loading_mode='porosity_target',pressure_mpa=1e-6,target_final_porosity=.02,
        ramp_s=.0001,hold_s=.0001,max_goal_time_s=.002,timestep_method='assembled_mass_bound',max_steps=10000)
    r=solve(tmp_path/'unreachable',X,hex_to_tet(X,h),cfg)
    assert r['failure'] is not None and r['status']=='diagnostics_only'
    assert not r['porosity_goal_equilibrated'] and not r['checks']['target_porosity_at_final']


def test_homogeneous_pressure_plastic_equilibrium_matches_analytic(tmp_path):
    import json
    from pathlib import Path
    from scipy.optimize import brentq
    from packing_fem import sintered
    from packing_fem.explicit_material import hex_to_tet
    from packing_fem.explicit_solver import solve
    c=json.loads(Path('configs/explicit-prototype.json').read_text())
    c.update(contact_backend='distance_barrier',loading_mode='nominal_pressure',pressure_mpa=.1,length_mm=1.,ramp_s=.016,hold_s=.002,max_steps=50000,timestep_method='assembled_mass_bound')
    X,h,_=sintered.mesh(np.ones((1,1,1),bool),1.);r=solve(tmp_path/'plastic-pressure',X,hex_to_tet(X,h),c)
    expected=brentq(lambda lam:c['pressure_mpa']*lam-c['yield_mpa']-c['hardening_mpa']*(-np.log(lam)-c['pressure_mpa']*lam/c['young_mpa']),.5,1.)
    assert all(r['checks'].values()),r
    assert 1-r['final_axial_compression']==pytest.approx(expected,abs=1e-4)


def test_zero_momentum_mass_matches_exact_schur_and_virtual_work():
    from packing_fem import sintered
    X,_,_=sintered.mesh(np.ones((2,2,2),bool),1.);p=Periodic(X,2.,mass_frame='zero_momentum')
    m=np.linspace(1.,2.,len(X));p.mass(m);rng=np.random.default_rng(5);q=rng.normal(size=p.ndof)
    velocity=p.velocities(q);np.testing.assert_allclose(np.sum(m[:,None]*velocity,axis=0),0,atol=1e-13)
    assert .5*q@(p.M@q)==pytest.approx(.5*np.sum(m[:,None]*velocity**2),rel=1e-13)
    f=rng.normal(size=X.shape);assert p.force(f)@q==pytest.approx(np.sum(f*velocity),rel=1e-13)
    A=p.A.toarray();W=np.diag(np.repeat(m,3));T=np.tile(np.eye(3),(len(X),1))
    exact=A.T@(W-W@T@np.linalg.inv(T.T@W@T)@T.T@W)@A
    np.testing.assert_allclose(p.M@np.eye(p.ndof),exact,atol=1e-13)
    rhs=rng.normal(size=len(p.free));solution=p.solve_free_mass(rhs)
    np.testing.assert_allclose(exact[np.ix_(p.free,p.free)]@solution,rhs,atol=1e-12)


def test_zero_momentum_pressure_elastic_analytic_and_energy(tmp_path):
    import json
    from pathlib import Path
    from scipy.optimize import brentq
    from packing_fem import sintered
    from packing_fem.explicit_material import hex_to_tet
    from packing_fem.explicit_solver import solve
    c=json.loads(Path('configs/explicit-prototype.json').read_text())
    c.update(mass_frame='zero_momentum',loading_mode='nominal_pressure',pressure_mpa=.01,contact_backend='distance_barrier',length_mm=1.,yield_mpa=1e6,ramp_s=.0006,hold_s=.0003,damping_per_s=20000.)
    X,h,_=sintered.mesh(np.ones((1,1,1),bool),1.);r=solve(tmp_path/'zero-momentum',X,hex_to_tet(X,h),c)
    assert all(r['checks'].values()),r
    expected=brentq(lambda stretch:-c['young_mpa']*np.log(stretch)/stretch-.01,.99,1.)
    assert 1-r['final_axial_compression']==pytest.approx(expected,abs=1e-7)


def test_partial_bulk_refinement_preserves_verification_domain():
    from tools.run_barrier_compaction import fixture
    from packing_fem.explicit_material import geometry
    for pattern in ('isotropic','bulk'):
        X,t=fixture(2,pattern)
        assert np.isclose(geometry(X,t)[0].sum(),64*(1-.0025),atol=1e-12)
        assert np.allclose(np.ptp(X,axis=0),4.)
        # The thin void stays exactly between z=1.98 and z=2.02.
        center=X[t].mean(axis=1)
        assert not np.any((center[:,0]>1)&(center[:,0]<3)&(center[:,1]>1)&(center[:,1]<3)&(center[:,2]>1.98)&(center[:,2]<2.02))
