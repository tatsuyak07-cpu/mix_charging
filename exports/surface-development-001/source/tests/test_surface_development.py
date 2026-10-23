import json
from pathlib import Path
import numpy as np
import pytest
from packing_fem.area_contact import AreaContact,SurfaceResolutionError
from packing_fem.explicit_material import oriented,geometry,response
from packing_fem.explicit_periodic import Periodic,deformation
from packing_fem.surface_remesh import bisect,transfer_contact,audit
from tools.verify_plane_contact import fixture


def config():
    c=json.loads(Path('configs/explicit-prototype.json').read_text());c.update(length_mm=1.,friction=.3)
    return c


def sandwich(height=.012):
    lower=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,-1.]])
    upper=np.array([[0.,0,.03],[0,1,.03],[1,0,.03],[0,0,1.]])
    source=np.array([.3,.3,height])+np.array([[0,0,0],[.2,0,0],[0,.2,0],[0,0,.003]])
    X=np.r_[lower,upper,source]+2;t=oriented(X,np.arange(12).reshape(3,4))
    return X,t


@pytest.mark.parametrize('height',[.012,.015])
def test_multi_patch_force_is_energy_gradient_and_histories_separate(height):
    X,t=sandwich(height);c=AreaContact(X,t,Periodic(X,5.),20.,.3,.1,thickness=.02,multiple_patches=True)
    f,_,hist,r=c.evaluate(X,X*0,np.eye(3)*5,np.zeros((3,3)),0.)
    assert r['contacts']>=2
    assert sum(k[1]==8 for k in hist)==2
    assert all(k[0]=='area_patch' for k in hist)
    eps=1e-7;d=np.zeros_like(X);d[8:,2]=eps
    E=lambda x:c.evaluate(x,x*0,np.eye(3)*5,np.zeros((3,3)),0.)[-1]['normal_energy']
    assert -(E(X+d)-E(X-d))/(2*eps)==pytest.approx(f[8:,2].sum(),abs=1e-9)
    c.history=hist
    y=X.copy();y[8:,0]+=.00001
    _,_,h,r=c.evaluate(y,y*0,np.eye(3)*5,np.zeros((3,3)),.001)
    assert set(h)==set(hist) and r['tangent_energy']>0 and r['max_coulomb_ratio']<=.3+1e-12
    assert r['friction_increment']>=-1e-15


def test_default_still_rejects_multiple_patches():
    X,t=sandwich();c=AreaContact(X,t,Periodic(X,5.),20.,.3,.1,thickness=.02)
    with pytest.raises(SurfaceResolutionError):c.evaluate(X,X*0,np.eye(3)*5,np.zeros((3,3)),0.)


@pytest.mark.parametrize('n',[1,2,3,4,6])
def test_certified_reference_local_filter_retains_across_pore_force(n):
    root=Path('exports/edge-stationarity-002/preflight')
    with np.load(root/f'mesh-{n}.npz') as z:X=z['reference_points'];t=z['tets'];x=z['points']
    with np.load(root/'shape-source/last-state.npz') as z:H=4*deformation(z['generalized_position'][-6:])
    c=AreaContact(X,t,Periodic(X,4.),20.,.3,.1,thickness=.02,edge_stationarity_filter=True,certified_local_filter=True)
    f,_,_,r=c.evaluate(x,x*0,H,H*0,0.)
    assert r['contacts']>0 and r['normal_energy']>0
    np.testing.assert_allclose(f.sum(0),0,atol=1e-13)


def plane_state():
    X,t,x,H,upper=fixture(2,.015,0.)
    c=config();V0=geometry(X,t)[0];b=np.repeat(np.eye(3)[None],len(t),axis=0);ep=np.linspace(0,.2,len(t))
    # A surface edge with periodic copies; avoid cutting through particle void.
    edges=np.unique(np.sort(np.concatenate([t[:,[i,j]] for i in range(4) for j in range(i+1,4)]),axis=1),axis=0)
    edge=next(e for e in edges if np.all(np.isclose(X[e,2],.6)) and np.linalg.norm(X[e[1]]-X[e[0]])==.5)
    return X,t,x,H,upper,c,V0,b,ep,edge


def test_periodic_surface_refinement_conserves_parent_state_and_affine_fields():
    X,t,x,H,upper,c,V0,b,ep,edge=plane_state();velocity=np.tile([1.,2.,3.],(len(X),1))
    p=bisect(X,x,t,b,ep,V0,velocity,edge,1.)
    assert p['report']['new_nodes']>=1
    assert p['report']['reference_volume_error']<1e-14 and p['report']['current_volume_error']<1e-14
    np.testing.assert_array_equal(p['ep'],ep[p['parents']]);np.testing.assert_array_equal(p['b'],b[p['parents']])
    np.testing.assert_allclose(np.bincount(p['parents'],weights=p['V0'],minlength=len(t)),V0,atol=1e-16)
    from packing_fem.adaptive_tet import boundary
    # Every new boundary facet lies on an old planar boundary, and area is preserved.
    def area(z,T):
        f=z[boundary(T)];return np.linalg.norm(np.cross(f[:,1]-f[:,0],f[:,2]-f[:,0]),axis=1).sum()/2
    assert area(p['X'],p['t'])==pytest.approx(area(X,t),abs=1e-14)
    periodic=Periodic(p['X'],1.);u=p['x']-p['X']@(H.T)
    for cl in np.unique(periodic.classes):np.testing.assert_allclose(u[periodic.classes==cl],np.tile(u[periodic.classes==cl][0],(sum(periodic.classes==cl),1)),atol=1e-14)


def test_active_sticking_surface_transfer_and_conservative_audit():
    X,t,x,H,upper,c,V0,b,ep,edge=plane_state();velocity=X*0
    old=AreaContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02)
    old.history=old.evaluate(x,x*0,H,H*0,0.)[2]
    x=x.copy();H=H.copy();x[upper,0]+=.0001;H[0,2]+=.0001
    before=old.evaluate(x,x*0,H,H*0,.001);old.history=before[2]
    p=bisect(X,x,t,b,ep,V0,velocity,edge,1.)
    new=AreaContact(p['X'],p['t'],Periodic(p['X'],1.),20.,.3,.1,thickness=.02)
    after=transfer_contact(old,new,p,H)
    r=audit((X,x,t,b,ep,V0,velocity),p,c,before[-1],after[-1],1e-7,old_response=before,new_response=after)
    assert r['accepted'],r
    assert after[-1]['tangent_energy']==pytest.approx(before[-1]['tangent_energy'],rel=1e-10)
    assert after[0][p['X'][:,2]>.5].sum(0)==pytest.approx(before[0][upper].sum(0),abs=1e-12)
    # A deliberately lost slip is rejected, not silently accepted.
    bad=dict(after[-1],tangent_energy=0.)
    assert not audit((X,x,t,b,ep,V0,velocity),p,c,before[-1],bad,1e-7)['accepted']


def test_pressure_solver_commits_surface_refinement_and_matches_analytic_limit(tmp_path):
    from scipy.optimize import brentq
    from packing_fem import sintered
    from packing_fem.explicit_material import hex_to_tet
    from packing_fem.explicit_solver import solve
    c=config();c.update(contact_backend='area_guarded',loading_mode='nominal_pressure',pressure_mpa=.01,yield_mpa=1e6,ramp_s=.0006,hold_s=.0003,damping_per_s=20000.,surface_remesh_steps=[10],surface_remesh_max_events=1,certified_local_filter=True,remesh_every=0)
    X,h,_=sintered.mesh(np.ones((2,2,2),bool),.5);t=hex_to_tet(X,h)
    r=solve(tmp_path/'surface-pressure',X,t,c)
    events=[e for e in r['remesh_events'] if e['operation']=='periodic_edge_bisection']
    assert len(events)==1 and events[0]['accepted'],events
    assert all(r['checks'].values()),r
    expected=brentq(lambda lam:-c['young_mpa']*np.log(lam)/lam-.01,.99,1.)
    assert 1-r['final_axial_compression']==pytest.approx(expected,abs=1e-7)


def test_certificate_rejects_overlap_without_bypassing_local_guard():
    from packing_fem.barrier_contact import ContactResolutionError
    X,t,x,H,upper=fixture(2,.015,0.);x[upper,2]-=.02;H[2,2]-=.02
    c=AreaContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02,certified_local_filter=True)
    with pytest.raises(ContactResolutionError,match='certificate failed'):c.evaluate(x,x*0,H,H*0,0.)


def test_rejected_remesh_momentum_and_energy_are_explicit():
    X,t,x,H,upper,c,V0,b,ep,edge=plane_state();velocity=1e5*X.copy()
    p=bisect(X,x,t,b,ep,V0,velocity,edge,1.)
    empty=dict(normal_energy=0.,tangent_energy=0.,max_coulomb_ratio=0.)
    r=audit((X,x,t,b,ep,V0,velocity),p,c,empty,empty,1e-10)
    assert not r['accepted'] and not r['checks']['kinetic_energy']
    np.testing.assert_array_equal(velocity,1e5*X)


def test_failed_extension_build_rejects_transaction_without_changing_trajectory(tmp_path,monkeypatch):
    from packing_fem import sintered
    from packing_fem.explicit_material import hex_to_tet
    from packing_fem.explicit_solver import solve
    c=config();c.update(contact_backend='area_guarded',loading_mode='nominal_pressure',pressure_mpa=.01,yield_mpa=1e6,ramp_s=.0006,hold_s=.0003,damping_per_s=20000.,certified_local_filter=True,remesh_every=0)
    X,h,_=sintered.mesh(np.ones((2,2,2),bool),.5);t=hex_to_tet(X,h)
    base=solve(tmp_path/'base',X,t,c)
    def unsupported(*args):raise ValueError('Unsupported extension lineage')
    monkeypatch.setattr('packing_fem.refined_extension.refine_extension',unsupported)
    r=solve(tmp_path/'rejected',X,t,dict(c,surface_remesh_steps=[10]))
    assert not r['remesh_events'][0]['accepted']
    assert 'Unsupported extension' in r['remesh_events'][0]['reason']
    assert all(base['checks'].values()) and all(r['checks'].values())
    with np.load(tmp_path/'base/last-state.npz') as a,np.load(tmp_path/'rejected/last-state.npz') as b:
        for key in a.files:np.testing.assert_array_equal(a[key],b[key])


def test_refined_sliding_keeps_coulomb_force_and_positive_friction_work():
    X,t,x,H,upper,c,V0,b,ep,edge=plane_state();old=AreaContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02)
    old.history=old.evaluate(x,x*0,H,H*0,0.)[2]
    x=x.copy();H=H.copy();x[upper,0]+=.0001;H[0,2]+=.0001
    before=old.evaluate(x,x*0,H,H*0,.001);old.history=before[2]
    p=bisect(X,x,t,b,ep,V0,x*0,edge,1.);new=AreaContact(p['X'],p['t'],Periodic(p['X'],1.),20.,.3,.1,thickness=.02)
    transfer_contact(old,new,p,H)
    y=p['x'].copy();G=H.copy();mask=p['X'][:,2]>.5;y[mask,0]+=.01;G[0,2]+=.01
    f,_,_,r=new.evaluate(y,y*0,G,G*0,.001)
    assert r['max_coulomb_ratio']==pytest.approx(.3,abs=1e-12)
    assert r['friction_increment']>0
    assert abs(f[mask,0].sum())==pytest.approx(.3*f[mask,2].sum(),rel=1e-10)


def test_void_extension_refinement_preserves_field_and_rejects_overlap():
    from packing_fem.structured_extension import StructuredExtension
    from packing_fem.refined_extension import refine_extension
    X,t,x,H,upper,c,V0,b,ep,edge=plane_state()
    p=bisect(X,x,t,b,ep,V0,x*0,edge,1.);extension=refine_extension(StructuredExtension(X,t),p)
    r,z=extension.evaluate(p['x'],H)
    assert r['certificate_passed']
    np.testing.assert_array_equal(z['points'][extension.ids],p['x'])
    y=p['x'].copy();G=H.copy();mask=p['X'][:,2]>.5;y[mask,2]-=.02;G[2,2]-=.02
    r,_=extension.evaluate(y,G)
    assert r['minimum_material_J']>0 and not r['certificate_passed']


def test_multiple_patch_friction_is_objective_and_each_patch_is_coulomb_bounded():
    X,t=sandwich();c=AreaContact(X,t,Periodic(X,5.),20.,.3,.1,thickness=.02,multiple_patches=True)
    c.capture_audit=True;H=np.eye(3)*5;c.history=c.evaluate(X,X*0,H,H*0,0.)[2]
    x=X.copy();x[8:,0]+=.001
    f,_,_,r=c.evaluate(x,x*0,H,H*0,.001)
    for record in c.trial_audit['records']:
        assert record['kt']*np.linalg.norm(record['spring'])<=.3*record['normal_force']+1e-12
    theta=.37;R=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    y=x@R.T+[.1,-.2,.3];rot=c.evaluate(y,y*0,R@H,H*0,.001)
    np.testing.assert_allclose(rot[0],f@R.T,atol=1e-11)
    assert rot[-1]['tangent_energy']==pytest.approx(r['tangent_energy'],abs=1e-13)
