import json
from pathlib import Path
import numpy as np
import pytest
from copy import deepcopy
from packing_fem.area_contact import AreaContact
from packing_fem.explicit_material import geometry
from packing_fem.explicit_periodic import Periodic,affine_basis
from packing_fem.surface_remesh import bisect,transfer_contact,audit
from tools.verify_plane_contact import fixture


def plane(slip=.0001):
    X,t,x,H,upper=fixture(2,.015,.25)
    c=json.loads(Path('configs/explicit-prototype.json').read_text());c.update(length_mm=1.,friction=.3)
    V=geometry(X,t)[0];b=np.repeat(np.eye(3)[None],len(t),axis=0);ep=np.linspace(0,.2,len(t))
    old=contact(X,t);old.history=old.evaluate(x,x*0,H,H*0,0.)[2]
    x=x.copy();H=H.copy();x[upper,0]+=slip;H[0,2]+=slip
    before=old.evaluate(x,x*0,H,H*0,.001);old.history=before[2]
    edges=np.unique(np.sort(np.concatenate([t[:,[i,j]] for i in range(4) for j in range(i+1,4)]),axis=1),axis=0)
    edge=next(e for e in edges if np.all(np.isclose(X[e,2],.6)) and np.linalg.norm(X[e[1]]-X[e[0]])==.5)
    return X,t,x,H,upper,c,V,b,ep,old,before,edge


def contact(X,t):return AreaContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02,quadrature_mode='material')


@pytest.mark.parametrize('fraction',[.2,.5,.8])
@pytest.mark.parametrize('slip',[.0001,.01])
def test_material_weights_and_stick_slip_history_survive_refinement(fraction,slip):
    X,t,x,H,upper,c,V,b,ep,old,before,edge=plane(slip)
    states={k:{n:np.copy(v) if isinstance(v,np.ndarray) else v for n,v in s.items()} for k,s in old.history.items()}
    p=bisect(X,x,t,b,ep,V,x*0,edge,1.,fraction=fraction);new=contact(p['X'],p['t'])
    after=transfer_contact(old,new,p,H)
    report=audit((X,x,t,b,ep,V,x*0),p,c,before[-1],after[-1],1e-7,old_response=before,new_response=after)
    assert report['accepted'],report
    np.testing.assert_array_equal(new.integration_area0[:len(X)],old.integration_area0)
    assert not np.any(new.integration_area0[len(X):])
    assert set(new.history)==set(old.history)
    assert after[-1]['tangent_energy']>0
    for key in states:
        np.testing.assert_allclose(new.history[key]['spring'],states[key]['spring'],atol=1e-14)
        np.testing.assert_allclose(new.history[key]['plastic'],states[key]['plastic'],atol=1e-14)
        np.testing.assert_array_equal(old.history[key]['spring'],states[key]['spring'])
    # New target-basis forces restrict by the transpose of the geometric
    # interpolation to the same force/work on the original FE space.
    restricted=np.zeros_like(x)
    for j in (0,1):np.add.at(restricted,p['supports'][:,j],p['weights'][:,j,None]*after[0])
    np.testing.assert_allclose(restricted,before[0],atol=1e-11)
    assert after[-1]['normal_energy']==pytest.approx(before[-1]['normal_energy'],rel=1e-12)
    assert after[-1]['tangent_energy']==pytest.approx(before[-1]['tangent_energy'],rel=1e-10)


def test_next_sliding_step_has_same_work_force_and_energy_after_refinement():
    X,t,x,H,upper,c,V,b,ep,old,before,edge=plane()
    p=bisect(X,x,t,b,ep,V,x*0,edge,1.,fraction=.2);new=contact(p['X'],p['t']);transfer_contact(old,new,p,H)
    y=x.copy();y[upper,0]+=.01;z=p['x'].copy();mask=p['X'][:,2]>.5;z[mask,0]+=.01
    G=H.copy();G[0,2]+=.01
    a=old.evaluate(y,y*0,G,G*0,.001);d=new.evaluate(z,z*0,G,G*0,.001)
    for key in ('normal_energy','tangent_energy','friction_increment'):
        assert d[-1][key]==pytest.approx(a[-1][key],rel=1e-9,abs=1e-14)
    assert d[-1]['friction_increment']>0 and d[-1]['max_coulomb_ratio']==pytest.approx(.3,abs=1e-12)
    np.testing.assert_allclose(d[0][mask].sum(0),a[0][upper].sum(0),atol=1e-11)


def test_material_points_keep_nonuniform_elastic_slip_without_averaging():
    X,t,x,H,upper,c,V,b,ep,old,before,edge=plane()
    for key,s in old.history.items():
        s['spring']=s['spring']*(.2+.6*X[key[1],0]);s['plastic']=s['coordinate']-s['spring']
    before=old.evaluate(x,x*0,H,H*0,0.);old.history=before[2]
    p=bisect(X,x,t,b,ep,V,x*0,edge,1.,fraction=.2);new=contact(p['X'],p['t']);after=transfer_contact(old,new,p,H)
    for key in old.history:np.testing.assert_allclose(new.history[key]['spring'],old.history[key]['spring'],atol=1e-14)
    assert after[-1]['tangent_energy']==pytest.approx(before[-1]['tangent_energy'],rel=1e-10)


def test_mode_mismatch_and_missing_source_lineage_are_rejected():
    X,t,x,H,upper,c,V,b,ep,old,before,edge=plane();p=bisect(X,x,t,b,ep,V,x*0,edge,1.)
    wrong=AreaContact(p['X'],p['t'],Periodic(p['X'],1.),20.,.3,.1,thickness=.02)
    with pytest.raises(ValueError,match='mode'):transfer_contact(old,wrong,p,H)
    new=contact(p['X'],p['t']);bad=dict(p,supports=p['supports'].copy());bad['supports'][0]=[0,1]
    with pytest.raises(ValueError,match='retained'):transfer_contact(old,new,bad,H)


def test_applied_force_replay_is_objective_and_does_not_reset_committed_history():
    X,t,x,H,upper,c,V,b,ep,old,before,edge=plane(.01)
    old.accept_geometry(x,H);committed=deepcopy(old.history)
    replay=old.replay_force_state(x,x*0,H)
    np.testing.assert_allclose(replay[0],before[0],atol=1e-12)
    for key in committed:
        for field in ('plastic','spring','coordinate','anchor_ids'):np.testing.assert_array_equal(old.history[key][field],committed[key][field])
    p=bisect(X,x,t,b,ep,V,x*0,edge,1.,fraction=.2);new=contact(p['X'],p['t']);after=transfer_contact(old,new,p,H)
    theta=.37;R=np.array([[np.cos(theta),-np.sin(theta),0.],[np.sin(theta),np.cos(theta),0.],[0,0,1.]])
    rotated=new.replay_force_state(p['x']@R.T+[.1,-.2,.3],p['x']*0,R@H)
    np.testing.assert_allclose(rotated[0],after[0]@R.T,atol=1e-11)
    assert rotated[-1]['tangent_energy']==pytest.approx(after[-1]['tangent_energy'],rel=1e-10)


def test_repeated_refinement_does_not_add_integration_points_or_change_work():
    X,t,x,H,upper,c,V,b,ep,old,before,edge=plane(.01)
    initial=old.integration_area0.copy();reference_energy=before[-1]['normal_energy']+before[-1]['tangent_energy']
    for _ in range(3):
        n=len(X);p=bisect(X,x,t,b,ep,V,x*0,edge,1.);new=contact(p['X'],p['t']);after=transfer_contact(old,new,p,H)
        assert after[-1]['normal_energy']+after[-1]['tangent_energy']==pytest.approx(reference_energy,rel=1e-11)
        np.testing.assert_array_equal(new.integration_area0[:len(initial)],initial)
        assert not np.any(new.integration_area0[len(initial):])
        X,t,x,b,ep,V=p['X'],p['t'],p['x'],p['b'],p['ep'],p['V0'];old=new;edge=np.array([edge[0],n])


@pytest.mark.parametrize('capture',[False,True])
def test_pressure_solver_material_mode_commits_and_saves_quadrature(tmp_path,capture):
    from packing_fem.sintered import mesh
    from packing_fem.explicit_material import hex_to_tet
    from packing_fem.explicit_solver import solve
    X,h,_=mesh(np.ones((2,2,2),bool),.5);t=hex_to_tet(X,h)
    c=json.loads(Path('configs/explicit-prototype.json').read_text())
    c.update(length_mm=1.,contact_backend='area_guarded',contact_quadrature_mode='material',loading_mode='nominal_pressure',pressure_mpa=.01,yield_mpa=1e6,ramp_s=.0006,hold_s=.0003,damping_per_s=20000.,surface_remesh_steps=[10,20],surface_remesh_max_events=2,remesh_every=0,save_contact_audit=capture)
    r=solve(tmp_path/'pressure',X,t,c)
    assert all(r['checks'].values()),r
    assert len(r['remesh_events'])==2 and all(e['accepted'] for e in r['remesh_events'])
    with np.load(tmp_path/'pressure/contact-quadrature.npz') as z:assert z['mode'].item()=='material'


def test_unresolved_integration_is_explicit_and_preserves_existing_history():
    root=Path('exports/closest-selection-001/validation/pressure-run/calculation')
    with np.load(root/'last-state.npz') as z:X=z['reference_points'];x=z['points'];t=z['tets'];q=z['generalized_position']
    from packing_fem.explicit_periodic import deformation
    capture=json.loads((root/'contact-audit.json').read_text());H=12.1*deformation(q[-6:])
    c=AreaContact(X,t,Periodic(X,12.1),20.,.3,.1,thickness=.02,multiple_patches=True,edge_stationarity_filter=True,closest_tie_rule='clarke',quadrature_mode='material')
    c.history={tuple(v['key']):{k:np.array(a) if isinstance(a,list) else a for k,a in v['state'].items()} for v in capture['committed_history']}
    before=deepcopy(c.history)
    from packing_fem.area_contact import SurfaceResolutionError
    with pytest.raises(SurfaceResolutionError,match='integration refinement') as error:c.evaluate(x,x*0,H,H*0,0.)
    assert error.value.feature['requires_quadrature_refinement']
    for key in before:
        for field in ('plastic','spring','coordinate','anchor_ids'):np.testing.assert_array_equal(c.history[key][field],before[key][field])


def test_force_and_committed_histories_roundtrip_separately(tmp_path):
    from packing_fem.contact_audit import save_capture
    X,t,x,H,upper,c,V,b,ep,old,before,edge=plane(.01)
    old.accept_geometry(x,H);old.capture_audit=True
    response=old.replay_force_state(x,x*0,H);old.accept_geometry(x,H)
    count=Periodic(X,1.).ndof
    path=tmp_path/'capture.json';save_capture(path,old,np.zeros(count),np.zeros(count),0.,1)
    data=json.loads(path.read_text());assert data['applied_force_history'] and data['committed_history']
    decode=lambda rows:{tuple(row['key']):{k:np.array(v) if isinstance(v,list) else v for k,v in row['state'].items()} for row in rows}
    restored=contact(X,t);restored.history=decode(data['committed_history']);restored.accepted_force_history=decode(data['applied_force_history'])
    again=restored.replay_force_state(x,x*0,H)
    np.testing.assert_allclose(again[0],response[0],atol=1e-12)
    for key in restored.history:np.testing.assert_array_equal(restored.history[key]['plastic'],old.history[key]['plastic'])
