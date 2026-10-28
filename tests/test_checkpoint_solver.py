import json
from pathlib import Path
import numpy as np
import pytest
from packing_fem.checkpoint import save,load
from packing_fem.explicit_solver import solve
from packing_fem.sintered import mesh
from packing_fem.explicit_material import hex_to_tet


def model():
    X,h,_=mesh(np.ones((2,2,2),bool),.5);t=hex_to_tet(X,h)
    c=json.loads(Path('configs/explicit-prototype.json').read_text())
    c.update(length_mm=1.,contact_backend='embedded_area',loading_mode='cauchy_pressure',pressure_mpa=.01,yield_mpa=1e6,ramp_s=.0006,hold_s=.0003,damping_per_s=20000.,remesh_every=0,output_every=5,save_checkpoint=True)
    return X,t,c


def test_pickle_free_checkpoint_preserves_keys_arrays_and_checksums(tmp_path):
    state={('area',7):dict(spring=np.array([.01,.02]),peeq=np.arange(3),limit=float('inf')), 'history':[1.,2.]}
    save(tmp_path/'cp',state,dict(config={'friction':.3}))
    restored,metadata=load(tmp_path/'cp');np.testing.assert_array_equal(restored[('area',7)]['spring'],state[('area',7)]['spring'])
    assert metadata['config']['friction']==.3
    p=tmp_path/'cp/arrays.npz';raw=p.read_bytes();p.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
    with pytest.raises(ValueError,match='checksum'):load(tmp_path/'cp')


def test_complete_solver_restart_matches_uninterrupted_trajectory(tmp_path):
    X,t,c=model();whole=solve(tmp_path/'whole',X,t,c)
    stopped=solve(tmp_path/'stopped',X,t,dict(c,checkpoint_stop_after=10))
    assert not stopped['checks']['path_completed']
    resumed=solve(tmp_path/'resumed',X,t,dict(c,resume_checkpoint=stopped['checkpoint_paths'][-1]))
    with np.load(tmp_path/'whole/last-state.npz') as a,np.load(tmp_path/'resumed/last-state.npz') as b:
        for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    np.testing.assert_array_equal(np.loadtxt(tmp_path/'whole/accepted-step-audit.csv.gz',delimiter=',',skiprows=1),np.loadtxt(tmp_path/'resumed/accepted-step-audit.csv.gz',delimiter=',',skiprows=1))
    assert whole['checks']==resumed['checks']
    with pytest.raises(ValueError,match='configuration mismatch'):
        solve(tmp_path/'wrong',X,t,dict(c,friction=.2,resume_checkpoint=stopped['checkpoint_paths'][-1]))


def test_pressure_stages_hold_before_advancing(tmp_path):
    X,t,c=model();c.update(pressure_mpa=.1,pressure_stages=[dict(pressure_mpa=P,ramp_s=.0006,hold_s=.0003,max_hold_s=.002) for P in (.001,.01,.1)])
    result=solve(tmp_path/'stages',X,t,c)
    assert len(result['pressure_stages'])==3,result
    assert all(r['passed'] for r in result['pressure_stages'])
    assert result['checks']['requested_pressure_reached']
    assert result['checks']['path_completed'],result['failure']


def test_new_quadrature_replays_only_from_contact_free_prefix(tmp_path):
    from tools.verify_plane_contact import fixture
    from packing_fem.embedded_contact import EmbeddedContact
    from packing_fem.explicit_periodic import Periodic
    from packing_fem.triangle_quadrature import RULES
    X,t,_,_,_=fixture(1,.015,0.)
    _,_,c=model();c.update(pressure_mpa=.001,save_checkpoint=True)
    stop=solve(tmp_path/'coarse',X,t,dict(c,checkpoint_stop_after=5))
    reference=EmbeddedContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02)
    face=next(i for i,f in enumerate(reference.faces) if np.all(np.isclose(X[f,2],.6)))
    B,w=RULES[4];tri=X[reference.faces[face]];A=np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))/2
    rules={str(face):dict(bary=B.tolist(),weights=(A*w).tolist())}
    fine=solve(tmp_path/'fine',X,t,dict(c,embedded_quadrature_rules=rules))
    replay=solve(tmp_path/'replay',X,t,dict(c,embedded_quadrature_rules=rules,resume_checkpoint=stop['checkpoint_paths'][-1],inactive_quadrature_replay=True))
    with np.load(tmp_path/'fine/last-state.npz') as a,np.load(tmp_path/'replay/last-state.npz') as b:
        for key in a.files:np.testing.assert_allclose(a[key],b[key],rtol=1e-12,atol=1e-13)
    assert fine['checks']==replay['checks']
    with pytest.raises(ValueError,match='contact-free'):
        solve(tmp_path/'unsafe',X,t,dict(c,embedded_quadrature_rules=rules,resume_checkpoint=stop['checkpoint_paths'][-1]))


def test_contact_operator_update_requires_free_prefix_and_unchanged_bulk(tmp_path):
    X,t,c=model();stopped=solve(tmp_path/'old',X,t,dict(c,checkpoint_stop_after=5))
    state,metadata=load(stopped['checkpoint_paths'][-1]);metadata['implementation_sha256']['surface_patch.py']='older-contact-operator'
    save(tmp_path/'compatible',state,metadata)
    with pytest.raises(ValueError,match='implementation'):
        solve(tmp_path/'rejected',X,t,dict(c,resume_checkpoint=str(tmp_path/'compatible')))
    result=solve(tmp_path/'updated',X,t,dict(c,resume_checkpoint=str(tmp_path/'compatible'),allow_contact_operator_update=True))
    assert result['checkpoint_contact_operator_update']==['surface_patch.py']
    metadata['implementation_sha256']['explicit_material.py']='different-bulk-law'
    save(tmp_path/'incompatible',state,metadata)
    with pytest.raises(ValueError,match='implementation'):
        solve(tmp_path/'bulk-rejected',X,t,dict(c,resume_checkpoint=str(tmp_path/'incompatible'),allow_contact_operator_update=True))


def test_inactive_solver_update_checks_physical_state_before_continuation(tmp_path):
    X,t,c=model();stopped=solve(tmp_path/'old-solver',X,t,dict(c,checkpoint_stop_after=5))
    state,metadata=load(stopped['checkpoint_paths'][-1]);metadata['implementation_sha256']['explicit_solver.py']='older-equivalent-solver'
    save(tmp_path/'old-state',state,metadata)
    flags=dict(resume_checkpoint=str(tmp_path/'old-state'),allow_contact_operator_update=True,allow_solver_update_at_inactive_checkpoint=True)
    result=solve(tmp_path/'updated-solver',X,t,dict(c,**flags))
    assert 'explicit_solver.py' in result['checkpoint_contact_operator_update']
    state['f_int']=state['f_int']+1.
    save(tmp_path/'corrupt-physical-state',state,metadata)
    with pytest.raises(ValueError,match='physical-state consistency'):
        solve(tmp_path/'rejected-physical-state',X,t,dict(c,**dict(flags,resume_checkpoint=str(tmp_path/'corrupt-physical-state'))))


def test_saved_internal_force_matches_current_material_state(tmp_path):
    from packing_fem.explicit_material import internal_force
    X,t,c=model();stopped=solve(tmp_path/'saved-force',X,t,dict(c,checkpoint_stop_after=10))
    state,metadata=load(stopped['checkpoint_paths'][-1])
    actual=internal_force(state['x'],state['t'],state['b'],state['ep'],c)[0]
    np.testing.assert_allclose(state['f_int'],actual,rtol=1e-12,atol=1e-14)
    assert metadata['checkpoint_fields_version']==2


def test_embedded_inactive_remesh_is_audited_and_checkpointed(tmp_path):
    X,t,c=model();c.update(surface_remesh_steps=[10,20],surface_remesh_max_events=2)
    result=solve(tmp_path/'remesh',X,t,c)
    assert len(result['remesh_events'])==2,result
    assert all(e['accepted'] and e.get('inactive_quadrature_reinitialized') for e in result['remesh_events']),result['remesh_events']
    state,_=load(result['checkpoint_paths'][-1])
    assert len(state['X'])>len(X)
    assert state['runtime_rules']=={}
    assert state['contact_free_prefix']


def test_interrupted_checkpoint_is_preserved_and_recovered_exclusively(tmp_path,monkeypatch):
    from packing_fem import checkpoint
    real_save=checkpoint.save;calls=[]
    def interrupt_once(path,state,metadata):
        calls.append(path)
        if len(calls)==1:
            Path(path).mkdir();(Path(path)/'partial-write.txt').write_text('interrupted')
            raise KeyboardInterrupt
        return real_save(path,state,metadata)
    monkeypatch.setattr(checkpoint,'save',interrupt_once)
    X,t,c=model();result=solve(tmp_path/'interrupted',X,t,dict(c,checkpoint_every=5))
    assert result['failure']=='KeyboardInterrupt()'
    assert (calls[0]/'partial-write.txt').read_text()=='interrupted'
    assert not (calls[0]/'manifest.json').exists()
    saved,_=load(result['checkpoint_paths'][-1]);assert saved['steps']==5
    assert 'recovery' in result['checkpoint_paths'][-1]


def test_orderly_stop_file_saves_a_complete_accepted_state(tmp_path):
    stop=tmp_path/'STOP';stop.write_text('stop after current accepted step')
    X,t,c=model();result=solve(tmp_path/'orderly',X,t,dict(c,stop_request_file=str(stop)))
    saved,_=load(result['checkpoint_paths'][-1])
    assert saved['steps']==0 and 'orderly stop' in result['failure']
    resumed=solve(tmp_path/'continue',X,t,dict(c,resume_checkpoint=result['checkpoint_paths'][-1]))
    assert resumed['checks']['path_completed']


def test_hold_budget_extension_preserves_failed_attempt_and_continues_same_load(tmp_path):
    X,t,c=model();c.update(pressure_mpa=.001,ramp_s=.0002,force_tolerance=1e-6,
                          pressure_stages=[dict(pressure_mpa=.001,ramp_s=.0002,hold_s=.000001,max_hold_s=.000001)])
    old=solve(tmp_path/'short-hold',X,t,c)
    assert old['failure'] and 'equilibration budget' in old['failure']
    assert not old['pressure_stages'][-1]['passed']
    extended=dict(c,pressure_stages=[dict(c['pressure_stages'][0],max_hold_s=.002)],
                  resume_checkpoint=old['checkpoint_paths'][-1],extend_stage_hold_budget=True)
    result=solve(tmp_path/'extended',X,t,extended)
    assert result['hold_budget_extended']
    assert result['checks']['path_completed']
    assert result['pressure_stages'][-1]['passed']
    assert not result['pressure_stages'][0]['passed']
    changed=dict(extended,pressure_mpa=.002,pressure_stages=[dict(extended['pressure_stages'][0],pressure_mpa=.002)])
    with pytest.raises(ValueError,match='hold budgets'):
        solve(tmp_path/'changed-load-rejected',X,t,changed)
