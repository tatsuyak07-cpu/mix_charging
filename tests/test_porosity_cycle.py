import json
from pathlib import Path
import numpy as np
import pytest
from packing_fem.porosity_cycle import PorosityCycle
from packing_fem.checkpoint import save,load


def config():
    c=json.loads(Path('configs/explicit-prototype.json').read_text())
    c.update(loading_mode='porosity_cycle',target_final_porosity=.1,porosity_tolerance=.001,
             strain_increment=.025,minimum_strain_increment=1e-5,maximum_axial_compression=.8,
             maximum_hold_s=.1,unload_s=.008,recovery_hold_s=.002,maximum_cycle_time_s=4.)
    return c


def test_cycle_commands_are_smooth_and_resume_is_exact(tmp_path):
    c=config();p=PorosityCycle(c)
    for time in (0.,c['ramp_s']):
        _,v,a=p.displacement(time);assert abs(v)<1e-12 and abs(a)<1e-12
    time=c['ramp_s']*.37;dt=1e-8
    numerical=(p.displacement(time+dt)[0]-p.displacement(time-dt)[0])/(2*dt)
    assert numerical==pytest.approx(p.displacement(time)[1],rel=1e-7)
    save(tmp_path/'state',p.state,{})
    restored=PorosityCycle(c,load(tmp_path/'state')[0])
    assert restored.displacement(time)==p.displacement(time)


def test_target_requires_loaded_equilibrium_then_zero_force_recovery():
    c=config();p=PorosityCycle(c);T=c['ramp_s'];H=c['hold_s']
    p.observe(T,.1,.025,12.,0.,0.,0.)
    assert p.observe(T+H,.1,.025,12.,1.,0.,0.) is None
    assert not p.state['loaded_target']
    assert p.observe(T+H+.001,.1,.025,12.,0.,0.,0.)=='release'
    p.release(T+H+.001,12.)
    assert p.external_force(p.state['start'])==12.
    end=p.interval_end();assert p.external_force(end)==0.
    p.observe(end,.12,.02,0.,0.,0.,0.)
    assert p.observe(end+c['recovery_hold_s']+1e-12,.12,.02,0.,0.,0.,0.)=='complete'
    assert p.state['loaded_porosity']==.1 and p.state['records'][-1]['porosity']==.12


def test_target_overshoot_is_rejected_instead_of_being_called_a_solution():
    c=config();p=PorosityCycle(c);p.observe(c['ramp_s'],.09,.025,12.,0.,0.,0.)
    with pytest.raises(RuntimeError,match='overshot'):
        p.observe(c['ramp_s']+c['hold_s']+1e-12,.09,.025,12.,0.,0.,0.)
    assert not p.state['loaded_target']


def test_checkpoint_blocks_keep_many_small_arrays_and_aliases(tmp_path):
    shared=np.arange(3.);state=dict(points=[np.array([i,i+1,i+2.]) for i in range(2000)],shared=[shared,shared])
    save(tmp_path/'packed',state,{})
    with np.load(tmp_path/'packed/arrays.npz') as a:assert len(a.files)==1
    loaded,_=load(tmp_path/'packed');assert loaded['shared'][0] is loaded['shared'][1]
    for a,b in zip(state['points'],loaded['points']):np.testing.assert_array_equal(a,b)


def test_elastic_voided_cell_reaches_loaded_target_and_recovers(tmp_path):
    from packing_fem.sintered import mesh
    from packing_fem.explicit_material import hex_to_tet
    from packing_fem.explicit_solver import solve
    mask=np.ones((2,2,2),bool)
    for index in ((0,0,0),(1,0,0),(0,1,0)):mask[index]=False
    X,h,_=mesh(mask,.5);t=hex_to_tet(X,h);c=config()
    c.update(length_mm=1.,target_final_porosity=.37,porosity_tolerance=.001,strain_increment=.005,
             maximum_axial_compression=.3,ramp_s=.0006,hold_s=.0003,maximum_hold_s=.01,
             unload_s=.001,recovery_hold_s=.0005,maximum_cycle_time_s=.2,yield_mpa=1e6,
             damping_per_s=20000.,contact_backend='embedded_area',remesh_every=0,max_steps=100000,
             output_every=20,save_checkpoint=True)
    r=solve(tmp_path/'elastic-void-control',X,t,c)
    assert r['checks']['cycle_completed'],r
    assert r['checks']['loaded_porosity_equilibrated']
    assert .369<=r['porosity_cycle']['loaded_porosity']<=.37
    assert r['final_porosity']==pytest.approx(.375,abs=2e-4)
    assert abs(r['final_axial_compression'])<2e-4
