import json
from pathlib import Path
import numpy as np
import pytest
from packing_fem.explicit_solver import solve
from packing_fem.explicit_material import hex_to_tet
from packing_fem.sintered import mesh
from packing_fem.checkpoint import load,save


def test_parent_operator_switch_requires_free_prefix_and_explicit_update(tmp_path):
    X,h,_=mesh(np.ones((2,2,2),bool),.5);t=hex_to_tet(X,h)
    c=json.loads(Path('configs/explicit-prototype.json').read_text());c.update(length_mm=1.,contact_backend='embedded_area',surface_error_budget='normalized',loading_mode='cauchy_pressure',pressure_mpa=.01,yield_mpa=1e6,ramp_s=.0006,hold_s=.0003,damping_per_s=20000.,remesh_every=0,output_every=5,save_checkpoint=True)
    stopped=solve(tmp_path/'original',X,t,dict(c,checkpoint_stop_after=5));cp=stopped['checkpoint_paths'][-1]
    changed=dict(c,resume_checkpoint=cp,surface_audit_mode='parent',surface_redistribution='conservative',surface_velocity_transfer='momentum_energy',active_embedded_surface_remesh=True)
    with pytest.raises(ValueError,match='configuration mismatch'):solve(tmp_path/'implicit-rejected',X,t,changed)
    result=solve(tmp_path/'free-updated',X,t,dict(changed,allow_contact_operator_update=True,allow_solver_update_at_inactive_checkpoint=True))
    assert result['checks']['path_completed']
    state,metadata=load(cp);state['contact_free_prefix']=False;save(tmp_path/'past-contact',state,metadata)
    with pytest.raises(ValueError,match='configuration mismatch'):solve(tmp_path/'active-rejected',X,t,dict(changed,resume_checkpoint=str(tmp_path/'past-contact'),allow_contact_operator_update=True,allow_solver_update_at_inactive_checkpoint=True))
