import json
from pathlib import Path
import numpy as np
import pytest
from packing_fem.checkpoint import load,save
from packing_fem.explicit_solver import solve
from packing_fem.sintered import mesh
from packing_fem.explicit_material import hex_to_tet


def test_prescribed_inactive_cycle_operator_update_checks_acceleration(tmp_path):
    c=json.loads(Path('configs/packing-porosity-cycle.json').read_text())
    c.update(length_mm=1.,element_backend='numpy',max_steps=100,ramp_s=.0006,hold_s=.0003,
             checkpoint_stop_after=5,remesh_every=0,volume_reconnections=False,output_every=1,
             yield_mpa=1e6,requested_initial_porosity=.1)
    mask=np.ones((2,2,2),bool);mask[0,0,0]=False;X,h,_=mesh(mask,.5);t=hex_to_tet(X,h)
    result=solve(tmp_path/'old',X,t,c);state,metadata=load(result['checkpoint_paths'][-1])
    assert state['contact_free_prefix'] and state['prescribed_active']
    metadata['implementation_sha256']['explicit_solver.py']='older-inactive-cycle-operator'
    save(tmp_path/'changed',state,metadata)
    resume=dict(c,resume_checkpoint=str(tmp_path/'changed'),allow_contact_operator_update=True,allow_solver_update_at_inactive_checkpoint=True)
    new=solve(tmp_path/'new',X,t,resume)
    assert 'explicit_solver.py' in new['checkpoint_contact_operator_update']
    state['a'][0]+=1.
    save(tmp_path/'corrupt',state,metadata)
    with pytest.raises(ValueError,match='acceleration consistency'):
        solve(tmp_path/'rejected',X,t,dict(resume,resume_checkpoint=str(tmp_path/'corrupt')))
