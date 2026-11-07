import csv,json
import numpy as np
import pytest
from packing_fem.explicit_material import geometry,hex_to_tet
from packing_fem.explicit_periodic import Periodic
from tools.audit_compaction_history import audit


@pytest.mark.parametrize('corrupt_history',[False,True])
def test_audit_distinguishes_elastic_volume_change_from_inconsistent_history(tmp_path,corrupt_history):
    X=np.array([[i,j,k] for i in range(2) for j in range(2) for k in range(2)],float)
    t=hex_to_tet(X,np.array([[0,4,6,2,1,5,7,3]]));p=Periodic(X,1.);V0=geometry(X,t)[0]
    run=tmp_path/'run';run.mkdir()
    c=dict(length_mm=1.,young_mpa=10.,poisson=.33,yield_mpa=1e6,hardening_mpa=0.,transverse_boundary='fixed_cell',implicit_force_tolerance_N=.005,implicit_macro_tolerance=1e-6)
    (run/'config.json').write_text(json.dumps(c))
    for step,stretch in enumerate([1.,.8]):
        q=np.zeros(p.ndof);q[p.prescribed]=stretch-1
        elastic_stretch=1. if corrupt_history and step else stretch
        b=np.tile(np.diag([1.,1.,elastic_stretch**2]),(len(t),1,1))
        np.savez(run/f'state-{step:04d}.npz',X=X,t=t,q=q,x=p.positions(q),b=b,ep=np.zeros(len(t)),V0=V0,parent=np.arange(len(t)))
    with (run/'history.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['step','porosity','free_residual_N','macro_target_error']);writer.writeheader();writer.writerow(dict(step=1,porosity=0.,free_residual_N=0.,macro_target_error=0.))
    out=tmp_path/'audit.json'
    assert audit(run,out) is not corrupt_history
    report=json.loads(out.read_text());last=report['states'][-1]
    assert last['reference_material_relative_error']<1e-12
    assert abs(last['solid_volume_to_initial_ratio']-.8)<1e-12
