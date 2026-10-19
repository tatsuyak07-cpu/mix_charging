import json
from pathlib import Path
import numpy as np
from packing_fem.explicit_audit import whole_path_audit


def test_initial_contact_peak_and_empty_contact_are_distinguished():
    a=np.zeros((4,11));a[:,0]=[0,.1,.2,.3];a[:,3]=[0,.02,.1,.1];a[:,4]=[0,.04,.00001,0];a[:,5]=[0,.01,.02,.02];a[:,8]=[0,.01,.02,.02];a[:,10]=[0,8,8,8]
    r=whole_path_audit(a,1.)
    assert r['contact']['peak_kinetic_Nmm']==.04
    assert r['contact']['peak_kinetic_time_s']==.1
    assert r['contact']['peak_kinetic_internal_contact_ratio']==2.
    a[:,10]=0
    assert whole_path_audit(a,1.)['contact'] is None


def test_output_cadence_cannot_change_accepted_step_audit(tmp_path):
    from packing_fem.sintered import mesh
    from packing_fem.explicit_material import hex_to_tet
    from packing_fem.explicit_solver import solve
    X,h,_=mesh(np.ones((1,1,1),bool),1.);t=hex_to_tet(X,h)
    c=json.loads(Path('configs/explicit-area-cauchy-pressure.json').read_text());c.update(length_mm=1.,pressure_mpa=.02,ramp_s=.0001,hold_s=.00005,max_steps=5000,remesh_every=0)
    results=[];audits=[]
    for every in (1,53):
        directory=tmp_path/f'every{every}';result=solve(directory,X,t,dict(c,output_every=every));results.append(result)
        a=np.genfromtxt(directory/'accepted-step-audit.csv.gz',delimiter=',',names=True);audits.append(a)
        assert len(a)==result['steps']+1
        assert result['failure'] is None
    np.testing.assert_array_equal(audits[0],audits[1])
    assert results[0]['whole_path_audit']==results[1]['whole_path_audit']
