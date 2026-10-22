# Reconstructed initial failing setup from recorded edits; see contact-audit-tests-001.log for the original failure.
import json
from pathlib import Path
import numpy as np
from packing_fem.area_contact import AreaContact
from packing_fem.explicit_periodic import Periodic
from packing_fem.contact_audit import save_capture,reassemble
from tools.verify_plane_contact import fixture


def test_saved_sticking_and_sliding_witnesses_reproduce_force_and_images(tmp_path):
    for slip in (.0001,.03):
        X,t,x,H,upper=fixture(1,.015,.25);p=Periodic(X,1.)
        c=AreaContact(X,t,p,20.,.3,.1,thickness=.02);c.capture_audit=True
        c.history=c.evaluate(x,x*0,H,H*0,0.)[2];c.accept_geometry(x,H)
        x=x.copy();H=H.copy();x[upper,0]+=slip;H[0,2]+=slip
        f,e,h,r=c.evaluate(x,x*0,H,H*0,.001);c.history=h;c.accept_geometry(x,H)
        file=tmp_path/f'{slip}.json';save_capture(file,c,p.force(f,e),e,1.,1);data=json.loads(file.read_text());data['reference_length_mm']=1.
        assert len(data['committed_history'])==r['contacts']>0
        assert any(np.linalg.norm(k['spring'])>0 for k in data['accepted']['records'])
        for step in (1e-6,5e-7):
            fn,en=reassemble(data,step);np.testing.assert_allclose(fn,f,atol=2e-8,rtol=2e-5);np.testing.assert_allclose(en,e,atol=2e-8,rtol=2e-5)
        if slip>.01:assert r['friction_increment']>0
        # A rejected/uncommitted evaluation must never replace accepted capture.
        c.evaluate(x+np.array([.001,0,0]),x*0,H,H*0,.001)
        assert np.array_equal(c.accepted_audit['points'],x)


def test_opt_in_capture_does_not_change_solver_trajectory(tmp_path):
    from packing_fem.sintered import mesh
    from packing_fem.explicit_material import hex_to_tet
    from packing_fem.explicit_solver import solve
    X,h,_=mesh(np.ones((1,1,1),bool),1.);t=hex_to_tet(X,h)
    c=json.loads(Path('configs/explicit-area-cauchy-pressure.json').read_text());c.update(length_mm=1.,pressure_mpa=.02,ramp_s=.0001,hold_s=.00005,max_steps=5000,remesh_every=0)
    roots=[]
    for capture in (False,True):
        root=tmp_path/str(capture);solve(root,X,t,dict(c,save_contact_audit=capture));roots.append(root)
    a=np.loadtxt(roots[0]/'accepted-step-audit.csv.gz',delimiter=',',skiprows=1);b=np.loadtxt(roots[1]/'accepted-step-audit.csv.gz',delimiter=',',skiprows=1);np.testing.assert_array_equal(a,b)
    assert not (roots[0]/'contact-audit.json').exists()
    assert json.loads((roots[1]/'contact-audit.json').read_text())['accepted']['records']==[]
