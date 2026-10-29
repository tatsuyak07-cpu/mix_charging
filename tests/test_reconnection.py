import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from packing_fem.reconnection import project,propose_reconnection,oriented_surface
from packing_fem.explicit_material import geometry,oriented
from packing_fem.adaptive_tet import boundary


def config():return json.loads(Path('configs/packing-porosity-cycle.json').read_text())


def test_three_to_two_repairs_quality_without_losing_uniform_plastic_state():
    x=np.array([[1,0,0],[-.5,np.sqrt(3)/2,0],[-.5,-np.sqrt(3)/2,0],[0,0,2.],[0,0,-2.]])
    t=oriented(x,np.array([[3,4,0,1],[3,4,1,2],[3,4,2,0]]));V=geometry(x,t)[0]
    b=np.repeat(np.eye(3)[None],3,0);ep=np.full(3,.2)
    p=propose_reconnection(x,t,b,ep,V,config(),1.,1e-10);assert p is not None
    t2,b2,ep2,V2,r=p
    assert r['operation']=='3_to_2' and r['quality_after']>r['quality_before']
    assert oriented_surface(t)==oriented_surface(t2)
    assert V2.sum()==pytest.approx(V.sum(),rel=1e-14)
    np.testing.assert_allclose(b2,np.repeat(np.eye(3)[None],2,0),atol=1e-14)
    np.testing.assert_allclose(ep2,.2,atol=1e-14)


def test_four_to_four_repairs_the_same_closed_cavity():
    x=np.array([[0,0,2.],[0,0,-2.],[1,0,0],[0,1.,0],[-1,0,0],[0,-1.,0]])
    t=oriented(x,np.array([[0,1,2,3],[0,1,3,4],[0,1,4,5],[0,1,5,2]]));V=geometry(x,t)[0]
    p=project(x,t,np.repeat(np.eye(3)[None],4,0),np.full(4,.2),V,list(range(4)),
              [[2,4,0,3],[2,4,3,1],[2,4,1,5],[2,4,5,0]],config(),1e-10)
    assert p is not None and p[-1]['operation']=='4_to_4'
    assert oriented_surface(t)==oriented_surface(p[0])
    assert p[-1]['quality_after']>p[-1]['quality_before']


def test_active_contact_slip_and_applied_force_survive_internal_reconnection():
    from tools.verify_plane_contact import fixture
    from tools.verify_explicit_remesh import seed_bipyramid
    from packing_fem.embedded_contact import EmbeddedContact
    from packing_fem.explicit_periodic import Periodic
    from packing_fem.area_contact import SurfaceResolutionError
    X,t,_,_,upper=fixture(1,.015,0.)
    X,t=seed_bipyramid(X,t,0,.03);upper=X[:,2]>.5;rules={}
    for replay in range(12):
        c=EmbeddedContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02,quadrature_rules=rules,error_budget='normalized')
        try:
            for gap in np.linspace(.025,.015,9):
                x=X.copy();x[upper,2]-=.2-gap;H=np.diag([1.,1.,.8+gap])
                r=c.evaluate(x,x*0,H,H*0,0.);c.history=r[2];c.accept_geometry(x,H)
            for slip in np.linspace(0,.0001,9)[1:]:
                y=x.copy();y[upper,0]+=slip;G=H.copy();G[0,2]+=slip
                r=c.evaluate(y,y*0,G,G*0,.001/8);c.history=r[2];c.accept_geometry(y,G)
            break
        except SurfaceResolutionError as e:
            assert e.feature and e.feature.get('rules');rules.update(e.feature['rules'])
    else:raise AssertionError('Prescribed active-remesh setup did not converge')
    assert r[-1]['contacts']>0 and r[-1]['tangent_energy']>0
    old=deepcopy(c.history);V=geometry(y,t)[0]
    p=propose_reconnection(y,t,np.repeat(np.eye(3)[None],len(t),0),np.full(len(t),.2),V,config(),.8,1e-10)
    assert p is not None and oriented_surface(t)==oriented_surface(p[0])
    c.update_topology(p[0]);after=c.replay_force_state(y,y*0,G)
    np.testing.assert_allclose(after[0],r[0],atol=1e-12)
    np.testing.assert_allclose(after[1],r[1],atol=1e-12)
    assert after[-1]['normal_energy']==pytest.approx(r[-1]['normal_energy'],rel=1e-12)
    assert after[-1]['tangent_energy']==pytest.approx(r[-1]['tangent_energy'],rel=1e-12)
    assert c.history.keys()==old.keys()
    for key in old:
        for name in ('spring','plastic','director','bary','anchor_ids'):
            np.testing.assert_array_equal(c.history[key][name],old[key][name])
