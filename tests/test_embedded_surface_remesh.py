from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
from packing_fem.embedded_contact import EmbeddedContact
from packing_fem.embedded_remesh import refine_rules,transfer,coordinates
from packing_fem.explicit_material import geometry
from packing_fem.explicit_periodic import Periodic
from packing_fem.surface_remesh import bisect,audit
from packing_fem.triangle_quadrature import RULES
from tools.verify_plane_contact import fixture


from tools.verify_embedded_remesh import case


@pytest.mark.parametrize('fraction',[.2,.5,.8])
@pytest.mark.parametrize('slip',[.0001,.01])
def test_exact_active_embedded_bisection_retains_stick_slip_and_virtual_work(fraction,slip):
    old,new,p,H,before,mapping,parent,state=case(fraction,slip)
    saved=deepcopy(old.history)
    after,report=transfer(old,new,p,H,mapping,parent,before,1e-8)
    c=json.loads(Path('configs/explicit-prototype.json').read_text());c.update(length_mm=1.,friction=.3)
    check=audit(state,p,c,before[-1],after[-1],1e-8,old_response=before,new_response=after)
    assert check['accepted'],check
    assert report['new_material_samples']==0 and report['periodic_virtual_work_relative_error']<1e-8
    assert len(saved)==len(new.history)>0 and after[-1]['tangent_energy']>0
    for key in saved:
        for name,value in saved[key].items():np.testing.assert_array_equal(old.history[key][name],value)
    # Continue through reversal; refinement must not merely pass an instantaneous audit.
    y=p['x'].copy();G=H.copy();upper=p['X'][:,2]>.5;y[upper,0]-=.001;G[0,2]-=.001
    new.accept_geometry(p['x'],H)
    r=new.evaluate(y,y*0,G,G*0,.0001)
    assert r[-1]['max_coulomb_ratio']<=.3+1e-12 and r[-1]['friction_increment']>=-1e-12


def test_unaligned_active_split_is_rejected_without_reweighting_or_history_mutation():
    old,new,p,H,before,mapping,parent,state=case()
    saved=deepcopy(old.history)
    # A different geometric cut does not have the pre-existing area partition.
    original=p['supports'][len(old.X)]
    q=bisect(*state[:6],state[-1],original,1.,fraction=.3)
    other=EmbeddedContact(q['X'],q['t'],Periodic(q['X'],1.),20.,.3,.1,thickness=.02,error_budget='normalized')
    with pytest.raises(ValueError,match='weights do not partition'):
        refine_rules(old,other.faces,q)
    assert old.history.keys()==saved.keys()
    for key in saved:np.testing.assert_array_equal(old.history[key]['plastic'],saved[key]['plastic'])


@pytest.mark.parametrize('key_type',[str,int])
def test_full_surface_rules_leave_exactly_zero_physical_integration_area(key_type):
    X,t,_,_,_=fixture(2,.015,0.)
    base=EmbeddedContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02)
    B,w=RULES[8];rules={}
    for i,face in enumerate(base.faces):
        tri=X[face];area=np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))/2
        rules[key_type(i)]=dict(bary=B.tolist(),weights=(area*w).tolist())
    contact=EmbeddedContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02,quadrature_rules=rules)
    np.testing.assert_array_equal(contact.inner.integration_area0[:len(X)],0.)
