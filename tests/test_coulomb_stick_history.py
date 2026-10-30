"""Sticking must not trigger a sliding-anchor update from subtraction roundoff."""
from copy import deepcopy
import numpy as np
import pytest
from packing_fem.area_contact import AreaContact
from packing_fem.explicit_periodic import Periodic
from tools.verify_plane_contact import fixture


@pytest.mark.parametrize('embedded',[False,True])
def test_reverse_sticking_preserves_material_anchor_after_sliding(embedded):
    X,t,x,H,upper=fixture(2,.015,0.)
    c=AreaContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02,quadrature_mode='material')
    if embedded:
        from packing_fem.embedded_contact import EmbeddedContact
        from packing_fem.triangle_quadrature import RULES
        B,w=RULES[8];rules={}
        for i,ids in enumerate(c.faces):
            tri=X[ids];A=np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))/2
            rules[str(i)]=dict(bary=B.tolist(),weights=(A*w).tolist())
        c=EmbeddedContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02,quadrature_rules=rules,error_budget='normalized')
    r=c.evaluate(x,x*0,H,H*0,0.);c.history=r[2];c.accept_geometry(x,H)
    for d in np.linspace(.001,.01,10):
        _,_,x,H,_=fixture(2,.015,0.);x[upper,0]+=d;H[0,2]+=d
        r=c.evaluate(x,x*0,H,H*0,.0001);c.history=r[2];c.accept_geometry(x,H)
    saved=deepcopy(c.history)
    _,_,x,H,_=fixture(2,.015,0.);x[upper,0]+=.0099;H[0,2]+=.0099
    r=c.evaluate(x,x*0,H,H*0,.0001)
    assert r[-1]['max_coulomb_ratio']<.3
    assert r[2].keys()==saved.keys()
    for key,state in r[2].items():
        np.testing.assert_array_equal(state['anchor_ids'],saved[key]['anchor_ids'])
        np.testing.assert_array_equal(state['bary'],saved[key]['bary'])
        np.testing.assert_array_equal(state['anchor_shift'],saved[key]['anchor_shift'])
