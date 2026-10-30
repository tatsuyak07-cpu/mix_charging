import numpy as np
import pytest
from packing_fem.embedded_contact import EmbeddedContact
from packing_fem.explicit_periodic import Periodic
from packing_fem.area_contact import SurfaceResolutionError
from tools.verify_plane_contact import fixture


def test_lookahead_changes_only_cubature_controls_never_accepted_geometry():
    X,t,x,H,upper=fixture(1,.015,.25);velocity=np.zeros_like(x);velocity[upper,2]=-.01;Hdot=np.zeros((3,3));Hdot[2,2]=-.01
    contact=EmbeddedContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02,error_budget='normalized',rule_strategy='pose_atlas',lookahead_steps=[4,32])
    original=x.copy()
    with pytest.raises(SurfaceResolutionError) as captured:contact.evaluate(x,velocity,H,Hdot,.0001)
    rules=captured.value.feature['rules'];assert rules
    for rule in rules.values():
        assert len(rule['atlas'])==1 and rule['atlas_controls']
        actual=np.asarray(rule['atlas'][0]['source'])
        # Reference to the actual evaluation remains unmodified. Forecasts
        # are separately tagged and retained solely as integration witnesses.
        for control in rule['atlas_controls']:
            assert control['lookahead_steps'] in (4,32)
            assert np.linalg.norm(np.asarray(control['source'])-actual)>0 or np.allclose(actual[:,2],.4)
    np.testing.assert_array_equal(x,original)
    assert not contact.history
