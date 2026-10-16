"""Verify the diagnostic geometry and independent work checks, not convergence."""
import numpy as np
import pytest
from packing_fem.explicit_material import geometry
from packing_fem.explicit_periodic import Periodic
from tools.verify_plane_contact import fixture,measure


@pytest.mark.parametrize('n,phase',[(1,0.),(1,.25),(2,.01),(4,.5)])
def test_plane_fixture_keeps_area_volume_gap_and_periodic_identification(n,phase):
    X,t,x,H,upper=fixture(n,.015,phase)
    assert geometry(X,t)[0].sum()==pytest.approx(.8,abs=1e-13)
    assert geometry(x,t)[0].sum()==pytest.approx(.8,abs=1e-13)
    assert np.linalg.det(H)==pytest.approx(.815,abs=1e-13)
    assert x[upper,2].min()-x[~upper,2].max()==pytest.approx(.015,abs=1e-13)
    p=Periodic(X,1.)
    for cl in np.unique(p.classes):
        ids=np.flatnonzero(p.classes==cl);shift=np.rint(X[ids]-X[ids[0]])
        np.testing.assert_allclose(x[ids]-x[ids[0]],shift@H.T,atol=1e-13)


def test_plane_contact_diagnostic_resolves_component_balance_and_virtual_work():
    row=measure(1,.015,.01)
    assert row['force_N']==pytest.approx(row['point_triangle_force_N']+row['edge_edge_force_N'],abs=1e-13)
    assert row['force_balance_relative']<1e-12
    assert row['gap_gradient_relative']<1e-7
    assert row['macro_work_relative']<1e-7


def test_plane_contact_diagnostic_has_zero_response_beyond_activation_distance():
    row=measure(1,.03,.25)
    assert row['contacts']==0
    assert row['force_N']==0 and row['energy_Nmm']==0
    assert row['force_balance_relative']==0
