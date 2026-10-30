import numpy as np
from packing_fem.surface_patch import integrate,grouped_field
from tools.verify_onset_budget import plane_reference
from packing_fem.embedded_contact import EmbeddedContact
from packing_fem.explicit_periodic import Periodic
from tools.verify_plane_contact import fixture


def test_absolute_reference_budget_against_independent_decimal_integral():
    depth=1e-4;width=.01;z=.02*(1-depth)
    source=np.array([[0.,0.,z],[1.,0.,z+.02*depth/width],[0.,1.,z]])
    targets=np.array([[[-2.,-2.,0.],[4.,-2.,0.],[-2.,4.,0.]]]);mapping=np.array([[0,1,2]])
    exact,_=plane_reference(source)
    for budget in (1e-9,5e-10,2.5e-10):
        rule=integrate(source,targets,.5,kn=10.,tolerance=.00025,error_budget='normalized',absolute_energy_budget=budget/4,absolute_force_budget=budget/4)
        value=rule['weights']@grouped_field(rule['bary'],source,targets,[[0]],10.,.02,mapping,True)[0]
        assert abs(value[0]-exact[0])<=budget/2
        assert abs(value[1:]-exact[1:]).sum()<=budget/2
        assert np.all(rule['weights']>0) and abs(rule['weights'].sum()-.5)<1e-12


def test_weak_contact_still_forbids_inactive_history_reset():
    import pytest
    from packing_fem.barrier_contact import barrier
    from packing_fem.embedded_contact import reinitialize_inactive
    X,t,x,H,upper=fixture(1,.01999999,.25)
    contact=EmbeddedContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02,error_budget='normalized',absolute_energy_budget=1e-6,absolute_force_budget=1e-6)
    force,extra,history,report=contact.evaluate(x,x*0,H,H*0,0.)
    contact.history=history;contact.accept_geometry(x,H)
    audit=contact.last_integration_audit
    assert audit['absolute_force_excess_L1_N']<=audit['force_budget_N']/2
    assert audit['absolute_energy_excess_Nmm']<=audit['energy_budget_Nmm']/2
    assert report['minimum_separation_mm']<.02 and history
    assert abs(force[upper,2].sum()-barrier(.01999999,20.,.02)[1])<1e-12
    new=EmbeddedContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02)
    with pytest.raises(ValueError,match='active-history reset prohibited'):
        reinitialize_inactive(contact,new,x,x*0,H,True)


def test_nonzero_absolute_error_is_measured_without_changing_area_spring_law():
    from packing_fem.barrier_contact import barrier
    X,t,x,H,upper=fixture(1,.015,.25)
    # Deliberately loose UNIT-TEST budget exercises the ledger on a
    # nonmatching triangulation; this is not a production accuracy choice.
    contact=EmbeddedContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02,error_budget='normalized',absolute_energy_budget=100.,absolute_force_budget=100.)
    force,_,_,report=contact.evaluate(x,x*0,H,H*0,0.)
    audit=contact.last_integration_audit
    assert 0<audit['absolute_force_excess_L1_N']<=50.
    expected_energy,expected_force,_=barrier(.015,20.,.02)
    assert abs(force[upper,2].sum()-expected_force)<1e-12
    assert abs(report['normal_energy']-expected_energy)<1e-12
