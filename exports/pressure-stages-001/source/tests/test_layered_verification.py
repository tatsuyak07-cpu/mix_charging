"""Prevent contact-onset transients from being hidden by late-window reports."""
import numpy as np
from tools.summarize_layered_pressure import contact_window


def history(rows):
    names=['contacts','command_pressure_MPa','internal_Nmm','kinetic_Nmm']
    return np.array(rows,dtype=[(n,float) for n in names])


def test_early_contact_impact_remains_in_diagnostic():
    # Final equilibrium is quiet, but the first loaded contact has KE/I=0.2.
    h=history([(0,0,0,0),(8,.002,.001,.0002),(8,.02,.004,1e-8)])
    active,ratio=contact_window(h,.01)
    assert active.tolist()==[False,True,True]
    assert max(ratio[active])==.2
    assert ratio[-1]<.01


def test_free_flight_and_negligible_pressure_do_not_fake_contact_audit():
    h=history([(0,0,0,1.),(0,.005,.001,1.),(8,.00001,.001,1.),(8,.02,.004,1e-8)])
    active,ratio=contact_window(h,.01)
    assert active.tolist()==[False,False,False,True]
    assert np.isfinite(ratio).all()


def test_touching_periodic_boundary_is_not_a_spanning_cycle():
    from tools.verify_periodic_connectivity import winding
    X=np.array([[.9,.2,.2],[1.1,.2,.2],[.9,.4,.2],[.9,.2,.4]])
    report=winding(X,np.array([[0,1,2,3]]),1.)
    assert report['periodic_components']==1
    assert report['components'][0]['winding_rank']==0


def test_layered_topology_winding_is_translation_invariant():
    from tools.verify_periodic_connectivity import winding
    from tools.verify_plane_contact import fixture
    X,t,*_=fixture(1,.015,0.)
    for shift in (0.,np.array([.29,.31,.27])):
        report=winding(X+shift,t,1.)
        assert report['periodic_components']==1
        c=report['components'][0]
        assert c['winding_rank']==2 and c['winds_x'] and c['winds_y'] and not c['winds_z']
