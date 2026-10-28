import numpy as np
import pytest
from packing_fem.explicit_material import advance,response
from tools.verify_plastic_cycle_reference import plastic_reference


def test_reference_matches_coaxial_plastic_return_and_permanent_volume():
    c=dict(young_mpa=10.,poisson=.33,yield_mpa=.05,hardening_mpa=.2)
    pressure=np.array([0.,.02,.04,.06,.08,.1,.08,.06,.02,0.])
    r=plastic_reference(pressure,10.,.33,.05,.2)
    b=np.eye(3)[None];ep=np.zeros(1);old=np.eye(3)
    for i,p in enumerate(pressure):
        F=np.diag([r['lateral_stretch'][i],r['lateral_stretch'][i],r['axial_stretch'][i]])
        b,ep=advance(b,ep,(F@np.linalg.inv(old))[None],c);old=F
        stress,energy,J=response(b,ep,c)
        np.testing.assert_allclose(stress[0],np.diag([0.,0.,-p]),atol=1e-13)
        np.testing.assert_allclose(ep[0],r['peeq'][i],atol=1e-13)
        assert energy[0]==pytest.approx(r['specific_total_energy_MPa'][i],abs=1e-14)
        assert np.linalg.det(F)==pytest.approx(r['volume_ratio'][i],abs=1e-14)
    assert r['axial_compression'][-1]>.2
    assert np.linalg.det(old)==pytest.approx(1.,abs=1e-14)
    np.testing.assert_allclose(b,np.eye(3)[None],atol=1e-13)


def test_reference_stays_elastic_below_yield_and_rejects_perfect_plastic_ambiguity():
    r=plastic_reference(np.array([0.,.02,0.]),10.,.33,.05,.2)
    np.testing.assert_array_equal(r['peeq'],0.)
    assert r['axial_compression'][-1]==0.
    with pytest.raises(ValueError):plastic_reference(np.array([0.,.1]),10.,.33,.05,0.)


def test_face_traction_integration_has_hand_calculated_forces_and_moment_balance():
    from tools.plastic_cycle_audit import face_internal_force
    X=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    sigma=np.array([[2.,.5,.2],[.5,3.,.1],[.2,.1,4.]])
    force=face_internal_force(X,np.arange(4)[None],sigma[None])
    grads=np.array([[-1.,-1.,-1.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    np.testing.assert_allclose(force,-grads@sigma.T/6,atol=1e-15)
    np.testing.assert_allclose(force.sum(0),0.,atol=1e-15)
    np.testing.assert_allclose(np.cross(X,force).sum(0),0.,atol=1e-15)
