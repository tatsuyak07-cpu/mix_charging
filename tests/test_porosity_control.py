import numpy as np
from packing_fem.porosity_control import limited_increment,volume_limited_increment


def test_restart_endpoint_for_finite_elastic_slab():
    # Fixed cell with a free-sided slab: exact finite Hencky elastic solution.
    exponent = .33/(1-.33)
    compression = 1-(.5/(1-.13))**(1/exponent)
    phi = .13
    slope = None
    for _ in range(20):
        step = limited_increment(.08, phi, compression, .1, 1e-6, slope)
        new_compression = compression+step
        new_phi = 1-.5*(1-new_compression)**(-exponent)
        assert new_phi >= .1
        slope = (phi-new_phi)/step
        compression, phi = new_compression, new_phi
        if phi <= .101:
            break
    assert .1 <= phi <= .101
    # The former unrestricted first restart step overshoots the target.
    initial = 1-(.5/(1-.13))**(1/exponent)
    assert 1-.5*(1-initial-.08)**(-exponent) < .1


def test_high_porosity_and_existing_smaller_step_preserved():
    assert limited_increment(.03, .25, .2, .1, 1e-6) == .03
    assert limited_increment(1e-5, .13, .3, .1, 1e-6) == 1e-5


def test_volume_prediction_reaches_target_for_isochoric_solid_and_fixed_rigid_volume():
    solid=.6;rigid=.005;cell=.68
    phi=1-solid/(cell-rigid)
    increment=volume_limited_increment(.05,phi,1-cell,.1,1e-6,rigid,safety=1.)
    solved_phi=1-solid/(cell-increment-rigid)
    assert abs(solved_phi-.1)<1e-14
