import numpy as np
from scipy.integrate import quad
from packing_fem.random_packing import generate,graph_audit,neck_added_volume


def test_random_geometry_is_reproducible_nonoverlapping_and_spans_all_periods():
    c=dict(particle_count=16,packing_seed=20261109,diameter_mm=10.,neck_diameter_mm=2.,neck_selection_gap_mm=.4,maximum_neck_gap_mm=.5,requested_initial_porosity=.4)
    a=generate(c);b=generate(c)
    np.testing.assert_array_equal(a['fractional_centers'],b['fractional_centers'])
    assert a['metadata']['maximum_overlap_mm']==0
    assert a['metadata']['minimum_center_distance_mm']>=10
    assert a['metadata']['periodic_components']==1
    assert a['metadata']['winding_rank']==3
    assert abs(a['metadata']['analytic_porosity']-.4)<1e-12
    assert a['metadata']['maximum_neck_gap_mm']<=.5


def test_neck_volume_matches_independent_radial_integration():
    R=5.;r=1.;distance=10.3
    integral=quad(lambda rho:2*np.pi*rho*(distance-2*np.sqrt(R*R-rho*rho)),0,r,epsabs=1e-12)[0]
    np.testing.assert_allclose(neck_added_volume(distance,R,r),integral,rtol=1e-12)


def test_connected_graph_must_also_wind_in_three_directions():
    bonds=np.zeros((3,2),int);shifts=np.eye(3,dtype=int)
    assert graph_audit(1,bonds,shifts)[:2]==(1,3)
    assert graph_audit(1,bonds[:2],shifts[:2])[:2]==(1,2)
