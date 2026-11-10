import numpy as np
import pytest
from packing_fem.quasistatic_coupled import smooth_peak_merit


@pytest.mark.parametrize('power',[2.,4.,8.,16.])
def test_smooth_peak_merit_gradient_and_norm_bounds(power):
    force=np.array([-.009,.0090001,.004,-.0001,0.]);value,gradient=smooth_peak_merit(force,power)
    norm=np.sqrt(2*value);assert abs(force).max()<=norm<=len(force)**(1/power)*abs(force).max()
    for i in range(len(force)):
        delta=np.eye(len(force))[i]*1e-8
        fd=(smooth_peak_merit(force+delta,power)[0]-smooth_peak_merit(force-delta,power)[0])/2e-8
        assert fd==pytest.approx(gradient[i],rel=1e-7,abs=1e-12)
    for scale in [1e-100,1e100]:
        scaled,derivative=smooth_peak_merit(force*scale,power)
        assert scaled/scale**2==pytest.approx(value)
        np.testing.assert_allclose(derivative/scale,gradient,atol=1e-15)
    zero,dzero=smooth_peak_merit(np.zeros(4),power);assert zero==0;np.testing.assert_array_equal(dzero,0)
