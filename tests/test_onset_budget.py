import numpy as np
import pytest
from tools.experimental_surface_budget import integrate
from tools.verify_onset_budget import plane_reference,applied,errors


@pytest.mark.parametrize('depth,width',[(.01,1.),(.0001,.01),(.000001,.1)])
def test_normalized_onset_against_decimal_reduced_integral(depth,width):
    z0=.02*(1-depth);source=np.array([[0.,0.,z0],[1.,0.,z0+.02*depth/width],[0.,1.,z0]])
    targets=np.array([[[-2.,-2.,0.],[4.,-2.,0.],[-2.,4.,0.]]]);mapping=np.array([[0,1,2]])
    reference,details=plane_reference(source)
    result=integrate(source,targets,.5,kn=10.,tolerance=.00025,force_error_norm='resultant',normalized_floor=True)
    value=applied(result,source,targets,[[0]],mapping);error=errors(value,reference)
    assert max(error['energy_error'],error['force_error'])<.001
    np.testing.assert_allclose(reference[1:].reshape(-1,3).sum(0),0,atol=1e-14*np.linalg.norm(reference[1:]))
    assert result['weights'].sum()==pytest.approx(.5,abs=1e-12)
    assert details['quad_energy_error_estimate_Nmm']<abs(reference[0])*1e-10
