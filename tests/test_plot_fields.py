import numpy as np
import pytest
from packing_fem.plot_fields import element_peeq


@pytest.mark.parametrize('values,expected',[(np.full(3,.2),[.2,.2]),(np.array([.1,.5,.7]),[3.1/7,.7])])
def test_element_field_preserves_current_volume_integral(values,expected):
    J=np.array([1.,2.,3.]);reference=np.array([1.,3.,2.])
    state=dict(t=np.zeros((2,4),dtype=int),parent=np.array([0,0,1]),ep=values,V0=reference,b=J[:,None,None]**(2/3)*np.tile(np.eye(3),(3,1,1)))
    field=element_peeq(state)
    np.testing.assert_allclose(field,expected)
    np.testing.assert_allclose(field@np.array([7.,6.]),(reference*J)@values)
