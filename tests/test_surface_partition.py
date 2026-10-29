"""Persistent material integration cells must survive changing contact poses."""
import numpy as np
import pytest
from packing_fem.surface_patch import integrate,validate_partition,grouped_field
from tools.verify_onset_budget import plane_reference,errors


def test_cover_rejects_balanced_overlap_and_hole():
    # Two copies of the left half have total area one, but no right half.
    half=np.array([[1.,0.,0.],[.5,.5,0.],[0.,0.,1.]])
    with pytest.raises(ValueError,match='Overlapping'):validate_partition([half,half])
    with pytest.raises(ValueError):validate_partition([half])
    np.testing.assert_array_equal(validate_partition([np.eye(3)]),[np.eye(3)])


def test_changing_activation_retains_a_valid_cover_and_independent_accuracy():
    targets=np.array([[[-2.,-2.,0.],[4.,-2.,0.],[-2.,4.,0.]]]);mapping=np.array([[0,1,2]])
    poses=[];partition=None
    for depth,width in [(.001,.05),(.0011,.055),(.001,.05)]:
        z=.02*(1-depth);source=np.array([[0.,0.,z],[1.,0.,z+.02*depth/width],[0.,1.,z]])
        result=integrate(source,targets,.5,kn=10.,tolerance=.00025,error_budget='normalized',
                         initial_partition=partition,compress_rule=False)
        validate_partition(result['partition'])
        assert result['weights'].sum()==pytest.approx(.5,abs=1e-12)
        # No old cell may be crossed by a new cell: centroid/vertices must
        # lie together in at least one previous cell (up to roundoff).
        if partition is not None:
            for child in result['partition']:
                assert any(np.min(child@np.linalg.inv(parent))>=-1e-9 for parent in partition)
        partition=result['partition'];poses.append(source)
    # Recheck every preceding pose with the final fixed rule against the
    # independent 60-digit reduced plane integral, not the adaptive estimator.
    for source in poses:
        values=grouped_field(result['bary'],source,targets,[[0]],10.,.02,mapping,True)[0]
        reference,_=plane_reference(source)
        error=errors(result['weights']@values,reference)
        assert max(error['energy_error'],error['force_error'])<.001
