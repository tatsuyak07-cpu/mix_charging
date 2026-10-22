"""Regression evidence for refinement transfer and unresolved local wall contacts."""
import numpy as np
import pytest
from tools.run_barrier_compaction import fixture
from tools.verify_contact_mesh_preflight import prolongate,probe
import json
from pathlib import Path


def test_tetra_prolongation_preserves_affine_shape_and_rejects_void():
    X,t=fixture(1);Y,_=fixture(2)
    F=np.array([[1.1,.01,0],[0,.97,.02],[.01,0,.83]]);shift=np.array([.02,-.01,.04])
    result,error=prolongate(X,t,X@F.T+shift,Y)
    np.testing.assert_allclose(result,Y@F.T+shift,atol=2e-15,rtol=0)
    assert error<2e-15
    with pytest.raises(ValueError,match='outside source solid'):
        prolongate(X,t,X,np.array([[2.,2.,2.]]))


def test_initial_refinement_four_rejects_coplanar_corner_candidates():
    X,t=fixture(4);c=json.loads(Path('configs/explicit-area-cauchy-pressure.json').read_text())
    c['local_material_filter']=False
    result=probe(X,t,X,np.eye(3)*4,c)
    assert not result['resolved']
    assert result['error_type']=='SurfaceResolutionError'
    d=result['diagnostic'];assert d['distinct_equal_closest']
    np.testing.assert_allclose(d['source_reference'],[1.,1.,2.],atol=1e-14)
    assert len(d['candidates'])>=2
    for candidate in d['candidates']:
        assert candidate['reference_source_distance_to_target_plane_mm']<1e-14
        assert 0<candidate['distance_mm']<c['barrier_thickness_mm']
