"""Sufficient geometric certificate, including cases where it must not apply."""
import numpy as np
import pytest
from packing_fem.structured_extension import StructuredExtension
from packing_fem.explicit_material import geometry,hex_to_tet,oriented
from packing_fem.sintered import mesh
from packing_fem.explicit_periodic import deformation,Periodic
from packing_fem.area_contact import AreaContact
from tools.run_barrier_compaction import fixture as pore
from tools.verify_plane_contact import fixture


def test_affine_void_extension_preserves_material_and_exact_margin():
    X,t=pore(2);extension=StructuredExtension(X,t);A=np.array([[1.1,.02,0],[0,.9,.03],[.01,0,.7]]);shift=np.array([.2,-.1,.3])
    r,z=extension.evaluate(X@A.T+shift,4*A)
    assert r['certificate_passed'] and r['ghost_nodes']==9
    assert r['normalized_margin']==pytest.approx(1.,abs=2e-13)
    np.testing.assert_allclose(z['points'],z['reference_points']@A.T+shift,atol=3e-14)
    assert not r['candidate_exclusion_authorized'] and not r['physical_void_filled']


def saved(refinement=2):
    from pathlib import Path
    root=Path('exports/edge-stationarity-002/preflight')
    with np.load(root/f'mesh-{refinement}.npz') as z:X=z['reference_points'];t=z['tets'];x=z['points']
    with np.load(root/'shape-source/last-state.npz') as z:H=4*deformation(z['generalized_position'][-6:])
    return X,t,x,H


def test_saved_geometry_global_bound_including_periodic_images_and_objectivity():
    X,t,x,H=saved();extension=StructuredExtension(X,t);r,z=extension.evaluate(x,H)
    assert r['certificate_passed'] and r['normalized_margin']>.5
    rng=np.random.default_rng(219);indices=rng.integers(0,len(z['tets']),size=(2,100));w=rng.random((2,100,4));w/=w.sum(2)[:,:,None]
    points=[]
    for array in (z['reference_points'],z['points']):points.append(np.einsum('kni,knij->knj',w,array[z['tets'][indices]]))
    shifts=rng.integers(-1,2,size=(100,3));dX=points[0][1]-points[0][0]+4*shifts;dy=points[1][1]-points[1][0]+shifts@H.T;normalized=dy@np.linalg.inv(H/4).T
    assert np.all(np.einsum('ij,ij->i',normalized,dX)>=r['normalized_margin']*np.sum(dX*dX,axis=1)-1e-11)
    theta=.42;R=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    other,_=extension.evaluate(x@R.T+[.1,-.2,.4],R@H)
    assert other['certificate_passed']
    assert other['normalized_margin']==pytest.approx(r['normalized_margin'],abs=2e-12)


def test_periodic_boundary_mismatch_is_not_certified_even_if_margin_positive():
    X,t=pore(1);x=X.copy();x[np.isclose(X[:,0],4.),0]+=.001
    r,_=StructuredExtension(X,t).evaluate(x,np.eye(3)*4)
    assert r['normalized_margin']>0 and r['minimum_extension_J']>0
    assert not r['checks']['periodic_boundary_matches'] and not r['certificate_passed']


def test_material_positive_j_does_not_hide_interpenetrating_slabs():
    X,t,x,H,_=fixture(2,.015,.0);x[X[:,2]>.5,2]-=.02;H[2,2]-=.02
    assert geometry(x,t)[0].min()>0
    r,_=StructuredExtension(X,t).evaluate(x,H)
    assert r['minimum_material_J']>0 and r['minimum_void_extension_J']<0
    assert not r['certificate_passed'] and not r['candidate_exclusion_authorized']


def test_large_periodic_shear_is_conservatively_uncertified_despite_positive_j():
    X,h,_=mesh(np.ones((2,2,2),bool),.5);t=hex_to_tet(X,h);x=X.copy();x[np.isclose(X[:,2],.5),0]+=2.
    r,_=StructuredExtension(X,t).evaluate(x,np.eye(3))
    assert r['checks']['periodic_boundary_matches'] and r['minimum_extension_J']==pytest.approx(1.)
    assert r['normalized_margin']<0 and not r['certificate_passed']


def test_certified_wavy_gap_still_carries_valid_contact_force():
    X,t,x,H,upper=fixture(4,.015,.0);interface=np.isclose(X[:,2],.6);x[interface,2]+=.002*np.cos(2*np.pi*X[interface,0]+.23)
    r,_=StructuredExtension(X,t).evaluate(x,H)
    assert r['certificate_passed'] and not r['candidate_exclusion_authorized']
    c=AreaContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02,edge_stationarity_filter=True)
    force=c.evaluate(x,x*0,H,H*0,0.)[0]
    assert force[upper,2].sum()>.06


def test_noncanonical_physical_triangulation_is_explicitly_unsupported():
    X,h,_=mesh(np.ones((1,1,1),bool),1.);t=hex_to_tet(X,h);Y=X.copy();Y[:,0]=1-Y[:,0];t=oriented(Y,t)
    with pytest.raises(ValueError,match='canonical background'):StructuredExtension(Y,t)


def test_reference_local_chord_failure_and_global_certificate_are_distinct():
    import json
    from pathlib import Path
    from packing_fem.material_neighborhood import MaterialNeighborhood
    from packing_fem.barrier_contact import feature,ContactResolutionError
    X,t,x,H=saved(4);r,_=StructuredExtension(X,t).evaluate(x,H)
    assert r['certificate_passed']
    source=Path('exports/edge-stationarity-002/preflight/summary.json');hit=json.loads(source.read_text())['levels'][3]['mapped']['failure_feature'];ids=np.array(hit['ids']);shift=np.array(hit['shift']);points=x[ids].copy();points[1:]+=H@shift;_,w=feature(points,'pt');m=MaterialNeighborhood(X,t,4.,.1)
    assert m.witness('local',ids,'pt',shift) is not None
    with pytest.raises(ContactResolutionError,match='not certified'):
        m.local('local',ids,'pt',shift,points,w,x,H)
    assert not r['candidate_exclusion_authorized']


def test_same_body_across_void_force_retained_even_with_positive_certificate():
    X,t,x,H=saved(1);p=Periodic(X,4.);c=AreaContact(X,t,p,20.,.3,.1,thickness=.02);c.capture_audit=True
    first=c.evaluate(x,x*0,H,H*0,0.);assert first[-1]['contacts']==2
    for record in c.trial_audit['records']:
        ids=np.array(record['support'][:4]);shift=np.array(record['images'][1],int)
        assert len(set(c.material_neighborhood.components[ids]))==1
        assert c.material_neighborhood.witness(('pore',*ids),ids,'pt',shift) is None
        assert record['normal_force']>0
    r,_=StructuredExtension(X,t).evaluate(x,H);assert r['certificate_passed']
    second=c.evaluate(x,x*0,H,H*0,0.)
    np.testing.assert_array_equal(first[0],second[0]);np.testing.assert_array_equal(first[1],second[1])
    assert not r['candidate_exclusion_authorized']
