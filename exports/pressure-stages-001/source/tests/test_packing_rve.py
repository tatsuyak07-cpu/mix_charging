import numpy as np
import pytest
from packing_fem.packing_rve import build
from packing_fem.explicit_material import geometry,advance
from packing_fem.explicit_timestep import assembled_frequency_bound,triangle_frequency_bound
from packing_fem.explicit_solver import lumped_mass
from packing_fem.explicit_periodic import Periodic
from packing_fem.area_contact import AreaContact


def material():return dict(young_mpa=10.,poisson=.33,yield_mpa=.05,hardening_mpa=.2,density_tonne_mm3=1e-9)


def test_bcc_geometry_porosity_and_periodic_face_connectivity():
    X,t,mask,m=build()
    assert abs(m['initial_porosity']-.4)<.005
    assert m['periodic_components']==1 and m['winding_rank']==3
    assert m['periodic_bonds']==8 and m['cells_across_neck']>2
    assert m['diameter_mm']==10 and m['neck_diameter_mm']==2
    assert geometry(X,t)[0].sum()==pytest.approx((1-m['initial_porosity'])*12.1**3,rel=1e-14)
    for axis in range(3):
        other=[k for k in range(3) if k!=axis]
        lo=X[np.isclose(X[:,axis],0)][:,other];hi=X[np.isclose(X[:,axis],12.1)][:,other]
        np.testing.assert_array_equal(lo,hi)


def test_unloaded_rve_has_no_artificial_contact_forces():
    X,t,_,m=build();L=m['cell_length_mm'];p=Periodic(X,L)
    c=AreaContact(X,t,p,20.,.3,.1,thickness=.02,multiple_patches=True,edge_stationarity_filter=True)
    f,extra,h,r=c.evaluate(X,X*0,np.eye(3)*L,np.zeros((3,3)),0.)
    assert r['contacts']==0 and not h
    np.testing.assert_array_equal(f,0);np.testing.assert_array_equal(extra,0)


@pytest.mark.parametrize('nu',[-.2,0.,.33,.48])
def test_fast_row_majorant_never_underestimates_original_bound(nu):
    X,t,_,m=build(6);c=material();c['poisson']=nu;V=geometry(X,t)[0];mass=lumped_mass(t,V,c['density_tonne_mm3'],len(X));rng=np.random.default_rng(72)
    A=np.array([[.93,.03,0],[0,1.02,.02],[.01,0,.91]]);x=X@A.T
    inc=np.repeat(A[None],len(t),axis=0);b,ep=advance(np.repeat(np.eye(3)[None],len(t),axis=0),np.zeros(len(t)),inc,c)
    stress=rng.normal(size=(len(t),3,3));stress=(stress+stress.swapaxes(1,2))/2
    for sigma,elastic in ((None,None),(stress,b)):
        exact=assembled_frequency_bound(x,t,mass,c,sigma,elastic);bound=triangle_frequency_bound(x,t,mass,c,sigma,elastic)
        assert bound>=exact*(1-1e-13)
        assert np.isfinite(bound)


def test_sparse_linear_reference_matches_isotropic_cube_and_work():
    from packing_fem.sintered import mesh
    from packing_fem.explicit_material import hex_to_tet
    from packing_fem.linear_rve import solve
    X,h,_=mesh(np.ones((2,2,2),bool),.5);t=hex_to_tet(X,h);c=material();c['length_mm']=1.
    u,r=solve(X,t,c,.001)
    import json
    json.dumps(r)
    assert all(r['checks'].values()),r
    np.testing.assert_allclose(r['macro_strain'],[.33*.001/10,.33*.001/10,-.001/10,0,0,0],atol=1e-15)
    np.testing.assert_allclose(r['volume_average_stress_mpa'],np.diag([0,0,-.001]),atol=1e-15)


def test_active_only_timestep_uses_zero_barrier_hessian_outside_support():
    import json
    from pathlib import Path
    from packing_fem.explicit_solver import stable_dt
    X,t,_,m=build(6);L=m['cell_length_mm'];p=Periodic(X,L);c=json.loads(Path('configs/packing-rve-development.json').read_text());V=geometry(X,t)[0];mass=lumped_mass(t,V,c['density_tonne_mm3'],len(X));contact=AreaContact(X,t,p,20.,.3,.1,thickness=.02)
    contact.evaluate(X,X*0,np.eye(3)*L,np.zeros((3,3)),0.);contact.accept_geometry(X,np.eye(3)*L)
    b=np.repeat(np.eye(3)[None],len(t),axis=0);sigma=np.zeros_like(b)
    legacy=stable_dt(X,t,V,mass,dict(c,active_contact_timestep=False),X*0,contact,sigma,b)
    active=stable_dt(X,t,V,mass,c,X*0,contact,sigma,b)
    assert active>=legacy
    assert active==pytest.approx(c['cfl']*2/np.sqrt(triangle_frequency_bound(X,t,mass,c,sigma,b)),rel=1e-14)


def test_continuous_bcc_union_volume_and_fixed_cell_mesh_refinement():
    from packing_fem.packing_rve import continuum_porosity
    from packing_fem.sintered import draw
    from itertools import product
    phi=continuum_porosity(12.1)
    assert phi==pytest.approx(.40066184999880794,abs=1e-13)
    n=64;L=12.1;mask=np.zeros((n,n,n),bool)
    for p in (np.zeros(3),np.ones(3)*L/2):draw(mask,p,p,5.,L/n)
    for end in product((-.5,.5),repeat=3):draw(mask,np.zeros(3),L*np.array(end),1.,L/n)
    assert abs(1-mask.mean()-phi)<.002
    # Actual voxel porosity differs; do not silently rescale diameter/cell per mesh.
    assert build(16)[3]['cell_length_mm']==build(24)[3]['cell_length_mm']==12.1


def test_bcc_mask_has_cubic_symmetry_not_assumed_mechanical_isotropy():
    X,t,mask,m=build()
    np.testing.assert_array_equal(mask,mask.transpose(1,0,2))
    np.testing.assert_array_equal(mask,mask.transpose(2,1,0))
    np.testing.assert_array_equal(mask,np.roll(mask,[8,8,8],axis=(0,1,2)))
    assert not m['statistical_isotropy_verified']


def test_surface_bisection_resolves_true_edge_approach_without_edge_spring():
    from packing_fem.explicit_material import oriented
    from packing_fem.surface_remesh import bisect,transfer_contact
    from packing_fem.area_contact import SurfaceResolutionError
    X=np.array([[-1.,0,0],[1,0,0],[-1,-.1,-.1],[-1,.1,-.1],[0,-1,.1],[0,1,.1],[-.1,-1,.2],[.1,-1,.2]])+[2,2,2]
    t=oriented(X,np.arange(8).reshape(2,4));accepted=X.copy();accepted[4:,2]-=.079
    trial=X.copy();trial[4:,2]-=.085;H=np.eye(3)*5;c=material();b=np.repeat(np.eye(3)[None],len(t),axis=0);ep=np.zeros(len(t));V=geometry(X,t)[0]
    old=AreaContact(X,t,Periodic(X,5.),20.,.3,.2,thickness=.02,edge_stationarity_filter=True)
    old.history=old.evaluate(accepted,accepted*0,H,H*0,0.)[2]
    assert not old.history
    with pytest.raises(SurfaceResolutionError,match='Edge-only') as hit:old.evaluate(trial,trial*0,H,H*0,0.)
    p=bisect(X,accepted,t,b,ep,V,X*0,hit.value.feature['ids'][:2],5.)
    new=AreaContact(p['X'],p['t'],Periodic(p['X'],5.),20.,.3,.2,thickness=.02,edge_stationarity_filter=True)
    transfer_contact(old,new,p,H)
    supports=p['supports'];y=(trial[supports[:,0]]+trial[supports[:,1]])/2
    f,extra,h,r=new.evaluate(y,y*0,H,H*0,.001)
    assert r['contacts']>0 and r['normal_energy']>0
    assert r['edge_contacts']==0
    np.testing.assert_allclose(f.sum(0),0,atol=1e-13)


def test_exact_position_transfer_reconstructs_periodic_field_and_rejects_incompatibility():
    X,t,_,m=build(6);p=Periodic(X,12.1);rng=np.random.default_rng(123);q=rng.normal(size=p.ndof)*.001;x=p.positions(q)
    np.testing.assert_allclose(p.remap_positions(x,q[-6:]),q,atol=3e-15)
    # Break an aliased boundary node; the same gauge is no longer representable.
    counts=np.bincount(p.classes);cl=np.flatnonzero(counts>1)[0];node=np.flatnonzero(p.classes==cl)[0];bad=x.copy();bad[node,0]+=.001
    with pytest.raises(ValueError,match='Periodic displacement'):p.remap_positions(bad,q[-6:])


def test_saved_large_rve_periodic_transfer_stays_below_original_tolerance():
    from pathlib import Path
    with np.load(Path('exports/packing-rve-transfer-fixture-001/reconstructed-position.npz')) as z:
        X=z['reference_points'];x=z['points'];macro=z['generalized_position'][-6:]
    periodic=Periodic(X,12.1);q=periodic.remap_positions(x,macro)
    assert np.max(abs(periodic.positions(q)-x))<1e-12
    np.testing.assert_array_equal(q[-6:],macro)
