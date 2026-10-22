"""Solid-bridge exclusion: analytical coverage, periodic lift and fold rejection."""
import numpy as np
import pytest
from packing_fem import sintered
from packing_fem.explicit_material import hex_to_tet,geometry
from packing_fem.material_neighborhood import segment_covered,MaterialNeighborhood
from packing_fem.barrier_contact import feature,ContactResolutionError
from packing_fem.adaptive_tet import boundary
from tools.verify_plane_contact import fixture
from packing_fem.area_contact import AreaContact
from packing_fem.explicit_periodic import Periodic


def box(mask,scale=1.):
    X,h,_=sintered.mesh(mask,scale)
    return X,hex_to_tet(X,h)


def test_segment_interval_union_matches_box_and_resolves_small_void():
    X,t=box(np.ones((1,1,1),bool))
    assert segment_covered(np.array([.1,.2,.3]),np.array([.9,.8,.7]),X[t])
    assert segment_covered(np.array([0.,0.,0.]),np.array([1.,1.,0.]),X[t])
    lower=X.copy();lower[:,2]*=.5-1e-7
    upper=X.copy();upper[:,2]=.5+1e-7+upper[:,2]*(.5-1e-7)
    vertices=np.r_[lower[t],upper[t]]
    assert not segment_covered(np.array([.3,.3,.1]),np.array([.3,.3,.9]),vertices)


def test_short_periodic_material_segment_and_objective_current_witness():
    X,t=box(np.ones((100,1,1),bool),.01);X[:,1:]*=100
    source=np.flatnonzero(np.linalg.norm(X-[.99,0,0],axis=1)<1e-12)[0]
    target=[np.flatnonzero(np.linalg.norm(X-point,axis=1)<1e-12)[0] for point in ([.01,0,0],[.01,1,0],[.01,0,1])]
    ids=np.r_[source,target];shift=np.array([1,0,0]);points=X[ids].copy();points[1:]+=shift
    _,w=feature(points,'pt');c=MaterialNeighborhood(X,t,1.,.1)
    assert c.local('periodic',ids,'pt',shift,points,w,X,np.eye(3))
    theta=.4;R=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1.]])
    F=R@np.diag([1.1,.8,1.2]);x=X@F.T+[.3,-.2,.4];p=x[ids].copy();p[1:]+=F@shift;_,w=feature(p,'pt')
    assert c.local('periodic',ids,'pt',shift,p,w,x,F)


def test_local_fold_with_positive_j_is_rejected_not_silently_excluded():
    X,t=box(np.ones((2,1,1),bool),.02)
    node=np.argmin(np.linalg.norm(X,axis=1));faces=boundary(t)
    face=next(f for f in faces if np.all(abs(X[f,0]-.04)<1e-12) and np.min(np.linalg.norm(X[f]-[.04,0,0],axis=1))<1e-12)
    ids=np.r_[node,face];p=X[ids];_,w=feature(p,'pt');c=MaterialNeighborhood(X,t,1.,.1)
    assert c.local('fold',ids,'pt',np.zeros(3),p,w,X,np.eye(3))
    x=X.copy();x[abs(X[:,0]-.02)<1e-12,1]+=.02
    assert np.all(geometry(x,t)[0]>0)
    p=x[ids];_,w=feature(p,'pt')
    with pytest.raises(ContactResolutionError,match='Current local material bridge not certified'):
        c.local('fold',ids,'pt',np.zeros(3),p,w,x,np.eye(3))


def test_true_periodic_contact_force_identical_with_filter_enabled_or_disabled():
    X,t,x,H,_=fixture(2,.015,.25);outputs=[]
    for enabled in (False,True):
        c=AreaContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02,local_material_filter=enabled)
        for delta in (0.,.0001):
            y=x.copy();y[X[:,2]>.5,0]+=delta;G=H.copy();G[0,2]+=delta
            f,e,h,r=c.evaluate(y,y*0,G,G*0,.001);c.history=h
            outputs.append((enabled,delta,f,e,r))
    for i in range(2):
        a,b=outputs[i],outputs[i+2]
        np.testing.assert_array_equal(a[2],b[2]);np.testing.assert_array_equal(a[3],b[3])
        assert a[4]['normal_energy']==b[4]['normal_energy']>0
        assert b[4]['local_material_pairs']==0


def test_disconnected_touching_reference_bodies_are_not_local_material():
    A,t=box(np.ones((1,1,1),bool));B=A+np.array([1.,0,0]);X=np.r_[A,B];t=np.r_[t,t+len(A)]
    source=np.flatnonzero(np.linalg.norm(A-[1.,0,0],axis=1)<1e-12)[0]
    target=[len(A)+np.flatnonzero(np.linalg.norm(B-point,axis=1)<1e-12)[0] for point in ([1.,0,0],[1.,1,0],[1.,0,1])]
    c=MaterialNeighborhood(X,t,3.,.1)
    assert c.witness('disconnected',np.r_[source,target],'pt',np.zeros(3)) is None


def test_archived_same_body_pore_contact_force_unchanged():
    import json
    from pathlib import Path
    from packing_fem.explicit_periodic import deformation
    root=Path('exports/contact-cycle-cfl-001/fine/run')
    with np.load(root/'last-state.npz') as z:X=z['reference_points'];t=z['tets'];x=z['points'];H=4*deformation(z['generalized_position'][-6:])
    data=json.loads((root/'contact-audit.json').read_text());outputs=[]
    for enabled in (False,True):
        c=AreaContact(X,t,Periodic(X,4.),20.,.3,.1,thickness=.02,local_material_filter=enabled)
        c.history={tuple(row['key']):{k:np.asarray(v) if isinstance(v,list) else v for k,v in row['state'].items()} for row in data['committed_history']}
        outputs.append(c.evaluate(x,x*0,H,H*0,0.))
    for k in (0,1):np.testing.assert_array_equal(outputs[0][k],outputs[1][k])
    assert outputs[0][-1]['contacts']==outputs[1][-1]['contacts']==2
    assert outputs[1][-1]['max_coulomb_ratio']<=.3


def test_historical_corner_point_candidates_are_local_across_radius_sensitivity():
    import json
    from pathlib import Path
    from tools.run_barrier_compaction import fixture as pore_fixture
    old=json.loads(Path('exports/contact-mesh-preflight-002/summary.json').read_text())
    candidates=old['levels'][3]['initial']['diagnostic']['candidates'];X,t=pore_fixture(4)
    for radius in (.05,.1,.2):
        c=MaterialNeighborhood(X,t,4.,radius)
        for candidate in candidates:
            ids=np.array(candidate['ids']);points=X[ids];_,w=feature(points,'pt')
            assert c.local(('corner',*ids),ids,'pt',np.zeros(3),points,w,X,np.eye(3)*4)


def test_refined_edge_approach_still_stops_after_point_filter():
    from tools.run_barrier_compaction import fixture as pore_fixture
    from packing_fem.area_contact import SurfaceResolutionError
    X,t=pore_fixture(4);c=AreaContact(X,t,Periodic(X,4.),20.,.3,.1,thickness=.02,max_candidates=1000000,local_material_filter=True)
    with pytest.raises(SurfaceResolutionError,match='Edge-only') as hit:
        c.evaluate(X,X*0,np.eye(3)*4,np.zeros((3,3)),0.)
    assert hit.value.feature['kind']=='ee'
    assert hit.value.feature['distance_mm']==pytest.approx(.019968076595771916)
