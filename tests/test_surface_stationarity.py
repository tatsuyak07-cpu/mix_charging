"""Necessary closest-surface conditions; diagnostic screening is opt-in."""
import json
from pathlib import Path
import numpy as np
import pytest
from packing_fem.area_contact import AreaContact,SurfaceResolutionError
from packing_fem.barrier_contact import feature,relative
from packing_fem.explicit_material import oriented
from packing_fem.explicit_periodic import Periodic
from packing_fem.surface_stationarity import SurfaceStationarity
from tools.run_barrier_compaction import fixture as pore_fixture
from tools.verify_plane_contact import fixture


def crossed_edges():
    X=np.array([[-1.,0,0],[1,0,0],[-1,-.1,-.1],[-1,.1,-.1],
                [0,-1,.1],[0,1,.1],[-.1,-1,.2],[.1,-1,.2]])+2
    t=oriented(X,np.array([[0,1,2,3],[4,5,6,7]]));x=X.copy();x[4:,2]-=.085
    return X,t,x


def test_corner_pair_has_a_feasible_distance_decreasing_surface_direction():
    X,t=pore_fixture(4);p=Periodic(X,4.);c=AreaContact(X,t,p,20.,.3,.1,thickness=.02)
    hit=json.loads(Path('exports/local-material-neighborhood-001/preflight/summary.json').read_text())['levels'][3]['initial']['failure_feature']
    ids=np.array(hit['ids']);shift=np.array(hit['shift']);s=SurfaceStationarity(X,c.faces,p.classes,4.);r=s.classify(ids,shift,X,np.eye(3)*4)
    assert r['certified'] and r['dominated'];witness=r['witness']
    assert witness['descent_cosine']>.99
    q=np.array(witness['point']);v=np.array(witness['direction']);toward=np.array(witness['toward'])
    # Finite displacement stays in the incident triangle and lowers distance.
    for step in (1e-3,1e-4):assert np.linalg.norm(toward-step*v)<np.linalg.norm(toward)
    triangle=X[witness['face']];v0,v1=triangle[1:]-triangle[0];uv=np.linalg.lstsq(np.column_stack((v0,v1)),q+1e-3*v-triangle[0],rcond=None)[0]
    assert min(uv[0],uv[1],1-uv.sum())>=-1e-12


def test_true_crossed_edge_pair_is_not_screened_and_guard_still_stops():
    X,t,x=crossed_edges();p=Periodic(X,5.);c=AreaContact(X,t,p,20.,.3,.2,thickness=.02,edge_stationarity_filter=True)
    r=c.surface_stationarity.classify(np.array([0,1,4,5]),np.zeros(3,int),x,np.eye(3)*5)
    assert r['certified'] and not r['dominated'];assert r['distance_mm']==pytest.approx(.015)
    with pytest.raises(SurfaceResolutionError,match='Edge-only'):c.evaluate(x,x*0,np.eye(3)*5,np.zeros((3,3)),0.)


def test_stationarity_is_objective_and_uses_periodic_surface_stars():
    X,t,x,H,upper=fixture(1,.015,.25);p=Periodic(X,1.);c=AreaContact(X,t,p,20.,.3,.1,thickness=.02)
    c.guard.prepare_geometry(x,H);candidate=None
    for _,ids,kind,shift,_,_ in c.guard.neighbors:
        if kind!='ee' or not np.any(shift) or np.any(upper[ids[:2]]) or not np.all(upper[ids[2:]]):continue
        points=x[ids].copy();points[2:]+=H@shift;branch,_=feature(points,'ee')
        if np.linalg.norm(relative(points,'ee',branch))<.02:candidate=(ids,np.array(shift));break
    assert candidate is not None
    ids,shift=candidate;s=SurfaceStationarity(X,c.faces,p.classes,1.);r=s.classify(ids,shift,x,H)
    assert r['certified']
    theta=.47;R=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    rot=s.classify(ids,shift,x@R.T+[.2,-.1,.3],R@H)
    assert rot['dominated']==r['dominated'];assert rot['maximum_descent_cosine']==pytest.approx(r['maximum_descent_cosine'],abs=1e-12)
    # Move the upper body's cell representative: the physical pair is unchanged.
    Y=X.copy();Y[upper]+=np.array([1.,0,0]);y=x.copy();y[upper]+=H[:,0]
    rebased=SurfaceStationarity(Y,c.faces,Periodic(Y,1.).classes,1.).classify(ids,shift-[1,0,0],y,H)
    assert rebased['certified'] and rebased['dominated']==r['dominated']
    assert rebased['maximum_descent_cosine']==pytest.approx(r['maximum_descent_cosine'],abs=1e-12)


@pytest.mark.parametrize('n,phase',[(1,0.),(2,.25),(4,.5)])
def test_periodic_plane_reactions_and_energy_unchanged_with_screening(n,phase):
    X,t,x,H,upper=fixture(n,.015,phase);outputs=[]
    for enabled in (False,True):
        c=AreaContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02,edge_stationarity_filter=enabled)
        c.history=c.evaluate(x,x*0,H,H*0,0.)[2]
        y=x.copy();y[upper,0]+=.0001;G=H.copy();G[0,2]+=.0001
        outputs.append(c.evaluate(y,y*0,G,G*0,.001))
    for k in (0,1):np.testing.assert_array_equal(outputs[0][k],outputs[1][k])
    for key in ('normal_energy','tangent_energy','friction_increment','max_coulomb_ratio'):assert outputs[0][-1][key]==outputs[1][-1][key]
    assert outputs[1][-1]['contacts']>0


def test_missing_star_or_zero_gap_never_certifies_exclusion():
    X,t,x=crossed_edges();s=SurfaceStationarity(X,np.empty((0,3),int),Periodic(X,5.).classes,5.)
    r=s.classify(np.array([0,1,4,5]),np.zeros(3,int),x,np.eye(3)*5)
    assert not r['certified'] and not r['dominated']
    x=x.copy();x[4:,2]-=.015
    r=s.classify(np.array([0,1,4,5]),np.zeros(3,int),x,np.eye(3)*5)
    assert not r['dominated'] and r['distance_mm']<1e-12


def test_nonlocal_crossed_edges_of_face_connected_solid_are_retained():
    from packing_fem.explicit_material import geometry
    X,base,x=crossed_edges()
    # Triangular tube joins faces away from the two critical edges.
    bridge=np.array([[0,2,3,7],[0,2,6,7],[0,4,6,7]])
    t=oriented(X,np.r_[base,bridge])
    assert np.all(geometry(x,t)[0]>0)
    # All five tetrahedra belong to one component via shared faces.
    reached={0}
    while True:
        next_set=reached|{j for j in range(len(t)) for i in reached if len(set(t[i])&set(t[j]))==3}
        if next_set==reached:break
        reached=next_set
    assert len(reached)==len(t)
    c=AreaContact(X,t,Periodic(X,5.),20.,.3,.2,thickness=.02,edge_stationarity_filter=True)
    r=c.surface_stationarity.classify(np.array([0,1,4,5]),np.zeros(3,int),x,np.eye(3)*5)
    assert r['certified'] and not r['dominated']
    assert r['distance_mm']==pytest.approx(.015)
