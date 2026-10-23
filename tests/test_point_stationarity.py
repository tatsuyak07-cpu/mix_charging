"""PT source stationarity cannot classify quadrature points out of contact."""
import json
from pathlib import Path
import numpy as np
import pytest
from packing_fem.area_contact import AreaContact,SurfaceResolutionError
from packing_fem.explicit_periodic import Periodic,deformation
from packing_fem.explicit_material import geometry
from packing_fem.surface_stationarity import SurfaceStationarity
from tools.verify_plane_contact import fixture


def wavy(amplitude=.002,phase=.23):
    X,t,x,H,upper=fixture(4,.015,0.)
    interface=abs(X[:,2]-.6)<1e-12
    x[interface,2]+=amplitude*np.cos(2*np.pi*X[interface,0]+phase)
    p=Periodic(X,1.);c=AreaContact(X,t,p,20.,.3,.1,thickness=.02,edge_stationarity_filter=True);c.capture_audit=True
    return X,t,x,H,upper,interface,p,c


@pytest.mark.parametrize('amplitude,phase',[(.001,.23),(.002,.23),(.002,.37)])
def test_valid_wavy_quadrature_source_descent_must_not_remove_contact(amplitude,phase):
    X,t,x,H,upper,interface,p,c=wavy(amplitude,phase)
    assert np.all(geometry(x,t)[0]>0)
    micro=x-X@H.T
    for cl in np.unique(p.classes):
        group=micro[p.classes==cl];np.testing.assert_allclose(group,group[0]+np.zeros_like(group),atol=2e-15,rtol=0)
    f,e,h,r=c.evaluate(x,x*0,H,H*0,0.);s=SurfaceStationarity(X,c.faces,p.classes,1.);found=[]
    for record in c.trial_audit['records']:
        ids=np.array(record['support'][:4]);node=ids[0]
        if not interface[node]:continue
        value=s.classify_point(ids,np.array(record['images'][1],int),x,H)
        assert not value['candidate_exclusion_authorized']
        if value['source']['has_descent'] and not value['target']['has_descent']:
            gap=x[node,2]-.4;kn=.5*20*c.node_area0[node];u=(gap/.02)**2
            expected=-kn*gap/4*(2*(1-u)*np.log(u)-(1-u)**2/u)
            assert record['normal_force']==pytest.approx(expected,rel=1e-11)
            assert record['normal_force']>0;found.append(node)
    assert len(found)>0 and f[upper,2].sum()>0


def test_symmetric_wavy_case_still_reports_unresolved_equal_closest_targets():
    _,_,x,H,_,_,_,c=wavy(phase=0.)
    with pytest.raises(SurfaceResolutionError,match='Multiple distinct closest'):
        c.evaluate(x,x*0,H,H*0,0.)


@pytest.mark.parametrize('refinement',[4,6])
def test_archived_stopped_point_has_source_only_descent_and_remains_diagnostic(refinement):
    root=Path('exports/edge-stationarity-002/preflight');rows=json.loads((root/'summary.json').read_text())['levels'];row=next(v for v in rows if v['refinement']==refinement)
    with np.load(root/f'mesh-{refinement}.npz') as z:X=z['reference_points'];t=z['tets'];x=z['points']
    with np.load(root/'shape-source/last-state.npz') as z:H=4*deformation(z['generalized_position'][-6:])
    p=Periodic(X,4.);c=AreaContact(X,t,p,20.,.3,.1,thickness=.02);s=SurfaceStationarity(X,c.faces,p.classes,4.)
    hit=row['mapped']['failure_feature'];ids=np.array(hit['ids']);shift=np.array(hit['shift']);value=s.classify_point(ids,shift,x,H)
    assert value['source']['complete_star'] and value['source']['has_descent']
    assert value['target']['complete_star'] and not value['target']['has_descent']
    assert not value['candidate_exclusion_authorized']
    w=value['source']['witness'];v=np.array(w['direction']);toward=np.array(w['toward'])
    assert np.linalg.norm(toward-1e-4*v)<np.linalg.norm(toward)


def test_point_diagnostic_is_objective_and_periodic_representation_invariant():
    X,t,x,H,_,interface,p,c=wavy();c.evaluate(x,x*0,H,H*0,0.)
    record=next(v for v in c.trial_audit['records'] if interface[v['support'][0]])
    ids=np.array(record['support'][:4]);shift=np.array(record['images'][1],int);s=SurfaceStationarity(X,c.faces,p.classes,1.);before=s.classify_point(ids,shift,x,H)
    theta=.37;R=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1.]])
    after=s.classify_point(ids,shift,x@R.T+[.3,-.2,.1],R@H)
    for side in ('source','target'):
        assert before[side]['has_descent']==after[side]['has_descent']
        assert before[side]['maximum_descent_cosine']==pytest.approx(after[side]['maximum_descent_cosine'],abs=1e-11)
    # Rebase the entire target body by a periodic cell; preserve the source body.
    target_upper=X[ids[1],2]>.5;target_body=(X[:,2]>.5)==target_upper
    Y=X.copy();Y[target_body,0]+=1.;y=x.copy();y[target_body]+=H[:,0]
    # The selected source/target belong to separate slabs in this control.
    assert not target_body[ids[0]]
    other=SurfaceStationarity(Y,c.faces,Periodic(Y,1.).classes,1.).classify_point(ids,shift-[1,0,0],y,H)
    for side in ('source','target'):
        assert before[side]['has_descent']==other[side]['has_descent']
        assert before[side]['maximum_descent_cosine']==pytest.approx(other[side]['maximum_descent_cosine'],abs=1e-11)


def test_point_target_interior_uses_only_its_actual_triangle_star():
    X,_,x,H,_,interface,p,c=wavy();c.evaluate(x,x*0,H,H*0,0.)
    node=int(np.flatnonzero(interface)[0]);face=c.faces[0]
    s=SurfaceStationarity(X,c.faces,p.classes,1.)
    star=s.star(face,np.zeros(3,int))
    assert star and all(len(set(face)&set(tri))==3 for tri,_ in star)
