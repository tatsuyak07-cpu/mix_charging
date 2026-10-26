from pathlib import Path
import numpy as np
import pytest
from packing_fem.area_contact import AreaContact,SurfaceResolutionError
from packing_fem.explicit_periodic import Periodic,deformation
from tools.verify_plane_contact import fixture


@pytest.mark.parametrize('angle,offset',[(0.,[0,0,0]),(.37,[.3,-.2,.1]),(1.19,[100,-50,23])])
def test_saved_unresolved_pair_is_not_masked_by_other_endpoint_contact(angle,offset):
    root=Path('exports/closest-selection-001/validation/pressure-run/calculation')
    with np.load(root/'last-state.npz') as z:X=z['reference_points'];x=z['points'];t=z['tets'];H=12.1*deformation(z['generalized_position'][-6:])
    R=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0,0,1.]])
    c=AreaContact(X,t,Periodic(X,12.1),20.,.3,.1,thickness=.02,multiple_patches=True,edge_stationarity_filter=True,closest_tie_rule='clarke')
    with pytest.raises(SurfaceResolutionError) as caught:c.evaluate(x@R.T+offset,x*0,R@H,H*0,0.)
    assert caught.value.feature['kind']=='ee'
    assert caught.value.feature['distance_mm']<1e-8


@pytest.mark.parametrize('n,phase',[(1,.01),(2,.25),(4,.5)])
def test_planar_coverage_preserves_exact_uniform_traction(n,phase):
    from packing_fem.barrier_contact import barrier
    X,t,x,H,upper=fixture(n,.015,phase)
    c=AreaContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02)
    force,extra,history,report=c.evaluate(x,x*0,H,H*0,0.)
    energy,traction,_=barrier(.015,20.,.02)
    assert report['normal_energy']==pytest.approx(energy,rel=1e-11)
    assert force[upper,2].sum()==pytest.approx(traction,rel=1e-11)
    assert sum(report['edge_coverage_counts'].values())>0


@pytest.mark.parametrize('resolution,expected',[(0.,None),(.00025,None),(.001,'resolved_endpoint_neighborhood')])
def test_near_endpoint_resolution_is_bounded_by_gap_and_position(resolution,expected):
    from packing_fem.explicit_material import oriented
    from packing_fem.barrier_contact import feature,relative,segment_distances
    gap=.02;s=1e-5;slope=.1;xc=slope*gap+s*(1+slope*slope)
    X=np.array([[0,0,gap],[1,0,gap+slope],[0,.1,gap+.1],[0,-.1,gap+.1],
                [xc,-1,0],[xc,1,0],[xc-.2,0,-.1],[xc+.1,0,-.2]])+2
    t=oriented(X,np.arange(8).reshape(2,4));H=np.eye(3)*5
    c=AreaContact(X,t,Periodic(X,5.),20.,.3,.1,thickness=.03,edge_quadrature_relative_tolerance=resolution)
    face=next(i for i,f in enumerate(c.faces) if set(f)=={4,5,6});ptids=np.r_[0,c.faces[face]];points=X[ptids]
    branch,w=feature(points,'pt');r=relative(points,'pt',branch)
    candidate=(np.linalg.norm(r),('pt',0,face,0,0,0),ptids,(0,0,0),points,branch,w,r)
    ids=np.array([0,1,4,5]);ee=X[ids];distance=segment_distances(ee[None])[0]
    result=c.edge_coverage.resolved(ids,np.zeros(3,int),ee,distance,{0:[candidate]},X,H,[(0,0,0)])
    assert result==expected


def test_disconnected_coplanar_endpoint_patch_does_not_cover_crossed_edge():
    from packing_fem.explicit_material import oriented
    from packing_fem.barrier_contact import feature,relative,segment_distances
    X=np.array([[-1,0,.015],[1,0,.015],[-1,-.1,.115],[-1,.1,.115],
                [0,-1,0],[0,1,0],[.1,0,0],[0,0,-.1],
                [-1.1,-.1,0],[-.9,-.1,0],[-1,.1,0],[-1,0,-.1]])+2
    t=oriented(X,np.arange(12).reshape(3,4));H=np.eye(3)*5
    c=AreaContact(X,t,Periodic(X,5.),20.,.3,.1,thickness=.02,edge_quadrature_relative_tolerance=.001)
    face=next(i for i,f in enumerate(c.faces) if set(f)=={8,9,10});ptids=np.r_[0,c.faces[face]];points=X[ptids]
    branch,w=feature(points,'pt');r=relative(points,'pt',branch)
    candidate=(np.linalg.norm(r),('pt',0,face,0,0,0),ptids,(0,0,0),points,branch,w,r)
    ids=np.array([0,1,4,5]);ee=X[ids];distance=segment_distances(ee[None])[0]
    assert c.edge_coverage.resolved(ids,np.zeros(3,int),ee,distance,{0:[candidate]},X,H,[(0,0,0)]) is None
