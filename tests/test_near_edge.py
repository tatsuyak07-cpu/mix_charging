from decimal import Decimal
from pathlib import Path
import numpy as np
import pytest
from packing_fem.area_contact import AreaContact
from packing_fem.explicit_periodic import Periodic,deformation
from tools.verify_near_edge import decimal_pair,decimal_cosine,exact_separating_projection


def saved():
    root=Path('exports/closest-selection-001/validation/pressure-run/calculation')
    with np.load(root/'last-state.npz') as z:
        X=z['reference_points'];x=z['points'];t=z['tets'];H=deformation(z['generalized_position'][-6:])*12.1
    c=AreaContact(X,t,Periodic(X,12.1),20.,.3,.1,thickness=.02,edge_stationarity_filter=True)
    return x,H,c.surface_stationarity


def test_saved_near_zero_pair_has_no_offedge_descent_witness():
    x,H,stationarity=saved();ids=np.array([175,384,385,401])
    r=stationarity.classify(ids,np.zeros(3,int),x,H)
    assert r['certified'] and not r['dominated'] and r['witness'] is None
    assert r['maximum_descent_cosine']<-.01
    q=decimal_pair(x[ids]);edge_scores=[];offedge_scores=[]
    for side,part,point,toward in ((0,ids[:2],q['q'],q['toward']),(1,ids[2:],q['p'],tuple(v.copy_negate() for v in q['toward']))):
        for face,lift in stationarity.star(part,np.zeros(3,int)):
            for node in face:
                score=decimal_cosine(toward,point,x[node]+H@lift)
                (edge_scores if node in part else offedge_scores).append(score)
    assert max(map(abs,edge_scores))<1e-60
    assert max(offedge_scores)<-.01


@pytest.mark.parametrize('angle',[0.,.37,1.19])
def test_corrected_pair_is_not_excluded_after_rigid_motion(angle):
    x,H,stationarity=saved();R=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0,0,1.]])
    for offset in ([0,0,0],[100,-50,23]):
        r=stationarity.classify(np.array([175,384,385,401]),np.zeros(3,int),x@R.T+offset,R@H)
        assert not r['dominated'] and r['witness'] is None


def test_decimal_crossed_edges_match_analytic_gap_and_branch_limits():
    points=np.array([[-1.,0,0],[1,0,0],[0,-1,1e-10],[0,1,1e-10]])
    q=decimal_pair(points)
    assert Decimal(q['s'])==Decimal('.5') and Decimal(q['t'])==Decimal('.5')
    assert abs(Decimal(q['distance'])-Decimal.from_float(1e-10))<Decimal('1e-80')
    points[2:,0]+=3
    with pytest.raises(ValueError,match='interior'):decimal_pair(points)


@pytest.mark.parametrize('crosses',[False,True])
def test_exact_projection_accepts_separation_and_rejects_crossing(crosses):
    first=np.array([[0.,0,0],[1.,0,0],[0.,1,0]])
    second=first+[0.,0,.1]
    if crosses:second=np.array([[.2,.2,-.1],[.2,.2,.1],[.3,.3,0.]])
    r=exact_separating_projection(first,second,[Decimal(0)]*3,[Decimal(0),Decimal(0),Decimal(1)])
    assert r['strict_separation'] is (not crosses)
