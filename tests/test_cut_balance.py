import numpy as np
import pytest
from tools.cut_balance import balance,intersect,residual


B=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6]])


def test_all_admissible_fractions_and_incompatible_adjacent_faces():
    a=balance(B,[.5,.5],0,1,1.,1e-10);b=balance(B,[.4,.6],0,1,1.,1e-10)
    assert a['roots']==[.5] and b['roots']==[.4]
    assert a['intervals'][0]==pytest.approx([.5-1e-10,.5+1e-10],abs=1e-15)
    assert intersect(a['intervals'],b['intervals'])==[]
    assert residual(B,[.4,.6],0,1,.5,1.)==pytest.approx(.1)
    assert intersect(a['intervals'],balance(B,[.25,.25],0,1,.5,5e-11)['intervals'])


def test_random_fraction_scan_agrees_with_direct_mass_measurement():
    rng=np.random.default_rng(91);b=rng.dirichlet([1,1,1],40);w=rng.random(40);w/=w.sum();tol=.01
    report=balance(b,w,0,1,1.,tol)
    for f in np.linspace(.001,.999,1001):
        s=b[:,1]/(b[:,0]+b[:,1])
        measured=residual(b,w,0,1,f,1.)<=tol and np.any(s<=f) and np.any(s>f)
        contained=any(lo<=f<=hi for lo,hi in report['intervals'])
        assert measured==contained


def test_apex_and_duplicate_cut_coordinates_are_not_split_or_reweighted():
    b=np.r_[B,[[0.,0.,1.]]];w=[.45,.45,.1];report=balance(b,w,0,1,1.,1e-10)
    assert .45 in report['roots'] and .55 in report['roots']
    assert report['apex_mass_fraction']==pytest.approx(.1)
    assert balance(np.array([[0.,0.,1.]]),[1.],0,1,1.,1e-10)['intervals']==[]
    duplicate=balance(np.r_[B,B],[.25]*4,0,1,1.,1e-10)
    assert duplicate['roots']==[.5]


def test_periodic_incident_faces_match_full_geometric_enumeration():
    from tools.verify_packing_cut_balance import incident
    from tools.verify_plane_contact import fixture
    from packing_fem.periodic_contact import Contact
    from packing_fem.explicit_periodic import Periodic
    X,t,_,_,_=fixture(2,.015,0.);p=Periodic(X,1.);g=Contact(X,t,p,20.,.3,.1)
    edge=g.faces[0][:2];a,b=X[edge];expected=[]
    for face,ids in enumerate(g.faces):
        for ia in range(3):
            for ib in range(3):
                if ia==ib:continue
                shift=X[ids[ia]]-a
                if np.linalg.norm(shift-np.rint(shift))<1e-12 and np.linalg.norm(X[ids[ib]]-b-np.rint(shift))<1e-12:expected.append((face,ia,ib,np.rint(shift).astype(int).tolist()))
    assert incident(g,p,edge,1.)==expected
    reversed_edges=incident(g,p,edge[::-1],1.)
    assert sorted((f,j,i,tuple(s)) for f,i,j,s in reversed_edges)==sorted((f,i,j,tuple(s)) for f,i,j,s in expected)


def test_cut_boundary_relaxation_includes_fractional_but_not_actual_point_split():
    from tools.check_cut_boundaries import relaxed_intervals
    # Both samples lie on f=.5. Current deterministic ownership cannot assign
    # half their combined mass, but the deliberately relaxed bound can.
    b=np.array([[.4,.4,.2],[.2,.2,.6]])
    report=balance(b,[.3,.7],0,1,1.,1e-10)
    assert report['intervals']==[]
    assert relaxed_intervals(report)==[[.5,.5]]
    a=balance(B,[.5,.5],0,1,1.,1e-10);c=balance(B,[.4,.6],0,1,1.,1e-10)
    relaxed=intersect(relaxed_intervals(a,1e-8),relaxed_intervals(c,1e-8))
    assert any(lo<=.2<=hi for lo,hi in relaxed)
    assert any(lo<=.8<=hi for lo,hi in relaxed)
    assert intersect(a['intervals'],c['intervals'])==[]
