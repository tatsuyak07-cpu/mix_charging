"""Nonsmooth normal potential at equal closest points: no duplicate energy."""
import json
from pathlib import Path
import numpy as np
import pytest
from packing_fem.area_contact import AreaContact,SurfaceResolutionError
from packing_fem.explicit_periodic import Periodic
from packing_fem.contact_audit import energy_difference_force,potential

SOURCE=Path('results/packing-rve-development-001/stop-diagnostic-002')


def tie_fixture():
    # Clean concave L-wall fixture: two active source vertices have two
    # equidistant target feet, with no unresolved interior EE approach.
    from packing_fem import sintered
    from packing_fem.explicit_material import hex_to_tet,oriented
    mask=np.ones((2,2,1),bool);mask[1,1]=False
    X,h,_=sintered.mesh(mask,1.);X+=1.
    p=np.array([[2.015,2.015,1.45],[2.015,2.015,1.55],[2.115,2.015,1.5],[2.015,2.115,1.5]])
    t=np.r_[hex_to_tet(X,h),oriented(p,np.arange(4).reshape(1,4))+len(X)];X=np.r_[X,p]
    return X,t,X.copy(),X.copy(),np.eye(3)*12.1


def archived_trial():
    # This regression fixture is archived with the development outcome.
    source=SOURCE if SOURCE.exists() else Path('exports/packing-rve-contact-fixture-001')
    with np.load(source/'trial-geometry.npz') as z:return [z[k] for k in ('reference_points','tets','accepted_points','trial_points','cell')]


def contact(X,t,mu=0.,rule='clarke'):
    return AreaContact(X,t,Periodic(X,12.1),20.,mu,.1,thickness=.02,multiple_patches=True,edge_stationarity_filter=True,closest_tie_rule=rule)


def test_archived_trial_still_rejects_unresolved_edges_after_tie_resolution():
    X,t,old,x,H=archived_trial()
    with pytest.raises(SurfaceResolutionError,match='Edge-only'):contact(X,t).evaluate(x,x*0,H,H*0,0.)


def test_clean_concave_tie_is_resolved_only_with_explicit_tie_option():
    X,t,old,x,H=tie_fixture()
    with pytest.raises(SurfaceResolutionError,match='Multiple distinct closest'):contact(X,t,rule='stop').evaluate(x,x*0,H,H*0,0.)
    c=contact(X,t);c.capture_audit=True;f,_,h,r=c.evaluate(x,x*0,H,H*0,0.)
    assert r['closest_tie_branches']>=2 and r['contacts']>0
    tied=[rec for rec in c.trial_audit['records'] if 'normal_branches' in rec]
    assert len(tied)>=2
    for record in tied:
        assert len(record['normal_branches'])==2
        assert potential(record['z'],record)>0
        # Independent centered differences of max(branch energies). At a
        # two-way exact tie they recover the uniform Clarke subgradient.
        for step in (5e-8,2e-8):
            fd=energy_difference_force(record,step)
            np.testing.assert_allclose(fd,record['local_force'],atol=2e-7,rtol=2e-4)
    np.testing.assert_allclose(f.sum(0),0,atol=1e-12)
    moment=sum((np.cross(k['z'],k['local_force']).sum(0) for k in c.trial_audit['records']),np.zeros(3))
    np.testing.assert_allclose(moment,0,atol=1e-12)
    assert sum(potential(k['z'],k) for k in c.trial_audit['records'])==pytest.approx(r['normal_energy'],abs=1e-12)


def test_tie_friction_has_no_initial_spring_and_is_objective():
    X,t,old,x,H=tie_fixture();c=contact(X,t,.3);c.capture_audit=True
    f,_,history,r=c.evaluate(x,x*0,H,H*0,0.)
    for rec in c.trial_audit['records']:
        if 'normal_branches' in rec:assert np.linalg.norm(rec['spring'])<1e-14
    c.history=history;theta=.23;R=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1.]])
    rotated=c.evaluate(x@R.T+[.1,-.2,.3],x*0,R@H,H*0,0.)
    np.testing.assert_allclose(rotated[0],f@R.T,atol=2e-10)
    assert rotated[-1]['normal_energy']==pytest.approx(r['normal_energy'],rel=1e-9)
    assert rotated[-1]['max_coulomb_ratio']<=.3+1e-10


def test_symmetric_contact_uses_one_max_potential_with_no_normal_energy_gain():
    X,t,old,x,H=tie_fixture();c=contact(X,t);c.capture_audit=True;c.evaluate(x,x*0,H,H*0,0.)
    for rec in c.trial_audit['records']:
        if 'normal_branches' not in rec:continue
        from packing_fem.barrier_contact import barrier
        values=[]
        for branch in rec['normal_branches']:
            z=rec['z'][branch['indices']];from packing_fem.contact_geometry import closest_triangle
            point,_=closest_triangle(z[0],z[1:]);values.append(barrier(np.linalg.norm(z[0]-point),rec['kn'],rec['thickness'])[0])
        assert potential(rec['z'],rec)==pytest.approx(max(values),abs=1e-14)
        assert potential(rec['z'],rec)==pytest.approx(sum(values)/len(values),abs=1e-14)


def test_tied_contact_preserves_nonzero_friction_history_and_full_potential_gradient():
    X,t,old,x,H=tie_fixture();c=contact(X,t,.3);c.capture_audit=True
    c.history=c.evaluate(x,x*0,H,H*0,0.)[2]
    y=x.copy();G=H.copy();slip=1e-5;y[:,:2]+=slip*X[:,2,None]/12.1;G[:2,2]+=slip
    f,_,hist,r=c.evaluate(y,y*0,G,G*0,.001)
    assert r['tangent_energy']>0 and r['max_coulomb_ratio']<=.3+1e-10
    assert r['closest_tie_branches']>=2
    for rec in c.trial_audit['records']:
        if 'normal_branches' not in rec:continue
        fd=energy_difference_force(rec,2e-8)
        np.testing.assert_allclose(fd,rec['local_force'],atol=2e-7,rtol=2e-4)
    assert sum(potential(rec['z'],rec) for rec in c.trial_audit['records'])==pytest.approx(r['normal_energy']+r['tangent_energy'],abs=1e-11)
    np.testing.assert_allclose(f.sum(0),0,atol=1e-12)
