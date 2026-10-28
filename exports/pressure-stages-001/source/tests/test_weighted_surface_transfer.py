import json
from pathlib import Path

import numpy as np
import pytest

from packing_fem.area_contact import AreaContact, frame
from packing_fem.explicit_material import geometry, oriented
from packing_fem.explicit_periodic import Periodic
from packing_fem.surface_remesh import bisect, transfer_contact, audit
from tools.verify_plane_contact import fixture


def state():
    X,t,x,H,upper=fixture(2,.015,0.)
    c=json.loads(Path('configs/explicit-prototype.json').read_text());c.update(length_mm=1.,friction=.3)
    V=geometry(X,t)[0];b=np.repeat(np.eye(3)[None],len(t),axis=0);ep=np.linspace(0,.2,len(t))
    edges=np.unique(np.sort(np.concatenate([t[:,[i,j]] for i in range(4) for j in range(i+1,4)]),axis=1),axis=0)
    edge=next(e for e in edges if np.all(np.isclose(X[e,2],.6)) and np.linalg.norm(X[e[1]]-X[e[0]])==.5)
    return X,t,x,H,upper,c,V,b,ep,edge


@pytest.mark.parametrize('fraction',[.2,.5885708790180821,.8])
def test_weighted_material_affine_fields_and_periodic_velocity(fraction):
    X,t,x,H,upper,c,V,b,ep,edge=state();velocity=X@np.diag([1.,2.,3.])
    p=bisect(X,x,t,b,ep,V,velocity,edge,1.,fraction=fraction)
    np.testing.assert_array_equal(p['b'],b[p['parents']]);np.testing.assert_array_equal(p['ep'],ep[p['parents']])
    np.testing.assert_allclose(np.bincount(p['parents'],weights=p['V0'],minlength=len(t)),V,atol=1e-16)
    np.testing.assert_allclose(p['velocity'],p['X']@np.diag([1.,2.,3.]),atol=1e-15)
    np.testing.assert_allclose(p['X'][len(X)],(1-fraction)*X[edge[0]]+fraction*X[edge[1]],atol=1e-15)
    reverse=bisect(X,x,t,b,ep,V,velocity,edge[::-1],1.,fraction=1-fraction)
    np.testing.assert_allclose(reverse['X'],p['X'],atol=1e-15)
    assert abs(geometry(p['x'],p['t'])[0].sum()-geometry(x,t)[0].sum())<1e-14


@pytest.mark.parametrize('fraction',[.2,.5885708790180821,.8])
def test_active_weighted_sticking_transfer_then_sliding(fraction):
    X,t,x,H,upper,c,V,b,ep,edge=state()
    old=AreaContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02)
    old.history=old.evaluate(x,x*0,H,H*0,0.)[2]
    x=x.copy();H=H.copy();x[upper,0]+=.0001;H[0,2]+=.0001
    before=old.evaluate(x,x*0,H,H*0,.001);old.history=before[2]
    p=bisect(X,x,t,b,ep,V,x*0,edge,1.,fraction=fraction)
    new=AreaContact(p['X'],p['t'],Periodic(p['X'],1.),20.,.3,.1,thickness=.02)
    after=transfer_contact(old,new,p,H)
    r=audit((X,x,t,b,ep,V,x*0),p,c,before[-1],after[-1],1e-7,old_response=before,new_response=after)
    assert r['accepted'],r
    assert any(k[1]>=len(X) for k in new.history)
    assert after[-1]['tangent_energy']>0
    y=p['x'].copy();G=H.copy();mask=p['X'][:,2]>.5;y[mask,0]+=.01;G[0,2]+=.01
    f,_,_,report=new.evaluate(y,y*0,G,G*0,.001)
    assert report['friction_increment']>0
    assert report['max_coulomb_ratio']==pytest.approx(.3,abs=1e-12)
    assert abs(f[mask,0].sum())==pytest.approx(.3*f[mask,2].sum(),rel=1e-10)
    lost=dict(after[-1],tangent_energy=0.)
    assert not audit((X,x,t,b,ep,V,x*0),p,c,before[-1],lost,1e-7)['accepted']


def test_weighted_periodic_copy_with_reversed_endpoint_numbering():
    X=np.array([[1.,1,1],[2.,1,1],[1.,2,1],[1.,1,2]])
    X=np.r_[X,(X+[5,0,0])[[1,0,2,3]]];t=oriented(X,np.arange(8).reshape(2,4))
    b=np.repeat(np.eye(3)[None],2,axis=0);V=geometry(X,t)[0]
    p=bisect(X,X,t,b,np.zeros(2),V,X*0,[0,1],5.,fraction=.2)
    np.testing.assert_allclose(p['X'][-1]-p['X'][-2],[5,0,0],atol=1e-14)
    np.testing.assert_allclose(p['weights'][-1],p['weights'][-2][::-1])
    for f in [0.,1.,float('nan')]:
        with pytest.raises(ValueError,match='fraction'):bisect(X,X,t,b,np.zeros(2),V,X*0,[0,1],5.,fraction=f)


def test_nonuniform_elastic_slip_uses_geometric_weights_not_half_average():
    X,t,x,H,upper,c,V,b,ep,edge=state()
    old=AreaContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02)
    old.history=old.evaluate(x,x*0,H,H*0,0.)[2]
    x=x.copy();H=H.copy();x[upper,0]+=.0001;H[0,2]+=.0001
    old.history=old.evaluate(x,x*0,H,H*0,.001)[2]
    for key,s in old.history.items():
        factor=.25 if key[1]==edge[0] else .75
        s['spring']=s['spring']*factor;s['plastic']=s['coordinate']-s['spring']
    p=bisect(X,x,t,b,ep,V,x*0,edge,1.,fraction=.2)
    new=AreaContact(p['X'],p['t'],Periodic(p['X'],1.),20.,.3,.1,thickness=.02)
    transfer_contact(old,new,p,H)
    def world(s,node,z):
        a=np.zeros((9,3));a[0]=z[node];a[4]=z[s['peer']];a[8]=z[s['second']]
        return frame(a,s['director'])@s['spring']
    node=len(X);s=next(s for k,s in new.history.items() if k[1]==node)
    endpoints=[next(s for k,s in old.history.items() if k[1]==i) for i in p['supports'][node]]
    vectors=[world(s,int(i),x) for s,i in zip(endpoints,p['supports'][node])]
    expected=sum(w*v for w,v in zip(p['weights'][node],vectors))
    np.testing.assert_allclose(world(s,node,p['x']),expected,atol=1e-14)
    assert np.linalg.norm(expected-(vectors[0]+vectors[1])/2)>1e-8


def test_default_midpoint_matches_archived_implementation_bit_for_bit():
    import importlib.util
    archived=Path('exports/packing-rve-replay-001/validation/corrected-run/source/surface_remesh.py')
    spec=importlib.util.spec_from_file_location('packing_fem._midpoint_reference',archived)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    X,t,x,H,upper,c,V,b,ep,edge=state();velocity=np.sin(X)
    old=module.bisect(X,x,t,b,ep,V,velocity,edge,1.)
    new=bisect(X,x,t,b,ep,V,velocity,edge,1.)
    for key in ('X','x','t','b','ep','V0','velocity','supports','parents'):
        np.testing.assert_array_equal(new[key],old[key])
