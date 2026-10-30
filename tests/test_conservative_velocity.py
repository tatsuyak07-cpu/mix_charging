import numpy as np
import pytest
from packing_fem.explicit_material import geometry,hex_to_tet
from packing_fem.sintered import mesh
from packing_fem.explicit_periodic import Periodic
from packing_fem.surface_remesh import bisect
from packing_fem.conservative_velocity import project,moments


@pytest.mark.parametrize('fraction',[.2,.5,.8])
@pytest.mark.parametrize('frame',['pinned','zero_momentum'])
def test_preserves_physical_momenta_energy_and_prescribed_velocity(fraction,frame):
    X,h,_=mesh(np.ones((2,2,2),bool),.5);t=hex_to_tet(X,h);V=geometry(X,t)[0]
    def masses(t,V,n):
        m=np.zeros(n);np.add.at(m,t.ravel(),np.repeat(V/4,4));return m
    m=masses(t,V,len(X));old=Periodic(X,1.,mass_frame=frame);old.mass(m)
    v=np.random.default_rng(827).normal(size=old.ndof);velocity=old.velocities(v)
    p=bisect(X,X,t,np.repeat(np.eye(3)[None],len(t),0),np.zeros(len(t)),V,velocity,t[0,:2],1.,fraction=fraction)
    new=Periodic(p['X'],1.,mass_frame=frame);m2=masses(p['t'],p['V0'],len(p['X']));new.mass(m2)
    guess=new.remap_velocity(new.force(m2[:,None]*p['velocity']),v[old.prescribed]);K=float(.5*v@(old.M@v));P=moments(X,m,velocity)
    result,report=project(new,p['x'],m2,guess,P,K)
    np.testing.assert_allclose(moments(p['x'],m2,new.velocities(result)),P,rtol=1e-11,atol=1e-12)
    assert float(.5*result@(new.M@result))==pytest.approx(K,rel=1e-12)
    assert result[new.prescribed]==v[old.prescribed]
    assert report['momentum_scaled_error']<1e-11
    with pytest.raises(ValueError,match='require more kinetic energy'):
        project(new,p['x'],m2,guess,P,0.)


def test_zero_energy_transfer_and_rejection_of_unresolved_energy_mode():
    X,h,_=mesh(np.ones((2,2,2),bool),.5);t=hex_to_tet(X,h);V=geometry(X,t)[0]
    m=np.zeros(len(X));np.add.at(m,t.ravel(),np.repeat(V/4,4))
    p=Periodic(X,1.);p.mass(m);guess=np.zeros(p.ndof)
    v,report=project(p,X,m,guess,np.zeros(6),0.)
    np.testing.assert_array_equal(v,0.)
    assert report['kinetic_relative_error']==0.
    with pytest.raises(ValueError,match='No resolved momentum-null'):
        project(p,X,m,guess,np.zeros(6),1.)
