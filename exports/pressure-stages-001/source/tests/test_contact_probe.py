import numpy as np
import pytest
from packing_fem import sintered
from packing_fem.explicit_material import hex_to_tet
from packing_fem.contact_probe import PeriodicSolidProbe,friction_return


def fixture(axis=0,roll=0):
    mask=np.zeros((3,3,3),bool);mask[0,1,1]=True;mask[2,1,1]=True
    mask=np.roll(mask,roll,axis=0);mask=np.moveaxis(mask,0,axis)
    X,h,_=sintered.mesh(mask,1.)
    return X,PeriodicSolidProbe(X,hex_to_tet(X,h),3.,100.,.2)


@pytest.mark.parametrize('axis',[0,1,2])
def test_cut_surface_is_not_contact_surface(axis):
    X,probe=fixture(axis);other=(axis+1)%3
    for coordinate in (.02,2.98,3.02,-.02):
        p=np.ones(3)*1.5;p[axis]=coordinate;p[other]=1.04
        r=probe.evaluate(p,X,np.eye(3)*3)
        expected=np.zeros(3);expected[other]=-4.
        assert r['inside'] and r['energy']==pytest.approx(.08)
        np.testing.assert_allclose(r['force'],expected,atol=1e-12)
    p=np.ones(3)*1.5;p[axis]=.02;p[other]=.98
    assert not probe.evaluate(p,X,np.eye(3)*3)['inside']


def test_cell_cut_translation_and_generalized_work():
    X,probe=fixture();p=np.array([.02,1.04,1.5]);H=np.eye(3)*3
    r=probe.evaluate(p,X,H)
    for s in ([1,0,0],[-2,1,0],[0,0,-1]):
        s=np.array(s);rr=probe.evaluate(p+H@s,X,H)
        np.testing.assert_allclose(rr['force'],r['force'],atol=1e-12)
        # Include the probe's own reference coordinate in macro work.
        a=np.outer(r['force'],p)+r['master_force'].T@X+r['image_force_tensor']
        b=np.outer(rr['force'],p+H@s)+rr['master_force'].T@X+rr['image_force_tensor']
        np.testing.assert_allclose(a,b,atol=1e-12)
    # Same physical bar shifted one cell in the grid; cut moves into the pore.
    Y,other=fixture(roll=1)
    rr=other.evaluate(p+np.array([1,0,0]),Y,H)
    assert rr['energy']==pytest.approx(r['energy'])
    np.testing.assert_allclose(rr['force'],r['force'],atol=1e-12)


def test_deformed_periodic_cell_force_energy_and_image_work():
    X,probe=fixture();F=np.array([[1.,.1,0],[.1,1.03,.02],[0,.02,.97]])
    x=X@F.T;H=F*3;p=F@np.array([3.02,1.04,1.5]);r=probe.evaluate(p,x,H);eps=1e-7
    assert r['inside'];np.testing.assert_allclose(r['force']+r['master_force'].sum(axis=0),0,atol=1e-12)
    moment=np.cross(p,r['force'])+np.cross(x,r['master_force']).sum(axis=0)
    moment+=sum((np.cross(H@r['shift'],f) for f in r['master_force']),start=np.zeros(3))
    np.testing.assert_allclose(moment,0,atol=1e-12)
    for axis in range(3):
        dp=np.eye(3)[axis]*eps
        grad=(probe.evaluate(p+dp,x,H)['energy']-probe.evaluate(p-dp,x,H)['energy'])/(2*eps)
        assert -grad==pytest.approx(r['force'][axis],abs=1e-7)
    for i in range(3):
        for j in range(3):
            dH=np.zeros((3,3));dH[i,j]=eps
            grad=(probe.evaluate(p,x,H+dH)['energy']-probe.evaluate(p,x,H-dH)['energy'])/(2*eps)
            assert -grad==pytest.approx(r['image_force_tensor'][i,j]/3,abs=1e-7)


def test_friction_transfer_no_reset_and_discrete_work_identity():
    n=np.array([0.,0.,1.]);spring=np.zeros(3);friction=0.
    for step in range(100):
        r=friction_return(spring,n,n,[.001,0,0],1.,100.,.3)
        spring=r['spring'];friction+=r['friction_loss']
        assert np.linalg.norm(r['force'])<=.3+1e-14
        assert r['friction_loss']>=-1e-14 and r['algorithmic_loss']>=0
        assert abs(r['balance_residual'])<1e-15
    assert friction>0
    assert spring[0]==pytest.approx(.003)
    saved=r['energy']
    # Face numbering does not enter the known-patch state; a zero increment keeps it.
    r=friction_return(spring,n,n,[0,0,0],1.,100.,.3)
    assert r['energy']==pytest.approx(saved)
    np.testing.assert_array_equal(r['spring'],spring)


def test_spring_rotation_is_objective_including_spin_about_normal():
    n=np.array([0.,0.,1.]);s=np.array([.001,0,0]);theta=.71
    R=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1.]])
    base=friction_return(s,n,n,[.0001,0,0],1.,100.,.3)
    rotated=friction_return(s,n,R@n,R@np.array([.0001,0,0]),1.,100.,.3,rotation=R)
    np.testing.assert_allclose(rotated['force'],R@base['force'],atol=1e-14)
    assert rotated['energy']==pytest.approx(base['energy'])
    for theta in np.linspace(0,np.pi/2,21):
        new=np.array([np.sin(theta),0,np.cos(theta)])
        r=friction_return(s,n,new,np.zeros(3),1.,100.,.3)
        assert r['energy']==pytest.approx(.00005)
        assert abs(r['spring']@new)<1e-15
    with pytest.raises(ValueError,match='Antipodal'):friction_return(s,n,-n,[0,0,0],1.,100.,.3)


def test_reject_ambiguous_or_invalid_probe_and_transport_inputs():
    X,probe=fixture()
    with pytest.raises(ValueError,match='beyond'):probe.evaluate([.4,1.5,1.5],X,np.eye(3)*3)
    # Large search range must fail explicitly rather than silently omit images.
    wide=PeriodicSolidProbe(X,probe.t,3.,100.,20.)
    with pytest.raises(ValueError,match='budget'):wide.evaluate([.02,1.04,1.5],X,np.eye(3)*3)
    n=np.array([0.,0.,1.]);s=np.array([.001,0,0])
    with pytest.raises(ValueError,match='Rotation'):friction_return(s,n,n,[0,0,0],1.,100.,.3,rotation=np.eye(3)*1.000001)
    with pytest.raises(ValueError,match='Unit'):friction_return(s,n,n*1.000001,[0,0,0],1.,100.,.3)
