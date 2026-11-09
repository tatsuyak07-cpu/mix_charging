import numpy as np
import pytest
from packing_fem.load_predictor import search_stalled,bisected_initial_guess
from packing_fem.explicit_material import geometry,hex_to_tet
from packing_fem.explicit_periodic import Periodic


@pytest.mark.parametrize('norms,steps,expected',[(.01*.95**np.arange(10),[1e-9]*9,False),([.0073]*10,[1e-9]*9,True),([.0073]*10,[1e-4]*9,False)])
def test_stagnation_distinguishes_contraction_and_configuration_motion(norms,steps,expected):
    assert search_stalled(norms,steps)==expected


@pytest.mark.parametrize('fraction',[.5,1.,1.25])
def test_bisection_falls_back_to_exact_affine_elastic_state_without_history_changes(fraction):
    X=np.array([[i/2,j/2,k/2] for i in range(3) for j in range(3) for k in range(3)])
    def node(i,j,k):return i*9+j*3+k
    h=np.array([[node(i,j,k),node(i+1,j,k),node(i+1,j+1,k),node(i,j+1,k),node(i,j,k+1),node(i+1,j,k+1),node(i+1,j+1,k+1),node(i,j+1,k+1)] for i in range(2) for j in range(2) for k in range(2)])
    t=hex_to_tet(X,h);p=Periodic(X,1.);V,_,D=geometry(X,t);old=np.zeros(p.ndof);trial=old.copy();trial[p.prescribed]=-.02;trial[:p.nmicro]=np.random.default_rng(19).normal(scale=1e-5,size=p.nmicro)
    b=np.tile(np.eye(3),(len(t),1,1));ep=np.zeros(len(t));original_b=b.copy()
    class EmptyContact:
        def ccd(self,*args):return 1.
        def evaluate_coupled(self,*args):return 0.,np.zeros(p.ndof),None,0
    c=dict(young_mpa=10.,poisson=.33,yield_mpa=1e6,hardening_mpa=0.)
    q,report=bisected_initial_guess(p,t,old,trial,-.02*fraction,b,ep,np.linalg.inv(D),V,np.arange(len(t)),c,EmptyContact())
    np.testing.assert_allclose(p.positions(q),X@np.diag([1.,1.,1-.02*fraction]),atol=1e-14)
    np.testing.assert_array_equal(b,original_b);np.testing.assert_array_equal(ep,0);np.testing.assert_array_equal(old,0)
    assert report['initial_guess_only'] and report['load_fraction']==pytest.approx(fraction)


def test_driver_bisection_reuses_geometry_but_advances_only_accepted_elastic_history(tmp_path,monkeypatch):
    import json
    from scipy.sparse import csr_matrix
    import tools.run_implicit_compaction as driver
    X=np.array([[i/4,j/2,k/2] for i in range(3) for j in range(3) for k in range(3)])
    def node(i,j,k):return i*9+j*3+k
    h=np.array([[node(i,j,k),node(i+1,j,k),node(i+1,j+1,k),node(i,j+1,k),node(i,j,k+1),node(i+1,j,k+1),node(i+1,j+1,k+1),node(i,j+1,k+1)] for i in range(2) for j in range(2) for k in range(2)])
    t=hex_to_tet(X,h);mesh=tmp_path/'mesh.npz';np.savez(mesh,points=X,tets=t)
    class NoSurfaceContact:
        def __init__(self,X,t,p,c):self.X=X;self.n=p.ndof
        def ccd(self,*args):return 1.
        def evaluate_coupled(self,q,old,tangent=False):return 0.,np.zeros(self.n),csr_matrix((self.n,self.n)),0
    monkeypatch.setattr(driver,'PeriodicIPC',NoSurfaceContact)
    actual=driver.equilibrium;calls=[]
    def reject_first(p,t,q,b,ep,inverse,V0,c,contact,prescribed,external,log,parent=None,initial_guess=None):
        calls.append((q.copy(),b.copy(),ep.copy(),prescribed))
        if len(calls)==1:return dict(reason='Manufactured iteration-budget rejection',iterate=driver.affine_predictor(p,q,prescribed))
        return actual(p,t,q,b,ep,inverse,V0,c,contact,prescribed,external,log,parent,initial_guess)
    monkeypatch.setattr(driver,'equilibrium',reject_first)
    c=dict(length_mm=1.,young_mpa=10.,poisson=.33,yield_mpa=1e6,hardening_mpa=0.,target_final_porosity=.49,porosity_tolerance=1e-5,maximum_axial_compression=.1,implicit_strain_increment=.04,implicit_minimum_increment=1e-6,implicit_max_stages=8,implicit_max_newton=30,implicit_force_tolerance_N=1e-8,implicit_regularization=1e-8,implicit_macro_tolerance=1e-10,implicit_remesh=False,implicit_optimizer='coupled_newton',transverse_boundary='fixed_cell',loading_only=True,implicit_bisection_predictor=True)
    config=tmp_path/'config.json';config.write_text(json.dumps(c));out=tmp_path/'run';driver.run(config,mesh,out)
    assert len(calls)==3
    np.testing.assert_array_equal(calls[1][1],calls[0][1]);np.testing.assert_array_equal(calls[1][2],calls[0][2])
    assert calls[1][3]==pytest.approx(-.02) and calls[2][3]==pytest.approx(-.045)
    assert not np.array_equal(calls[2][1],calls[1][1])
    with np.load(out/'state-0002.npz') as f:
        expected_stretch=.955**(-.33/(1-.33));F=np.diag([expected_stretch,1.,.955])
        np.testing.assert_allclose(f['b'],np.broadcast_to(F@F.T,f['b'].shape),rtol=1e-6,atol=1e-8)
        np.testing.assert_array_equal(f['ep'],0)
    assert json.loads((out/'status.json').read_text())['target_reached']
    assert len(list(out.glob('predictor-*.npz')))==2
