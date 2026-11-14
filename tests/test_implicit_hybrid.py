"""Finite prestrained analytic slab: globalization must not advance history."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from packing_fem.explicit_material import geometry,hex_to_tet
from packing_fem.explicit_periodic import Periodic
from tools.run_implicit_compaction import equilibrium


class NoContact:
    def __init__(self,n):self.n=n
    def set_lag(self,*args):pass
    def evaluate(self,*args,**kwargs):return 0.,np.zeros(self.n),csr_matrix((self.n,self.n)),0
    def evaluate_coupled(self,*args,**kwargs):return self.evaluate()
    def ccd(self,*args):return 1.


@pytest.mark.parametrize('polish_budget',[0,20])
def test_globalization_and_polish_share_one_accepted_finite_material_history(polish_budget):
    X=np.array([[i/4,j/2,k/2] for i in range(3) for j in range(3) for k in range(3)])
    def node(i,j,k):return i*9+j*3+k
    cells=np.array([[node(i,j,k),node(i+1,j,k),node(i+1,j+1,k),node(i,j+1,k),node(i,j,k+1),node(i+1,j,k+1),node(i+1,j+1,k+1),node(i,j+1,k+1)] for i in range(2) for j in range(2) for k in range(2)])
    t=hex_to_tet(X,cells);p=Periodic(X,1.);V,_,_=geometry(X,t)
    def stretch(z):return np.diag([z**(-.33/(1-.33)),1.,z])
    F0=stretch(.97);macro=np.array([0.,0.,-.03,0.,0.,0.]);q=p.remap_positions(X@F0.T,macro)
    b=np.broadcast_to(F0@F0.T,(len(t),3,3)).copy();ep=np.zeros(len(t));inverse=np.linalg.inv(geometry(p.positions(q),t)[2])
    originals=[v.copy() for v in [q,b,ep]]
    c=dict(young_mpa=10.,poisson=.33,yield_mpa=1e6,hardening_mpa=0.,transverse_boundary='fixed_cell',implicit_optimizer='lagged_coupled',implicit_force_tolerance_N=1e-10,implicit_regularization=1e-8,implicit_macro_tolerance=1e-12,implicit_max_lag=8,implicit_max_newton=30,implicit_coupled_max_iterations=polish_budget,implicit_hybrid_initial_newton_iterations=0,implicit_hybrid_warm_lags=1,implicit_hybrid_warm_updates_per_lag=1)
    events=[];result=equilibrium(p,t,q,b,ep,inverse,V,c,NoContact(p.ndof),-.06,np.zeros(p.ndof),events.append)
    for current,expected in zip([q,b,ep],originals):np.testing.assert_array_equal(current,expected)
    assert any(event.get('hybrid_phase')=='energy_globalization' for event in events)
    assert any(event.get('hybrid_phase')=='original_force_polish' for event in events)
    if polish_budget==0:
        assert isinstance(result,dict) and 'budget exhausted' in result['reason']
        return
    assert not isinstance(result,dict),result
    newq,state,internal,_,iterations,norm=result;F1=stretch(.94)
    np.testing.assert_allclose(p.positions(newq),X@F1.T,atol=2e-10)
    np.testing.assert_allclose(state['b'],np.broadcast_to(F1@F1.T,state['b'].shape),rtol=2e-9,atol=2e-10)
    np.testing.assert_array_equal(state['ep'],0)
    assert norm<c['implicit_force_tolerance_N'] and iterations>1
