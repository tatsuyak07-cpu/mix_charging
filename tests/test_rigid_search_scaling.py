"""A balanced, almost-null rigid spin must not block an elastic body solve."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from packing_fem.rigid_inclusion import RigidPeriodic
from packing_fem.explicit_material import geometry,hex_to_tet
from packing_fem.implicit_material import evaluate
from packing_fem.quasistatic_coupled import equilibrium


class WeakSpin:
    """Exact weak torque and a displacement bound, independent of search damping."""
    def __init__(self,p):self.p=p;self.index=p.rigid_dofs[-1]
    def evaluate_coupled(self,q,oldq,tangent=False):
        g=np.zeros(self.p.ndof);g[self.index]=1e-3+1e-12*q[self.index]
        K=csr_matrix(([1e-12],([self.index],[self.index])),shape=(self.p.ndof,self.p.ndof))
        return 0.,g,K,1
    def ccd(self,q,dq):return min(1.,.1/max(float(abs(dq[:self.p.nmicro]).max()),1e-30))
    def force_balance_ok(self,q,tol):return abs(self.evaluate_coupled(q,q)[1][self.index])<tol


def case(floor):
    X=np.array([[i/4,j/2,k/2] for i in range(3) for j in range(3) for k in range(3)])
    def node(i,j,k):return i*9+j*3+k
    cells=np.array([[node(i,j,k),node(i+1,j,k),node(i+1,j+1,k),node(i,j+1,k),node(i,j,k+1),node(i+1,j,k+1),node(i+1,j+1,k+1),node(i,j+1,k+1)] for i in range(2) for j in range(2) for k in range(2)])
    t=hex_to_tet(X,cells)
    c=dict(rigid_inclusion=True,rigid_diameter_mm=.2,rigid_height_mm=.2,rigid_segments=16,rigid_center_mm=[.75,.5,.5],length_mm=1.,young_mpa=10.,poisson=.33,yield_mpa=1e6,hardening_mpa=0.,transverse_boundary='fixed_cell',implicit_force_tolerance_N=.005,implicit_regularization=1e-7,implicit_max_newton=8,implicit_coupled_max_iterations=8,implicit_rigid_step_scale_floor_N_per_mm=floor)
    p=RigidPeriodic(X,1.,c);F0=np.diag([.97**(-.33/(1-.33)),1.,.97]);q=p.remap_positions(X@F0.T,np.array([0.,0.,-.03,0.,0.,0.]))
    b=np.broadcast_to(F0@F0.T,(len(t),3,3)).copy();ep=np.zeros(len(t));V,_,D=geometry(p.positions(q),t)
    return p,t,q,b,ep,np.linalg.inv(D),V,c,WeakSpin(p)


def test_weak_balanced_spin_can_choke_unscaled_body_search():
    p,t,q,b,ep,inverse,V,c,contact=case(1.)
    result=equilibrium(p,t,q,b,ep,inverse,V,c,contact,-.06,np.zeros(p.ndof),lambda event:None)
    assert isinstance(result,dict)


@pytest.mark.parametrize('floor',[1e5,1e6,1e7])
def test_rigid_search_scaling_keeps_original_force_and_all_six_free_dofs(floor):
    p,t,q,b,ep,inverse,V,c,contact=case(floor);originals=[v.copy() for v in [q,b,ep]]
    result=equilibrium(p,t,q,b,ep,inverse,V,c,contact,-.06,np.zeros(p.ndof),lambda event:None)
    assert not isinstance(result,dict),result
    unknown,state,internal,_,iterations,norm=result
    _,gm,_,_=evaluate(p.positions(unknown),t,b,ep,inverse,V,c)
    physical=np.asarray(p.A.T@gm.ravel()).ravel()+contact.evaluate_coupled(unknown,q)[1]
    np.testing.assert_allclose(internal,physical,atol=1e-12)
    assert abs(physical[:p.nmicro]).max()<c['implicit_force_tolerance_N']
    assert abs(physical[contact.index])>5e-4  # No fictitious balancing spring force.
    assert len(p.rigid_dofs)==6 and unknown[contact.index]!=q[contact.index]
    F1=np.diag([.94**(-.33/(1-.33)),1.,.94])
    np.testing.assert_allclose(p.positions(unknown),p.X@F1.T,atol=2e-4)
    assert iterations<8 and norm<c['implicit_force_tolerance_N']
    for current,old in zip([q,b,ep],originals):np.testing.assert_array_equal(current,old)
    np.testing.assert_array_equal(state['ep'],0.)
