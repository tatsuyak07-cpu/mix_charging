import numpy as np
import pytest
pytest.importorskip("ipctk")
from scipy.sparse import csr_matrix
from packing_fem.explicit_material import geometry,hex_to_tet
from packing_fem.explicit_periodic import Periodic,deformation
from tools.run_implicit_compaction import active_dofs,equilibrium,affine_predictor

class NoContact:
    def __init__(self,n):self.n=n;self.friction=object()
    def set_lag(self,*args):pass
    def evaluate(self,q,tangent=False,project_hessian=None):return 0.,np.zeros(self.n),csr_matrix((self.n,self.n)),0
    def ccd(self,*args):return 1.
    def evaluate_coupled(self,q,old,tangent=False):return self.evaluate(q,tangent)

@pytest.mark.parametrize('sparse_backend',['superlu','pardiso'])
@pytest.mark.parametrize('acceleration_depth',[0,4])
@pytest.mark.parametrize('optimizer,project_material_tangent,tangent_switch,search',[('newton',False,np.inf,'armijo'),('newton',False,1e-3,'armijo'),('newton',True,np.inf,'armijo'),('lbfgs',True,1e-3,'armijo'),('lagged_lbfgs',True,1e-3,'armijo'),('lagged_coupled',True,1e-3,'armijo'),('lbfgs',True,1e-3,'strong_wolfe'),('coupled_newton',False,1e-3,'armijo'),('coupled_newton',False,1e-3,'diagonal'),('coupled_newton',False,1e-3,'homotopy'),('coupled_newton',False,1e-3,'maximum'),('coupled_newton',False,1e-3,'local'),('coupled_newton',False,1e-3,'hybrid'),('coupled_newton',False,1e-3,'peak'),('coupled_newton',False,1e-3,'force_box')])
@pytest.mark.parametrize('use_initial_guess',[False,True])
def test_fixed_cell_matches_finite_elastic_uniaxial_strain_limit(acceleration_depth,optimizer,project_material_tangent,tangent_switch,search,use_initial_guess,sparse_backend):
    if sparse_backend=='pardiso':pytest.importorskip('pypardiso')
    X=np.array([[i/2,j/2,k/2] for i in range(3) for j in range(3) for k in range(3)])
    def node(i,j,k):return i*9+j*3+k
    hexes=np.array([[node(i,j,k),node(i+1,j,k),node(i+1,j+1,k),node(i,j+1,k),node(i,j,k+1),node(i+1,j,k+1),node(i+1,j+1,k+1),node(i,j+1,k+1)] for i in range(2) for j in range(2) for k in range(2)])
    t=hex_to_tet(X,hexes);p=Periodic(X,1.);v,_,D=geometry(X,t)
    c=dict(transverse_boundary='fixed_cell',young_mpa=10.,poisson=.33,yield_mpa=1e6,hardening_mpa=0.,implicit_max_lag=8,implicit_max_newton=15,implicit_force_tolerance_N=1e-8,implicit_regularization=1e-8,implicit_macro_tolerance=1e-10)
    c['implicit_sparse_solver']=sparse_backend
    c['implicit_anderson_depth']=acceleration_depth
    c['implicit_project_material_tangent']=project_material_tangent
    c['implicit_exact_tangent_max_residual_N']=tangent_switch
    c['implicit_affine_predictor']=np.isfinite(tangent_switch)
    c['implicit_optimizer']=optimizer
    c['implicit_lbfgs_line_search']=search
    c['implicit_coupled_merit_weighting']='diagonal' if search=='diagonal' else 'none'
    c['implicit_coupled_least_squares_fallback']=search in ('diagonal','maximum','local','hybrid','peak','force_box')
    c['implicit_coupled_merit_norm']='hybrid' if search=='hybrid' else ('maximum' if search in ('maximum','local','peak','force_box') else 'l2')
    if search=='force_box':c.update(implicit_peak_violation_fraction=.8,implicit_force_box_inactive_weight=.05)
    if search=='peak':c['implicit_coupled_peak_lsmr']=True
    if search=='local':c.update(implicit_coupled_local_relaxation=True,implicit_local_force_threshold_multiplier=1e8)
    if search=='homotopy':c.update(implicit_friction_smoothing_multipliers=[100.,10.,1.],ipc_slip_regularization_mm=1e-4)
    q=np.zeros(p.ndof);b=np.tile(np.eye(3),(len(t),1,1));ep=np.zeros(len(t))
    guess=q.copy() if use_initial_guess else None
    if guess is not None:
        guess[p.prescribed]=-.00008;guess[:p.nmicro]=np.random.default_rng(14).normal(scale=1e-6,size=p.nmicro)
    contact=NoContact(p.ndof);original_friction=contact.friction
    events=[]
    result=equilibrium(p,t,q,b,ep,np.linalg.inv(D),v,c,contact,-.0001,np.zeros(p.ndof),events.append,initial_guess=guess)
    if search=='local' and use_initial_guess:assert any(row.get('local_force_relaxation') for row in events)
    assert contact.friction is original_friction
    assert not isinstance(result,dict),result
    q,state,g,_,_,_=result
    np.testing.assert_array_equal(q[-6:][[0,1,3,4,5]],0)
    stretch=1+q[p.prescribed];G=10/(2*1.33);lame=10*.33/(1.33*.34)
    np.testing.assert_allclose(g[p.prescribed],(lame+2*G)*np.log(stretch)/stretch,rtol=1e-5)
    np.testing.assert_allclose(g[p.nmicro:p.nmicro+2],lame*np.log(stretch),rtol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(deformation(q[-6:])[:,:2],axis=0),[1,1],atol=0)
    _,physical=active_dofs(p,c,None)
    assert p.prescribed in physical
    assert p.nmicro not in physical


def test_affine_predictor_maps_actual_nonaffine_geometry_and_element_volume():
    X=np.array([[i/2,j/2,k/2] for i in range(3) for j in range(3) for k in range(3)])
    p=Periodic(X,1.);q=np.zeros(p.ndof);q[:p.nmicro]=np.random.default_rng(42).normal(scale=.001,size=p.nmicro);q[p.prescribed]=-.1
    original=q.copy();trial=affine_predictor(p,q,-.2)
    change=np.diag([1.,1.,.8/.9])
    np.testing.assert_allclose(p.positions(trial),p.positions(q)@change.T,atol=1e-14)
    np.testing.assert_array_equal(q,original)
    t=np.array([[0,9,3,1]])
    old_volume=geometry(p.positions(q),t)[0];new_volume=geometry(p.positions(trial),t)[0]
    assert old_volume[0]>0 and new_volume[0]>0
    np.testing.assert_allclose(new_volume/old_volume,np.linalg.det(change),atol=1e-12)
    np.testing.assert_array_equal(trial[-6:][[0,1,3,4,5]],0.)
