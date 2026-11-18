import numpy as np
import pytest
from packing_fem.explicit_material import hex_to_tet
from packing_fem.uniform_tet import refine
from tools.diagnose_elastic_mesh import solve


def test_diagnostic_recovers_homogeneous_constrained_modulus_before_and_after_refinement(tmp_path):
    pytest.importorskip('pypardiso')
    X=np.array([[i/2,j/2,k/2] for i in range(3) for j in range(3) for k in range(3)])
    def node(i,j,k):return i*9+j*3+k
    hexes=np.array([[node(i,j,k),node(i+1,j,k),node(i+1,j+1,k),node(i,j+1,k),node(i,j,k+1),node(i+1,j,k+1),node(i+1,j+1,k+1),node(i,j+1,k+1)] for i in range(2) for j in range(2) for k in range(2)])
    t=hex_to_tet(X,hexes)
    c=dict(young_mpa=10.,poisson=.33,yield_mpa=.05,hardening_mpa=.2,length_mm=1.,
           rigid_inclusion=True,rigid_diameter_mm=.1,rigid_height_mm=.1,implicit_sparse_solver='pardiso',implicit_sparse_threads=1)
    report,u=solve(X,t,c,tmp_path/'coarse')
    expected=10*(1-.33)/((1+.33)*(1-2*.33))
    assert report['effective_C33_MPa']==pytest.approx(expected,rel=1e-10)
    Y,s,edges=refine(X,t)
    fine,_=solve(Y,s,c,tmp_path/'fine',coarse=(X,u,edges,report['energy_Nmm']))
    assert fine['effective_C33_MPa']==pytest.approx(expected,rel=1e-10)
    assert fine['nested_space_energy_relative_error']<1e-9
