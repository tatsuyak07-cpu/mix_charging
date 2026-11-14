"""Finite elastic slab: a restart must retain the actual accepted porosity slope."""
import csv
import json
import numpy as np
import pytest
pytest.importorskip('ipctk')
from scipy.sparse import csr_matrix
from packing_fem.explicit_material import geometry, hex_to_tet
from packing_fem.explicit_periodic import Periodic
from tools import run_implicit_compaction as driver


def test_finite_slab_restart_reaches_target_with_once_updated_history(tmp_path, monkeypatch):
    X=np.array([[i/4,j/2,k/2] for i in range(3) for j in range(3) for k in range(3)])
    def node(i,j,k):return i*9+j*3+k
    cells=np.array([[node(i,j,k),node(i+1,j,k),node(i+1,j+1,k),node(i,j+1,k),node(i,j,k+1),node(i+1,j,k+1),node(i+1,j+1,k+1),node(i,j+1,k+1)] for i in range(2) for j in range(2) for k in range(2)])
    t=hex_to_tet(X,cells);p=Periodic(X,1.);V,_,_=geometry(X,t)
    mesh=tmp_path/'mesh.npz';np.savez(mesh,points=X,tets=t)
    exponent=.33/(1-.33)
    def save(phi,path):
        z=(.5/(1-phi))**(1/exponent);F=np.diag([z**(-exponent),1.,z])
        q=p.remap_positions(X@F.T,np.array([0.,0.,z-1,0.,0.,0.]))
        np.savez(path,X=X,x=p.positions(q),t=t,q=q,b=np.broadcast_to(F@F.T,(len(t),3,3)),ep=np.zeros(len(t)),V0=V,parent=np.arange(len(t)))
        return 1-z
    previous=tmp_path/'previous.npz';current=tmp_path/'current.npz'
    old_compression=save(.14,previous);compression=save(.13,current)
    class NoContact:
        def __init__(self,X,t,p,c):self.X=X;self.n=p.ndof
        def ccd(self,*args):return 1.
        def evaluate_coupled(self,*args,**kwargs):return 0.,np.zeros(self.n),csr_matrix((self.n,self.n)),0
    monkeypatch.setattr(driver,'PeriodicIPC',NoContact)
    c=dict(length_mm=1.,young_mpa=10.,poisson=.33,yield_mpa=1e6,hardening_mpa=0.,target_final_porosity=.1,porosity_tolerance=.001,maximum_axial_compression=.95,implicit_strain_increment=.08,implicit_minimum_increment=1e-6,implicit_max_stages=32,implicit_max_newton=30,implicit_force_tolerance_N=1e-8,implicit_regularization=1e-8,implicit_macro_tolerance=1e-10,implicit_remesh=False,implicit_optimizer='coupled_newton',transverse_boundary='fixed_cell',loading_only=True,implicit_previous_accepted_state=str(previous),implicit_slip_predictor=False)
    config=tmp_path/'config.json';config.write_text(json.dumps(c));out=tmp_path/'run'
    driver.run(config,mesh,out,restart=current)
    with (out/'history.csv').open() as f:rows=list(csv.DictReader(f))
    secant=(.14-.13)/(compression-old_compression)
    assert float(rows[0]['compression'])==pytest.approx(compression+.5*(.13-.1)/secant,abs=1e-12)
    assert .1<=float(rows[-1]['porosity'])<=.101
    assert json.loads((out/'status.json').read_text())['target_reached']
    for path in out.glob('state-*.npz'):
        with np.load(path) as state:
            z=1+state['q'][-4];F=np.diag([z**(-exponent),1.,z])
            np.testing.assert_allclose(state['b'],np.broadcast_to(F@F.T,state['b'].shape),rtol=1e-6,atol=1e-8)
            np.testing.assert_array_equal(state['ep'],0)
