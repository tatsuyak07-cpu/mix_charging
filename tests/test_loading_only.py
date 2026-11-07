"""Compression-only runs must not silently enter the hold/unload cycle."""
import json
import numpy as np
import pytest
pytest.importorskip('ipctk')
from packing_fem.explicit_material import hex_to_tet
from tools import run_implicit_compaction as driver


def test_already_loaded_input_stops_without_an_unrequested_hold(tmp_path,monkeypatch):
    X=np.array([[i/2,j/2,k/2] for i in range(3) for j in range(3) for k in range(3)])
    def node(i,j,k):return i*9+j*3+k
    hexes=np.array([[node(i,j,k),node(i+1,j,k),node(i+1,j+1,k),node(i,j+1,k),node(i,j,k+1),node(i+1,j,k+1),node(i+1,j+1,k+1),node(i,j+1,k+1)] for i in range(2) for j in range(2) for k in range(2)])
    mesh=tmp_path/'mesh.npz';np.savez(mesh,points=X,tets=hex_to_tet(X,hexes))
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(length_mm=1.,implicit_strain_increment=.025,loading_only=True,target_final_porosity=.1,porosity_tolerance=.001)))
    class InitialContact:
        def __init__(self,X,t,p,c):self.X=X
    monkeypatch.setattr(driver,'PeriodicIPC',InitialContact)
    def unexpected_equilibrium(*args,**kwargs):pytest.fail('An already loaded input must not perform hold or unloading')
    monkeypatch.setattr(driver,'equilibrium',unexpected_equilibrium)
    out=tmp_path/'run';driver.run(config,mesh,out)
    status=json.loads((out/'status.json').read_text())
    assert status['status']=='loaded_target_reached'
    assert status['target_reached']
    assert (out/'state-0000.npz').exists()
    assert not (out/'history.csv').exists()
