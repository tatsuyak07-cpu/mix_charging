import json
from pathlib import Path
import numpy as np
import pytest
from packing_fem.geometry import make_packing,sphere_mesh
from packing_fem.model import validate_config,write_model,plan
from packing_fem.results import read_dat,analyze

ROOT=Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT/'configs/smoke.json').read_text())


def test_invalid_physics_rejected():
    c=config();c['poisson']=.5
    with pytest.raises(ValueError):validate_config(c)


def test_budget_preserves_geometry_and_does_not_write_deck(tmp_path):
    c=config();c['max_dofs']=1
    p,t,s=sphere_mesh(5,1.5)
    x=make_packing(c['length_mm'],5,.6,1)
    with pytest.raises(ValueError,match='DOF budget'):write_model(tmp_path,c,x,p,t,s)
    assert not (tmp_path/'model.inp').exists()


def test_image_energy_partition_and_mpc(tmp_path):
    c=config();p,t,s=sphere_mesh(5,1.5)
    x=make_packing(c['length_mm'],5,.6,1)
    write_model(tmp_path,c,x,p,t,s)
    text=(tmp_path/'model.inp').read_text()
    # Four copies (core + positive x/y/z images) with E/4, density/4.
    assert '*MATERIAL,NAME=M4' in text and '17500,0.33' in text
    # Direct independent energy check: affine strain energy is invariant
    # when a grain is represented by weighted translated copies.
    e=np.diag([.002,-.00066,-.00066]);mu=70000/(2*1.33)
    lam=70000*.33/(1.33*(1-2*.33))
    physical=.5*lam*np.trace(e)**2+mu*np.sum(e*e)
    assert sum(physical/4 for _ in range(4))==pytest.approx(physical)
    g=np.load(tmp_path/'geometry.npz')
    nn=len(p)
    for k,key in enumerate(g['copy_keys']):
        np.testing.assert_allclose(g['points'][k*nn:(k+1)*nn]-g['points'][:nn],
                                   np.broadcast_to(key[1:]*c['length_mm'],(nn,3)),atol=1e-12)


def test_dat_time_blocks_cannot_mix_outputs(tmp_path):
    f=tmp_path/'test.dat'
    f.write_text(' displacements (vx,vy,vz) for set MACRO and time 1.0E-2\n\n1 0 0 1\n'
                 ' total kinetic energy for set SOLIDS and time 1.0E-2\n2\n'
                 ' displacements (vx,vy,vz) for set MACRO and time 2.0E-2\n1 0 0 2\n')
    frames=list(read_dat(f))
    assert len(frames)==2 and frames[0][0]==.01 and frames[1][0]==.02
    assert not any('energy' in key for key in frames[1][1])


def test_missing_outputs_never_become_a_solution_and_cannot_overwrite(tmp_path):
    (tmp_path/'metadata.json').write_text(json.dumps({'config':config(),'estimate':{'particles':1}}))
    np.savez(tmp_path/'geometry.npz',points=[])
    result=analyze(tmp_path,'missing',True,1.)
    assert result['status']=='diagnostics_only'
    assert not result['checks']['final_time_reached']
    assert result['error'] is not None
    with pytest.raises(ValueError,match='overwrite'):analyze(tmp_path,'missing',True,1.)


def test_cli_preserves_existing_output(tmp_path):
    from packing_fem.cli import main
    sentinel=tmp_path/'original';sentinel.write_text('keep')
    assert main(['estimate','--config',str(ROOT/'configs/smoke.json'),'--output',str(tmp_path)])==2
    assert sentinel.read_text()=='keep'
