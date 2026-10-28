import json
from pathlib import Path
import shutil
import numpy as np
import pytest
from packing_fem import sintered
from packing_fem.model import validate_config

ROOT=Path(__file__).resolve().parents[1]


def test_new_specs_do_not_silently_use_friction():
    c=json.loads((ROOT/'configs/baseline.json').read_text())
    assert c['diameter_mm']==10 and c['neck_diameter_mm']==2
    with pytest.raises(ValueError,match='sintered backend'):validate_config(c)


def test_hex_volume_affine_and_rotation_invariant():
    mask=np.ones((2,2,2),bool);p,e,_=sintered.mesh(mask,1.)
    assert sintered.hex_quadrature(p,e).sum()==pytest.approx(8)
    F=np.array([[1.2,.1,0],[0,.8,0],[0,0,.9]])
    assert sintered.hex_quadrature(p@F.T+5,e).sum()==pytest.approx(8*np.linalg.det(F))


def test_periodic_solid_connectivity_and_winding():
    assert sintered.connectivity(np.ones((4,4,4),bool))==(1,3)
    mask=np.zeros((4,4,4),bool);mask[1:3,1:3,1:3]=True
    assert sintered.connectivity(mask)==(1,0)
    mask=np.zeros((8,8,8),bool);mask[1,1,1]=True;mask[5,5,5]=True
    assert sintered.connectivity(mask)==(2,0)


def test_periodic_raster_union_adds_real_neck_volume():
    mask=np.zeros((20,20,20),bool)
    sintered.draw(mask,np.array([.5,10,10]),np.array([.5,10,10]),2.,1.)
    assert mask[0].any() and mask[-1].any()
    count=mask.sum()
    sintered.draw(mask,np.array([.5,10,10]),np.array([8.,10,10]),1.,1.)
    assert mask.sum()>count
    old=mask.copy()
    sintered.draw(mask,np.array([.5,10,10]),np.array([8.,10,10]),1.,1.)
    np.testing.assert_array_equal(mask,old)


@pytest.mark.skipif(shutil.which('/workspace/environment/calculix/usr/bin/ccx') is None,reason='Native FEM unavailable')
def test_sintered_periodic_backend_against_homogeneous_elastic_solution(tmp_path):
    c=json.loads((ROOT/'configs/baseline.json').read_text());c['length_mm']=2.
    p,e,idx=sintered.mesh(np.ones((2,2,2),bool),1.)
    estimate={'grid_cells_per_axis':2,'porosity_error':0,'periodic_components':1,
              'winding_rank':3,'upper_bound_dofs':3*len(p)+6,'initial_porosity':0.}
    np.savez(tmp_path/'sintered-geometry.npz',points=p,elements=e)
    sintered.write_model(tmp_path,c,p,e,idx,estimate)
    r=sintered.solve(tmp_path,'/workspace/environment/calculix/usr/bin/ccx',30)
    assert r['solver_exit_code']==0
    np.testing.assert_allclose(r['mean_stress_mpa'],[0,0,-.1,0,0,0],atol=1e-7)
    E=np.array(r['macro_strain'])
    np.testing.assert_allclose(np.diag(E),[.33*.1/70000,.33*.1/70000,-.1/70000],rtol=1e-5)


def test_mesh_phase_preserves_periodic_particle_geometry():
    c=json.loads((ROOT/'configs/porosity40-to10-trial.json').read_text())
    moved,mask,_,_,bonds=sintered.voxel_geometry(c,18)
    base={**c,'mesh_phase_mm':[0.,0.,0.]}
    original,_,_,_,base_bonds=sintered.voxel_geometry(base,18)
    delta=moved-original-np.array(c['mesh_phase_mm'])
    np.testing.assert_allclose(delta/c['length_mm'],np.rint(delta/c['length_mm']),atol=1e-14)
    for points,other in [(moved,original)]:
        d=points[:,None,:]-points[None,:,:];d-=c['length_mm']*np.rint(d/c['length_mm'])
        d0=other[:,None,:]-other[None,:,:];d0-=c['length_mm']*np.rint(d0/c['length_mm'])
        np.testing.assert_allclose(np.linalg.norm(d,axis=-1),np.linalg.norm(d0,axis=-1),atol=1e-12)
    assert bonds==base_bonds
    assert sintered.connectivity(mask)==(1,3)
    assert abs((1-mask.mean())-c['porosity'])<c['porosity_tolerance']
    with pytest.raises(ValueError,match='three finite'):
        sintered.validate({**c,'mesh_phase_mm':[0.,float('nan'),0.]})
