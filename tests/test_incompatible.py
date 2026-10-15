"""Native 2.20 restart/topology checks and independent constitutive/stress limits."""
import json,shutil
from pathlib import Path
import numpy as np
import pytest
from packing_fem import sintered,plastic
from packing_fem.incompatible import write_model,run,read_restart_groups,restart_state
from packing_fem.energy import audit
from packing_fem.results import read_dat,find

ROOT=Path(__file__).resolve().parents[1]
SOLVER='/workspace/environment/calculix/usr/bin/ccx'


@pytest.mark.skipif(shutil.which(SOLVER) is None,reason='Native CalculiX unavailable')
@pytest.mark.parametrize('void,control',[(False,'displacement'),(True,'displacement'),(True,'force')])
def test_native_mode_reconstruction_energy_and_prescribed_force(tmp_path,void,control):
    c=json.loads((ROOT/'configs/plastic-smoke.json').read_text())
    c.update(length_mm=3.,compression_fraction=.03,element_type='C3D8I',compression_steps=6,control=control)
    if control=='force':c['target_mpa']=.001
    mask=np.ones((3,3,3),bool)
    if void:mask[1,1,1]=False
    p,e,idx=sintered.mesh(mask,1.);component,rank=sintered.connectivity(mask)
    estimate={'grid_cells_per_axis':3,'porosity_error':0,'periodic_components':component,'winding_rank':rank,
              'upper_bound_dofs':3*len(p)+6,'initial_porosity':float(1-mask.mean())}
    np.savez_compressed(tmp_path/'sintered-geometry.npz',points=p,elements=e)
    write_model(tmp_path,c,p,e,idx,estimate)
    r=run(tmp_path,60)
    assert all(r['checks'].values()),r
    assert r['max_native_green_strain_error']<1e-7
    if void and control=='displacement':
        assert r['max_omitting_modes_green_error']>.001
        assert r['max_mode_displacement_mm']>1e-4
    if control=='force':
        assert r['peak_nominal_compression_MPa']==pytest.approx(.001,rel=1e-5)
        assert abs(r['residual_compression_strain'])<1e-7
        assert r['peak_peeq']==0
    elif not void:
        assert r['residual_compression_strain']==pytest.approx(.02465,rel=.01)
        assert r['peak_cauchy_compression_MPa']==pytest.approx(.05496,rel=.01)
    # Ensure restart extraction does not merely pick a similarly sized vector.
    _,s=next(read_dat(tmp_path/'plastic.dat'))
    du=find(s,'displacements','PHYSICAL');du=du[np.argsort(du[:,0])][:,1:4]
    dm=find(s,'displacements','MACRO');dm=dm[np.argsort(dm[:,0])][:,1:4]
    group=next(read_restart_groups(tmp_path/'plastic.rout'))
    du[0,0]+=.001
    with pytest.raises(ValueError,match='disagrees'):restart_state(group,p,e,du,dm)
    with pytest.raises(ValueError,match='Internal-mode'):audit(tmp_path,tmp_path/'must-not-exist')
    assert not (tmp_path/'must-not-exist').exists()
    with pytest.raises(ValueError,match='incompatible'):plastic.analyze(tmp_path,True,0)


def test_restart_parser_refuses_unverified_version():
    with pytest.raises(ValueError,match='Unsupported restart version'):
        restart_state([b'Version 2.21'],np.zeros((8,3)),np.arange(8)[None,:],np.zeros((8,3)),np.zeros((6,3)))
