"""A stopped FEM job must retain its checkpoint and report failure honestly."""
import csv,hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
from packing_fem.explicit_material import hex_to_tet


def test_cli_stage_budget_failure_preserves_accepted_state_and_reports_failed(tmp_path):
    root=Path(__file__).resolve().parents[1]
    points=np.array([[i/4,j/2,k/2] for i in range(3) for j in range(3) for k in range(3)])
    def node(i,j,k):return i*9+j*3+k
    cells=np.array([[node(i,j,k),node(i+1,j,k),node(i+1,j+1,k),node(i,j+1,k),node(i,j,k+1),node(i+1,j,k+1),node(i+1,j+1,k+1),node(i,j+1,k+1)] for i in range(2) for j in range(2) for k in range(2)])
    mesh=tmp_path/'mesh.npz';np.savez(mesh,points=points,tets=hex_to_tet(points,cells))
    config=tmp_path/'config.json';config.write_text(json.dumps(dict(length_mm=1.,young_mpa=10.,poisson=.33,yield_mpa=1e6,hardening_mpa=0.,friction=.3,ipc_dhat_mm=.01,ipc_stiffness_mpa=10.,ipc_slip_regularization_mm=1e-4,target_final_porosity=.49,porosity_tolerance=1e-5,maximum_axial_compression=.1,implicit_strain_increment=.01,implicit_minimum_increment=1e-6,implicit_max_stages=1,implicit_max_newton=30,implicit_force_tolerance_N=1e-8,implicit_regularization=1e-8,implicit_macro_tolerance=1e-10,implicit_remesh=False,implicit_optimizer='coupled_newton',transverse_boundary='fixed_cell',loading_only=True)))
    out=tmp_path/'run';command=[sys.executable,str(root/'tools/run_implicit_compaction.py'),'--config',str(config),'--mesh',str(mesh),'--out',str(out)]
    environment=dict(os.environ,PYTHONPATH=str(root),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    result=subprocess.run(command,cwd=root,env=environment,capture_output=True,text=True)
    assert result.returncode!=0,result.stdout+result.stderr
    assert 'Stage budget exhausted' in (out/'failure.txt').read_text()
    status=json.loads((out/'status.json').read_text())
    assert status['status']=='failed' and not status['target_reached']
    with (out/'history.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==1 and status['last_converged_porosity']==float(rows[0]['porosity'])
    checkpoint=out/'state-0001.npz';digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest();saved_status=(out/'status.json').read_bytes()
    retry=subprocess.run(command,cwd=root,env=environment,capture_output=True,text=True)
    assert retry.returncode!=0 and 'FileExistsError' in retry.stderr
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
    assert (out/'status.json').read_bytes()==saved_status
