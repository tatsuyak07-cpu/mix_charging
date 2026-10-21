"""Actual Cauchy compression of the existing connected 4 mm pore fixture.

Initial porosity is 0.25%, not the requested 40% particle packing.
All-step audits distinguish path monitoring from endpoint equilibrium.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from tools.run_barrier_compaction import fixture
from packing_fem.explicit_solver import solve
from packing_fem.explicit_periodic import deformation
from packing_fem.explicit_material import geometry
from packing_fem.adaptive_tet import closed_surface_volume


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--pressure',type=float,default=.02);p.add_argument('--ramp-seconds',type=float,default=.004);p.add_argument('--cfl',type=float,default=.2);p.add_argument('--max-steps',type=int,default=200000);p.add_argument('--unload-seconds',type=float,default=0.);p.add_argument('--recovery-seconds',type=float,default=0.);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    c=json.loads(Path('configs/explicit-area-cauchy-pressure.json').read_text());c.update(pressure_mpa=a.pressure,ramp_s=a.ramp_seconds,hold_s=.002,unload_s=a.unload_seconds,recovery_hold_s=a.recovery_seconds,cfl=a.cfl,max_steps=a.max_steps,output_every=200,mass_frame='pinned',usage_scope='connected_4mm_fixture_initial_porosity_0.25percent_not_particle_packing')
    hashes={}
    for path in [Path(__file__),Path('tools/run_barrier_compaction.py'),*Path('packing_fem').glob('*.py')]:
        relative=path.resolve().relative_to(Path.cwd());target=a.output/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True);data=path.read_bytes();target.write_bytes(data);hashes[str(relative)]=hashlib.sha256(data).hexdigest()
    (a.output/'source-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    (a.output/'verification-plan.json').write_text(json.dumps(dict(config=c,reason='Initial relaxed stiffness already positive; test actual connected-cell pressure response without changing material/contact settings. Compare CFL and rate separately. Optional unloading runs continuously from rest, preserving material and friction history; compare zero final stress using peak pressure as scale.',endpoint_stress_relative_tolerance=.02,volume_relative_tolerance=1e-8,all_step_audit=True,particle_compaction=False),indent=2)+'\n')
    X,t=fixture();start=time.perf_counter();result=solve(a.output/'run',X,t,c)
    with np.load(a.output/'run/last-state.npz') as z:
        x=z['points'];t=z['tets'];V0=z['reference_volume'];b=z['elastic_b'];F=deformation(z['generalized_position'][-6:]);ep=z['peeq']
    volume=float(geometry(x,t)[0].sum());material=float(V0@np.sqrt(np.linalg.det(b)));boundary=closed_surface_volume(x,t);cell=float(np.linalg.det(F)*4**3)
    stress=np.asarray(result['final_work_equivalent_cauchy_stress_MPa']);target_pressure=0. if a.unload_seconds>0 else a.pressure;stress_error=float(np.max(abs(stress-np.diag([0.,0.,-target_pressure])))/a.pressure)
    independent=dict(geometry_vs_material_volume=abs(volume-material)/volume<1e-8,geometry_vs_boundary_volume=abs(volume-boundary)/volume<1e-10,endpoint_uniaxial_stress=stress_error<.02)
    audit=dict(checks=independent,geometric_porosity=1-volume/cell,material_porosity=1-material/cell,boundary_porosity=1-boundary/cell,stress_relative_error=stress_error,target_final_pressure_mpa=target_pressure,stress_normalization_pressure_mpa=a.pressure,work_equivalent_cauchy_stress_MPa=stress.tolist(),macro_F=F.tolist(),macro_shear_norm=float(np.linalg.norm(F[np.triu_indices(3,1)])),peak_peeq=float(ep.max()),scope='independent_volume_plus_comparison_of_solver_stress_not_independent_reassembly_of_friction')
    (a.output/'endpoint-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    summary=dict(pressure_mpa=a.pressure,ramp_s=a.ramp_seconds,unload_s=a.unload_seconds,recovery_hold_s=a.recovery_seconds,cfl=a.cfl,elapsed_s=time.perf_counter()-start,solver_checks=result['checks'],endpoint_checks=independent,solver_and_endpoint_checks_passed=all(result['checks'].values()) and all(independent.values()),failure=result['failure'],particle_compaction_completed=False,scope=c['usage_scope'])
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True);raise SystemExit(0 if summary['solver_and_endpoint_checks_passed'] else 2)
if __name__=='__main__':main()
