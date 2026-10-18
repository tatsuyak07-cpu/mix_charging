"""Actual periodic contact-collapse trajectory, with independent endpoint audits.

1 mm layered verification cell, 20% initial pore volume; not particle packing.
Analytic diagonal equilibrium is a branch comparison, not an imposed constraint.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from packing_fem.explicit_solver import solve
from packing_fem.explicit_periodic import deformation
from packing_fem.explicit_material import geometry
from packing_fem.adaptive_tet import closed_surface_volume
from packing_fem.barrier_contact import barrier
from tools.verify_plane_contact import fixture
from tools.verify_cauchy_pressure import analytic


def reference(c):
    a=analytic(c,c['pressure_mpa'])
    traction=c['pressure_mpa']*a['lateral']**2
    gap=brentq(lambda d:barrier(d,c['contact_penalty_mpa_mm'],c['barrier_thickness_mm'])[1]-traction,1e-8,c['barrier_thickness_mm'])
    height=.8*a['axial']+gap
    return dict(material_axial_stretch=a['axial'],material_peeq=a['ep'],gap_mm=gap,cell_height_mm=height,axial_compression=1-height,porosity=gap/height)


def audit(directory,c,result):
    with np.load(directory/'last-state.npz') as z:
        X=z['reference_points'];x=z['points'];t=z['tets'];b=z['elastic_b'];V0=z['reference_volume'];F=deformation(z['generalized_position'][-6:]);ep=z['peeq']
    cell_volume=float(np.linalg.det(F)*c['length_mm']**3)
    geometric_solid=float(geometry(x,t)[0].sum())
    material_solid=float(V0@np.sqrt(np.linalg.det(b)))
    boundary_solid=float(closed_surface_volume(x,t))
    normal=np.cross(F[:,0],F[:,1]);normal/=np.linalg.norm(normal)
    lower=x[np.isclose(X[:,2],.4)];upper=x[np.isclose(X[:,2],.6)]
    gap=float((upper.mean(axis=0)-lower.mean(axis=0))@normal)
    planarity=float(max(np.abs((lower-lower.mean(axis=0))@normal).max(),np.abs((upper-upper.mean(axis=0))@normal).max()))
    sigma=np.asarray(result['final_work_equivalent_cauchy_stress_MPa'])
    stress_error=float(np.max(abs(sigma-np.diag([0.,0.,-c['pressure_mpa']])))/c['pressure_mpa'])
    ref=reference(c);phi=1-geometric_solid/cell_volume
    # Criteria are fixed before calculation; no failure or branch mismatch is hidden.
    checks=dict(independent_volume=abs(geometric_solid-material_solid)/geometric_solid<1e-8 and abs(geometric_solid-boundary_solid)/geometric_solid<1e-10,
                endpoint_uniaxial_stress=stress_error<.02,positive_plane_gap=gap>0,plane_planarity=planarity<1e-8,
                active_contact_observed=False,
                diagonal_branch_axial=abs(result['final_axial_compression']-ref['axial_compression'])<1e-4,
                diagonal_branch_porosity=abs(phi-ref['porosity'])<1e-5,
                diagonal_branch_macro_shear=float(np.linalg.norm(F[np.triu_indices(3,1)]))<1e-8)
    h=np.genfromtxt(directory/'history.csv',delimiter=',',names=True);h=np.atleast_1d(h)
    checks['active_contact_observed']=bool(np.max(h['contacts'])>0)
    return dict(reference_assumptions='homogeneous_diagonal_equilibrium_branch_not_a_uniqueness_claim',reference=ref,checks={k:bool(v) for k,v in checks.items()},
                independent_geometric_porosity=phi,independent_material_porosity=1-material_solid/cell_volume,independent_boundary_porosity=1-boundary_solid/cell_volume,
                current_gap_mm=gap,plane_planarity_error_mm=planarity,cell_F=F.tolist(),macro_shear_norm=float(np.linalg.norm(F[np.triu_indices(3,1)])),
                work_equivalent_cauchy_MPa=sigma.tolist(),stress_relative_error=stress_error,peak_peeq=float(ep.max()),
                axial_branch_error=abs(result['final_axial_compression']-ref['axial_compression']),porosity_branch_error=abs(phi-ref['porosity']),
                scope='actual_layered_contact_transient_not_particle_packing')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--cfl',type=float,default=.2);p.add_argument('--ramp-seconds',type=float,default=.004);p.add_argument('--hold-seconds',type=float,default=.002);p.add_argument('--pressure',type=float,default=.02);p.add_argument('--refinement',type=int,default=1);p.add_argument('--max-steps',type=int,default=200000);p.add_argument('--mass-frame',choices=['pinned','zero_momentum'],default='pinned');a=p.parse_args()
    c=json.loads(Path('configs/explicit-area-cauchy-pressure.json').read_text());c.update(length_mm=1.,ramp_s=a.ramp_seconds,hold_s=a.hold_seconds,pressure_mpa=a.pressure,cfl=a.cfl,max_steps=a.max_steps,output_every=200,verification_refinement=a.refinement,mass_frame=a.mass_frame,usage_scope='1mm_layered_plane_pore_20percent_not_particle_packing')
    a.output.mkdir(parents=True,exist_ok=False)
    sources={}
    source_dir=a.output/'source';source_dir.mkdir()
    for f in [Path(__file__),Path('tools/verify_cauchy_pressure.py'),Path('tools/verify_plane_contact.py'),*Path('packing_fem').glob('*.py')]:
        data=f.read_bytes();sources[str(f)]=hashlib.sha256(data).hexdigest();dst=source_dir/f;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(data)
    (a.output/'source-sha256.json').write_text(json.dumps(sources,indent=2)+'\n')
    (a.output/'verification-plan.json').write_text(json.dumps(dict(config=c,reason='verify objective load on an actual contact-collapse path; rate and CFL comparisons keep material/contact tolerances unchanged',solver_criteria='unchanged config tolerances',endpoint_stress_relative_tolerance=.02,branch_axial_tolerance=1e-4,branch_porosity_tolerance=1e-5,branch_shear_tolerance=1e-8,particle_compaction=False),indent=2)+'\n')
    X,t,_,_,_=fixture(a.refinement,.015,0.)
    start=time.perf_counter();r=solve(a.output/'run',X,t,c);comparison=audit(a.output/'run',c,r)
    (a.output/'endpoint-audit.json').write_text(json.dumps(comparison,indent=2)+'\n')
    physical_keys=('independent_volume','endpoint_uniaxial_stress','positive_plane_gap','plane_planarity','active_contact_observed')
    summary=dict(scope=c['usage_scope'],elapsed_s=time.perf_counter()-start,solver_checks=r['checks'],independent_checks=comparison['checks'],
                 solver_and_endpoint_checks_passed=all(r['checks'].values()) and all(comparison['checks'][k] for k in physical_keys),
                 diagonal_branch_checks_passed=all(comparison['checks'][k] for k in comparison['checks'] if k.startswith('diagonal_branch')),
                 failure=r['failure'],pressure_mpa=a.pressure,ramp_s=a.ramp_seconds,hold_s=a.hold_seconds,cfl=a.cfl,refinement=a.refinement,mass_frame=a.mass_frame,particle_compaction_completed=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
    raise SystemExit(0 if summary['solver_and_endpoint_checks_passed'] and summary['diagonal_branch_checks_passed'] else 2)
if __name__=='__main__':main()
