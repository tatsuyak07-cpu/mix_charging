"""Independent checks of cached geometric inverses and unchanged timestep bounds."""
import argparse,importlib.util,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from packing_fem.explicit_solver import solve,lumped_mass
from packing_fem.explicit_material import geometry
from packing_fem.explicit_periodic import Periodic
from packing_fem.sintered import mesh
from packing_fem.explicit_material import hex_to_tet
from packing_fem.contact_audit import plain


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('packing_fem.reference_solver',a.reference_source);reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    X,h,_=mesh(np.ones((2,2,2),bool),.5);t=hex_to_tet(X,h)
    c=json.loads(Path('configs/explicit-prototype.json').read_text())
    c.update(length_mm=1.,contact_backend='embedded_area',loading_mode='cauchy_pressure',pressure_mpa=.01,yield_mpa=1e6,ramp_s=.0006,hold_s=.0003,damping_per_s=20000.,remesh_every=0,output_every=50,element_backend='native',native_library='/workspace/environment/mix_charging/native/element-a0cdacad3ca39dc241d1f99227ca085138325debf35b9850ab19ee1f8ebcad9c.so')
    results={};start=time.perf_counter()
    for name,solver,config in [('reference',reference.solve,c),('cached',solve,c),('half-step',solve,dict(c,cfl=c['cfl']/2))]:
        results[name]=solver(a.output/name,X,t,config)
    old=np.load(a.output/'reference/last-state.npz');new=np.load(a.output/'cached/last-state.npz');half=np.load(a.output/'half-step/last-state.npz')
    periodic=Periodic(X,1.);periodic.prescribed_active=False;periodic.free=np.arange(periodic.ndof);periodic.mass(lumped_mass(t,geometry(X,t)[0],c['density_tonne_mm3'],len(X)))
    dv=new['generalized_velocity']-old['generalized_velocity'];delta_ke=float(.5*dv@(periodic.M@dv))
    work=max(abs(results['reference']['energy_scale_Nmm']),c['young_mpa']*1e-12)
    axial_strain=brentq(lambda e:-c['young_mpa']*e*np.exp(-(1-2*c['poisson'])*e)-c['pressure_mpa'],-.1,0.)
    expected=1-np.exp(axial_strain)
    strain_error=abs(results['cached']['final_axial_compression']-expected)/expected
    timestep_change=abs(results['cached']['final_axial_compression']-results['half-step']['final_axial_compression'])/expected
    zero=dict(c,loading_mode='displacement',compression_fraction=0.,ramp_s=.0001,hold_s=.0001)
    zr=solve(a.output/'zero-load',X,t,zero)
    z=np.load(a.output/'zero-load/last-state.npz');zero_q=float(abs(z['generalized_position']).max())
    stages=dict(c,pressure_mpa=.1,save_checkpoint=True,pressure_stages=[dict(pressure_mpa=P,ramp_s=.0006,hold_s=.0003,max_hold_s=.002) for P in (.001,.01,.1)])
    sr=solve(a.output/'staged-control',X,t,stages)
    checks=dict(state_geometry=float(abs(new['points']-old['points']).max())<1e-10,
                state_tensor=float(abs(new['elastic_b']-old['elastic_b']).max())<1e-10,
                velocity_perturbation_energy=delta_ke/work<1e-10,
                hencky_analytic=strain_error<.02,timestep_sensitivity=timestep_change<1e-3,
                zero_load=zero_q<1e-10,
                staged_0_1_MPa=sr['checks'].get('requested_pressure_reached',False) and all(s['passed'] for s in sr['pressure_stages']),
                reference_path=results['reference']['checks']['path_completed'],cached_path=results['cached']['checks']['path_completed'])
    summary=plain(dict(scope=__doc__,passed=all(checks.values()),checks=checks,
                       geometry_difference_mm=float(abs(new['points']-old['points']).max()),
                       tensor_difference=float(abs(new['elastic_b']-old['elastic_b']).max()),
                       generalized_velocity_absolute_difference_mixed_units=float(abs(dv).max()),
                       velocity_perturbation_energy_Nmm=delta_ke,velocity_perturbation_energy_relative=delta_ke/work,
                       analytic_strain_relative_error=strain_error,timestep_strain_relative_change=timestep_change,
                       zero_load_generalized_displacement=zero_q,stage_records=sr['pressure_stages'],wall_s=time.perf_counter()-start,
                       limits=['Elastic solid cube control, not porous packing prediction.',
                               'No single absolute tolerance for mixed mm/s and 1/s velocity entries.',
                               'Initial 100-step uniform-array-absolute-tolerance failure retained separately.']))
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(a.output/'reference-solver.py').write_bytes(a.reference_source.read_bytes());print(json.dumps(summary),flush=True)
    if not summary['passed']:raise SystemExit(1)


if __name__=='__main__':main()
