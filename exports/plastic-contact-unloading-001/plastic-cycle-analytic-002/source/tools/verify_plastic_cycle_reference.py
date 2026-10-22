"""Independent coaxial Hencky J2 pressure-cycle reference, no contact.

Uniaxial Cauchy pressure from zero through a single maximum and back to zero.
Hardening must be positive; this reference is not a calibrated material law.
"""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
from scipy.special import lambertw
from packing_fem import sintered
from packing_fem.explicit_material import hex_to_tet
from packing_fem.explicit_periodic import deformation
from packing_fem.explicit_solver import solve


def plastic_reference(pressure,young,poisson,yield_stress,hardening):
    p=np.asarray(pressure,float)
    if p.ndim!=1 or not len(p) or not np.all(np.isfinite(p)) or np.any(p<0) or young<=0 or not -1<poisson<.5 or yield_stress<=0 or hardening<=0:raise ValueError('Invalid uniaxial hardening reference input')
    a=1-2*poisson;s=young*lambertw(a*p/young).real/a
    ep=np.maximum(np.maximum.accumulate(s)-yield_stress,0.)/hardening
    ez=-s/young-ep;el=poisson*s/young+.5*ep
    return dict(kirchhoff_compression=s,peeq=ep,axial_compression=-np.expm1(ez),axial_stretch=np.exp(ez),lateral_stretch=np.exp(el),volume_ratio=np.exp(-a*s/young),specific_total_energy_MPa=.5*s*s/young+yield_stress*ep+.5*hardening*ep*ep)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--reuse-unload8',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    base=json.loads(Path('configs/explicit-area-cauchy-pressure.json').read_text());base.update(length_mm=1.,pressure_mpa=.1,ramp_s=.032,hold_s=.002,unload_s=.008,recovery_hold_s=.002,max_steps=300000,output_every=200,usage_scope='homogeneous_1mm_plastic_cycle_no_contact_not_particle_packing')
    cases=[('unload8',{}),('unload16',dict(unload_s=.016)),('unload8-half-cfl',dict(cfl=.1))]
    plan=dict(config=base,cases=cases,reference='coaxial pure uniaxial Kirchhoff Hencky J2 with isotropic hardening and free transverse expansion',tolerances=dict(axial_path_error_over_peak=.01,final_residual_error_over_reference=.01,final_peeq_relative=.01,final_stress_over_peak=.02,final_volume_error=.000001),solver_acceptance_changed=False)
    (a.output/'verification-plan.json').write_text(json.dumps(plan,indent=2)+'\n');hashes={}
    for path in [Path(__file__),*Path('packing_fem').glob('*.py')]:
        relative=path.resolve().relative_to(Path.cwd());target=a.output/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);hashes[str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest()
    (a.output/'source-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    X,h,_=sintered.mesh(np.ones((1,1,1),bool),1.);t=hex_to_tet(X,h);summaries=[]
    exact=plastic_reference(np.array([0.,.1,0.]),base['young_mpa'],base['poisson'],base['yield_mpa'],base['hardening_mpa']);peak=float(exact['axial_compression'][1]);residual=float(exact['axial_compression'][-1]);epref=float(exact['peeq'][-1])
    for name,overrides in cases:
        c=dict(base,**overrides);start=time.perf_counter();root=a.output/name
        if name=='unload8' and a.reuse_unload8 is not None:
            previous=json.loads((a.reuse_unload8/'config.json').read_text())
            if previous!=c:raise ValueError('Reused case config mismatch')
            for source in Path('packing_fem').glob('*.py'):
                archived=a.reuse_unload8.parent/'source'/source
                if archived.read_bytes()!=source.read_bytes():raise ValueError('Reused solver source mismatch')
            r=json.loads((a.reuse_unload8/'summary.json').read_text())
            if r['failure'] is not None or not r['checks']['unload_completed']:raise ValueError('Cannot reuse incomplete cycle')
            shutil.copytree(a.reuse_unload8,root)
            (a.output/'reused-case.json').write_text(json.dumps(dict(source=str(a.reuse_unload8),reason='Completed FEM cycle retained after driver-only list-to-array postprocessing error; config and solver sources matched byte-for-byte.'),indent=2)+'\n')
        else:r=solve(root,X,t,c)
        b=np.loadtxt(root/'accepted-step-audit.csv.gz',delimiter=',',skiprows=1);ref=plastic_reference(b[:,3],c['young_mpa'],c['poisson'],c['yield_mpa'],c['hardening_mpa']);err=abs(b[:,1]-ref['axial_compression'])/peak
        with np.load(root/'last-state.npz') as z:F=deformation(z['generalized_position'][-6:]);ep=z['peeq'];final_ep_error=float(np.max(abs(ep-epref))/epref)
        error_residual=abs(r['final_axial_compression']-residual)/residual;stress_error=float(np.max(abs(np.asarray(r['final_work_equivalent_cauchy_stress_MPa'])))/.1);volume_error=abs(float(np.linalg.det(F))-1.)
        checks=dict(path_error=float(err.max())<.01,residual_error=error_residual<.01,peeq_error=final_ep_error<.01,unloaded_stress=stress_error<.02,isochoric_residual_volume=volume_error<1e-6,path_completed=r['failure'] is None and r['checks']['unload_completed'])
        summary=dict(case=name,config=c,elapsed_s=time.perf_counter()-start,solver_checks=r['checks'],reference_checks={k:bool(v) for k,v in checks.items()},max_axial_error_over_peak=float(err.max()),final_residual_compression=r['final_axial_compression'],reference_residual_compression=residual,final_peeq_relative_error=final_ep_error,final_stress_over_peak=stress_error,final_volume_error=volume_error,failure=r['failure'])
        np.savetxt(root/'analytic-comparison.csv.gz',np.c_[b[:,0],b[:,3],b[:,1],ref['axial_compression'],err,ref['peeq']],delimiter=',',comments='',header='time_s,pressure_MPa,FEM_axial_compression,analytic_quasistatic_axial_compression,error_over_analytic_peak,analytic_peeq')
        (root/'reference-audit.json').write_text(json.dumps(summary,indent=2)+'\n');summaries.append(summary);print(json.dumps(summary),flush=True)
    passed=all(all(s['solver_checks'].values()) and all(s['reference_checks'].values()) for s in summaries)
    (a.output/'summary.json').write_text(json.dumps(dict(all_solver_and_reference_checks_passed=passed,reference_peak_compression=peak,reference_residual_compression=residual,reference_residual_peeq=epref,cases=summaries,contact_validated=False,particle_compaction_validated=False),indent=2)+'\n');raise SystemExit(0 if passed else 2)

if __name__=='__main__':main()
