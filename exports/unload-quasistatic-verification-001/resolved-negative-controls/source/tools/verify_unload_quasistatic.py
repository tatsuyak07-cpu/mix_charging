"""Independent elastic reference and sensitivity tests for true-pressure unloading.

This is a homogeneous 1 mm cube control, not the porous particle packing.
Existing solver acceptance is never changed by these supplementary diagnostics.
"""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
from scipy.special import lambertw
from packing_fem import sintered
from packing_fem.explicit_material import hex_to_tet
from packing_fem.explicit_periodic import deformation
from packing_fem.explicit_solver import solve


def elastic_reference(pressure,young,poisson,volume):
    """Pure Cauchy compression, free lateral faces: E e_z = -p exp((1-2nu)e_z).

    Principal Lambert W gives the continuous elastic branch at p=0.
    This reference uses no FEM stress/force or material update routine.
    """
    p=np.asarray(pressure,float)
    if young<=0 or not -1<poisson<.5 or np.any(p<0):raise ValueError('Invalid elastic reference input')
    a=1-2*poisson;e=-lambertw(a*p/young).real/a
    return dict(axial_compression=-np.expm1(e),axial_log_strain=e,lateral_stretch=np.exp(-poisson*e),energy_Nmm=.5*young*volume*e**2)


def cycle_metrics(rows,ramp,hold,unload,reference_energy):
    """Report all rows, including zero pressure, without a vanishing denominator.

    A small global KE ratio alone cannot certify local/contact equilibrium.
    Reference energy must be supplied independently and remain positive.
    """
    a=np.asarray(rows,float)
    if reference_energy<=0 or not np.isfinite(reference_energy) or a.ndim!=2 or a.shape[1]!=11 or not np.all(np.isfinite(a)):raise ValueError('Invalid cycle audit input')
    change=abs(a[:,5]-a[0,5]);legacy=a[:,4]/np.maximum(change,reference_energy*1e-8)
    release=ramp+hold;end=release+unload
    report=dict(normalization='fixed reference energy; no near-zero pressure exclusion',reference_energy_Nmm=float(reference_energy),rows=len(a),peak_kinetic_over_reference=float(np.max(a[:,4])/reference_energy),late_instantaneous_ratio=float(np.max(legacy[a[:,0]>.5*ramp])) if np.any(a[:,0]>.5*ramp) else None)
    for name,mask in [('loading',a[:,0]<=ramp),('unloading',(a[:,0]>=release)&(a[:,0]<=end)),('recovery',a[:,0]>=end)]:
        report[name]=None if not np.any(mask) else dict(peak_kinetic_Nmm=float(np.max(a[mask,4])),peak_kinetic_over_reference=float(np.max(a[mask,4])/reference_energy))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--resolved-negative-controls',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    base=json.loads(Path('configs/explicit-area-cauchy-pressure.json').read_text());base.update(length_mm=1.,pressure_mpa=.02,ramp_s=.008,hold_s=.002,unload_s=.008,recovery_hold_s=.002,output_every=100,max_steps=200000)
    cases=[('baseline',{}),('half-cfl',dict(cfl=.1)),('slow-unload',dict(unload_s=.016)),('rapid-negative-control',dict(ramp_s=.000002,hold_s=0.,unload_s=.000002,recovery_hold_s=.00001))]
    if a.resolved_negative_controls:
        cases=[('resolved-rapid',dict(ramp_s=.000002,hold_s=.000001,unload_s=.000002,recovery_hold_s=.00001,cfl=.02)),('resolved-rapid-half-cfl',dict(ramp_s=.000002,hold_s=.000001,unload_s=.000002,recovery_hold_s=.00001,cfl=.01))]
    plan=dict(scope='homogeneous_1mm_elastic_control_not_porous_packing',config=base,cases=[dict(name=n,overrides=d) for n,d in cases],supplementary_tolerances=dict(peak_KE_over_analytic_peak_energy=.01,maximum_axial_error_over_analytic_peak_strain=.01,final_F_absolute=1e-7),reason='Validate fixed reference-energy diagnostics independently of vanishing instantaneous stored energy; test CFL and unloading rate separately; rapid negative control must be rejected.',solver_acceptance_changed=False,followup_reason='Original rapid control had only 18 steps, missed exact peak pressure and failed energy balance. Add a 1 microsecond pressure plateau and CFL 0.02/0.01 to separate physical inertia from temporal error.' if a.resolved_negative_controls else None)
    (a.output/'verification-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    hashes={}
    for path in [Path(__file__),*Path('packing_fem').glob('*.py')]:
        relative=path.resolve().relative_to(Path.cwd());target=a.output/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);hashes[str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest()
    (a.output/'source-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    X,h,_=sintered.mesh(np.ones((1,1,1),bool),1.);t=hex_to_tet(X,h);summaries=[]
    peak=elastic_reference(.02,base['young_mpa'],base['poisson'],1.);energy=float(peak['energy_Nmm']);strain=float(peak['axial_compression'])
    for name,overrides in cases:
        c=dict(base,**overrides);directory=a.output/name;start=time.perf_counter();r=solve(directory,X,t,c);rows=np.loadtxt(directory/'accepted-step-audit.csv.gz',delimiter=',',skiprows=1);reference=elastic_reference(rows[:,3],c['young_mpa'],c['poisson'],1.)
        error=abs(rows[:,1]-reference['axial_compression'])/strain;unloading=(rows[:,0]>=c['ramp_s']+c['hold_s'])&(rows[:,0]<=c['ramp_s']+c['hold_s']+c['unload_s']);metrics=cycle_metrics(rows,c['ramp_s'],c['hold_s'],c['unload_s'],energy)
        with np.load(directory/'last-state.npz') as z:F=deformation(z['generalized_position'][-6:]);peak_ep=float(z['peeq'].max())
        supplementary=dict(path_completed=r['failure'] is None and r['checks']['unload_completed'],peak_KE_small=metrics['peak_kinetic_over_reference']<.01,tracks_quasistatic_solution=float(error.max())<.01,returns_to_identity=float(np.max(abs(F-np.eye(3))))<1e-7,elastic=peak_ep<1e-10)
        summary=dict(name=name,config=c,elapsed_s=time.perf_counter()-start,metrics=metrics,max_axial_error_over_peak=float(error.max()),unload_max_axial_error_over_peak=float(np.max(error[unloading])) if np.any(unloading) else None,final_F=F.tolist(),peak_final_peeq=peak_ep,solver_checks=r['checks'],supplementary_checks={k:bool(v) for k,v in supplementary.items()},supplementary_pass=bool(all(supplementary.values())),negative_control=('rapid' in name),failure=r['failure'])
        np.savetxt(directory/'analytic-comparison.csv.gz',np.c_[rows[:,0],rows[:,3],rows[:,1],reference['axial_compression'],error,rows[:,4]/energy],delimiter=',',comments='',header='time_s,pressure_MPa,FEM_axial_compression,analytic_quasistatic_axial_compression,error_over_analytic_peak,KE_over_analytic_peak_energy')
        (directory/'reference-audit.json').write_text(json.dumps(summary,indent=2)+'\n');summaries.append(summary);print(json.dumps(summary),flush=True)
    passed=all(s['failure'] is None and s['supplementary_checks']['path_completed'] for s in summaries) and all(s['supplementary_pass'] for s in summaries if not s['negative_control']) and all(not s['supplementary_checks']['peak_KE_small'] or not s['supplementary_checks']['tracks_quasistatic_solution'] for s in summaries if s['negative_control'])
    (a.output/'summary.json').write_text(json.dumps(dict(benchmark_expectations_met=passed,analytic_peak_energy_Nmm=energy,analytic_peak_compression=strain,cases=summaries,solver_acceptance_changed=False,particle_compaction_verified=False),indent=2)+'\n');raise SystemExit(0 if passed else 2)

if __name__=='__main__':main()
