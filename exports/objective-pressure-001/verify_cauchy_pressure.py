"""Objective load, manufactured contact balance and actual homogeneous FEM runs.

No particle packing or contact-collapse trajectory is claimed by these tests.
Run from repository root; output directory must not already exist.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from packing_fem.macro_loading import uniaxial_load,recover_cauchy
from packing_fem.explicit_material import internal_force,geometry,hex_to_tet
from packing_fem.explicit_periodic import Periodic,deformation
from packing_fem.explicit_solver import solve
from packing_fem.area_contact import AreaContact
from packing_fem.barrier_contact import barrier
from packing_fem.sintered import mesh
from tools.verify_plane_contact import fixture


def analytic(c,P):
    E,nu,y,H=c['young_mpa'],c['poisson'],c['yield_mpa'],c['hardening_mpa']
    # J depends on elastic volume alone; plastic flow is isochoric.
    J=brentq(lambda J:np.log(J)+(1-2*nu)*P*J/E,.1,1.)
    tau=P*J;ep=max(0.,(tau-y)/H)
    return dict(J=J,tau=tau,ep=ep,axial=np.exp(-tau/E-ep),lateral=np.exp(nu*tau/E+.5*ep))


def equilibrium(n,P,c):
    a=analytic(c,P);lam=a['axial'];lt=a['lateral'];traction=P*lt**2
    gap=brentq(lambda d:barrier(d,c['contact_penalty_mpa_mm'],c['barrier_thickness_mm'])[1]-traction,1e-8,c['barrier_thickness_mm'])
    X,t,_,_,upper=fixture(n,.015,0.);x=X.copy();x[:,:2]*=lt;x[:,2]*=lam;x[upper,2]+=-.2*lam+gap
    F=np.diag([lt,lt,.8*lam+gap]);E,nu=c['young_mpa'],c['poisson']
    b=np.diag(np.exp(np.array([2*nu,2*nu,-2])*a['tau']/E))
    fint,volume,_,J,_,_=internal_force(x,t,np.repeat(b[None],len(t),axis=0),np.full(len(t),a['ep']),c)
    p=Periodic(X,1.);contact=AreaContact(X,t,p,c['contact_penalty_mpa_mm'],c['friction'],c['contact_skin_mm'],thickness=c['barrier_thickness_mm'])
    fc,extra,_,_=contact.evaluate(x,x*0,F,F*0,0.)
    f=p.force(fint+fc,extra);load=uniaxial_load(F,P,1.);f[-6:]+=load
    return dict(refinement=n,cauchy_pressure_MPa=P,porosity=float(1-volume.sum()/np.linalg.det(F)),analytic_porosity=gap/(.8*lam+gap),peeq=a['ep'],gap_mm=gap,
                force_residual_relative=float(np.linalg.norm(f)/traction),contact_force_relative=abs(float(fc[upper,2].sum())-traction)/traction,
                volume_error=float(np.max(abs(volume-geometry(X,t)[0]*J)/geometry(X,t)[0])))


def writecsv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    c=json.loads(Path('configs/explicit-area-cauchy-pressure.json').read_text())
    for source in ('packing_fem/macro_loading.py','packing_fem/explicit_solver.py','packing_fem/area_contact.py',__file__):
        (a.output/Path(source).name).write_bytes(Path(source).read_bytes())
    rows=[equilibrium(n,P,c) for n in (1,2,4) for P in (.02,.1)];writecsv(a.output/'contact-equilibrium.csv',rows)
    checks=dict(manufactured_force_balance=all(r['force_residual_relative']<1e-8 for r in rows),manufactured_contact_force=all(r['contact_force_relative']<1e-8 for r in rows),manufactured_volume=all(r['volume_error']<1e-10 for r in rows),manufactured_porosity=all(abs(r['porosity']-r['analytic_porosity'])<1e-12 for r in rows))
    old=Path('results/area-pressure-layered-002/last-state.npz')
    if old.exists():
        with np.load(old) as data:F=deformation(data['generalized_position'][-6:])
        nominal=np.array([0.,0.,-.02,0.,0.,0.]);corrected=uniaxial_load(F,.02,1.)
        (a.output/'historical-load-diagnostic.json').write_text(json.dumps(dict(source=str(old),scope='same_saved_cell_different_load_definition_not_a_recalculated_solution',F=F.tolist(),old_work_equivalent_cauchy_MPa=recover_cauchy(F,nominal,1.).tolist(),new_work_equivalent_cauchy_MPa=recover_cauchy(F,corrected,1.).tolist()),indent=2)+'\n')
    runs=[]
    for n,cfl,P in ((1,.2,.02),(1,.1,.02),(2,.2,.02),(2,.1,.02),(1,.2,.1),(1,.1,.1)):
        X,h,_=mesh(np.ones((n,n,n),bool),1/n);t=hex_to_tet(X,h)
        config=dict(c,length_mm=1.,pressure_mpa=P,cfl=cfl,ramp_s=.004,hold_s=.002,max_steps=100000,output_every=100,remesh_every=0,usage_scope='homogeneous_1mm_cube_no_pores_or_active_contact_not_particle_packing')
        name=f'cube-n{n}-cfl{cfl}-P{P}';result=solve(a.output/name,X,t,config);ref=analytic(c,P)
        with np.load(a.output/name/'last-state.npz') as state:F=deformation(state['generalized_position'][-6:]);ep=state['peeq'].max()
        stress=np.array(result['final_work_equivalent_cauchy_stress_MPa']);err=float(np.max(abs(stress-np.diag([0.,0.,-P])))/P)
        row=dict(case=name,refinement=n,cfl=cfl,cauchy_pressure_MPa=P,axial_stretch=float(F[2,2]),analytic_axial_stretch=ref['axial'],axial_error=abs(float(F[2,2])-ref['axial']),peeq=float(ep),analytic_peeq=ref['ep'],peeq_error=abs(float(ep)-ref['ep']),shear_norm=float(np.linalg.norm(F[np.triu_indices(3,1)])),stress_relative_error=err,energy_relative_error=result['energy_relative_error'],kinetic_ratio=result['kinetic_internal_ratio_loaded'],solver_checks_passed=all(result['checks'].values()),failure=result['failure'])
        runs.append(row);writecsv(a.output/'transient-cubes.csv',runs);print(json.dumps(row),flush=True)
    checks['actual_solver_checks']=all(r['solver_checks_passed'] for r in runs)
    checks['actual_axial_limit']=all(r['axial_error']<2e-5 for r in runs)
    checks['actual_plastic_limit']=all(r['peeq_error']<2e-5 for r in runs)
    checks['actual_stress_limit']=all(r['stress_relative_error']<1e-4 for r in runs)
    checks['no_shear_clamp_needed_for_homogeneous_cube']=all(r['shear_norm']<1e-8 for r in runs)
    summary=dict(scope='load_operator_and_manufactured_plane_contact_plus_actual_homogeneous_FEM_only',checks=checks,particle_compaction_completed=False,layered_contact_transient_revalidated=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));raise SystemExit(0 if all(checks.values()) else 2)
if __name__=='__main__':main()
