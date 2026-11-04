"""Read-only forecasts from a saved accepted state; no bulk step is accepted."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from packing_fem.checkpoint import load,restore_contact
from packing_fem.embedded_contact import EmbeddedContact
from packing_fem.explicit_periodic import Periodic,deformation
from packing_fem.explicit_solver import stable_dt
from packing_fem.porosity_cycle import PorosityCycle
from packing_fem.runtime_quadrature import repair


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    hashes=lambda:{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.checkpoint.iterdir() if p.is_file()}
    original=hashes();s,meta=load(a.checkpoint);c=meta['config'];X=s['X'];L=c['length_mm'];periodic=Periodic(X,L,mass_frame=c.get('mass_frame','pinned'));periodic.mass(s['mass']);H=deformation(s['q'][-6:])*L
    kwargs=dict(thickness=c.get('barrier_thickness_mm',.02),max_candidates=c.get('contact_candidate_budget',500000),search_method=c.get('contact_search_method','aabb'),local_material_filter=c.get('local_material_filter',True),edge_stationarity_filter=c.get('edge_stationarity_filter',False),multiple_patches=c.get('multiple_patches',False),certified_local_filter=c.get('certified_local_filter',False),closest_tie_rule=c.get('closest_tie_rule','stop'),edge_quadrature_relative_tolerance=c.get('edge_quadrature_relative_tolerance',0.),quadrature_mode=c.get('contact_quadrature_mode','nodal'),quadrature_tolerance=c.get('surface_quadrature_tolerance',.001),error_budget=c.get('surface_error_budget','component'),retain_partition=c.get('retain_surface_partition',False),rule_strategy=c.get('surface_rule_strategy','single_pose'),audit_mode='parent',redistribution_factor=c.get('surface_redistribution_factor',2.),lookahead_steps=c.get('atlas_lookahead_steps',[]),atlas_max_entries=c.get('atlas_max_matrix_entries',12000000),atlas_max_points=c.get('atlas_max_candidate_points',80000),absolute_energy_budget=c.get('contact_absolute_energy_budget_Nmm',0.),absolute_force_budget=c.get('contact_absolute_force_budget_N',0.))
    contact=EmbeddedContact(X,s['t'],periodic,c['contact_penalty_mpa_mm'],c['friction'],c['contact_skin_mm'],quadrature_rules=s['runtime_rules'],**kwargs);restore_contact(contact,s['contact'])
    before=contact.replay_force_state(s['x'],periodic.velocities(s['v']),H);f=periodic.force(s['fc'],s['extra']);error=float(np.linalg.norm(periodic.force(before[0],before[1])-f)/max(np.linalg.norm(f),1e-16))
    if error>1e-10:raise ValueError('Frozen forecast source does not reproduce accepted force')
    cycle=PorosityCycle(c,s['porosity_cycle_state']);dt=min(stable_dt(s['x'],s['t'],s['V0'],s['mass'],c,periodic.velocities(s['v']),contact,s['sigma'],s['b']),np.sqrt(.2*c['contact_skin_mm']/max(float(np.max(np.linalg.norm(periodic.velocities(s['a']),axis=1))),1e-30)),cycle.interval_end()-s['time'])
    rows=[]
    for count in (1,8,16,32):
        h=min(count*dt,cycle.interval_end()-s['time']);q=s['q']+h*s['v']+.5*h*h*s['a'];v=s['v']+.5*h*s['a'];pf=cycle.displacement(s['time']+h)
        q[periodic.prescribed]=pf[0];v[periodic.prescribed]=pf[1];G=deformation(q[-6:])*L;Gdot=(deformation(v[-6:])-np.eye(3))*L
        row=dict(accepted_step=s['steps'],horizon=count,forecast_time_s=s['time']+h,passed=False,bulk_step_accepted=False)
        try:
            _,_,report=repair(contact,s['t'],s['x'],periodic.velocities(s['v']),H,periodic.positions(q),periodic.velocities(v),G,Gdot,h,before,prediction_only=True)
            row.update(passed=True,report=report)
        except ValueError as exc:row['reason']=str(exc)
        rows.append(row);print(json.dumps(row),flush=True)
    assert hashes()==original
    (a.output/'summary.json').write_text(json.dumps(dict(scope=__doc__,source_force_error=error,checkpoint_unchanged=True,cases=rows),indent=2)+'\n')


if __name__=='__main__':main()
