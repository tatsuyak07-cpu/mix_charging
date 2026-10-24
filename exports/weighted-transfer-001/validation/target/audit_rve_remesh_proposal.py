"""Replay one rejected large-cell remesh transaction without advancing time."""
import argparse,json
from pathlib import Path
import numpy as np
from packing_fem.explicit_periodic import Periodic,deformation
from packing_fem.explicit_solver import lumped_mass
from packing_fem.area_contact import AreaContact
from packing_fem.surface_remesh import bisect,transfer_contact,audit
from packing_fem.contact_audit import plain
from packing_fem.adaptive_tet import quality


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--edge',type=int,nargs=2);parser.add_argument('--fraction',type=float,default=.5);a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    c=json.loads((a.source/'config.json').read_text());capture=json.loads((a.source/'contact-audit.json').read_text());files=sorted(a.source.glob('surface-remesh-attempt-*.npz'))
    with np.load(a.source/'last-state.npz') as z:old={k:z[k] for k in z.files}
    if a.edge is None:
        with np.load(files[-1]) as z:p={k:z[k] for k in z.files}
        p={**p,'X':p['reference_points'],'x':p['points'],'t':p['tets'],'b':p['elastic_b'],'ep':p['peeq'],'V0':p['reference_volume']}
    else:
        initial=Periodic(old['reference_points'],c['length_mm'])
        p=bisect(old['reference_points'],old['points'],old['tets'],old['elastic_b'],old['peeq'],old['reference_volume'],initial.velocities(old['generalized_velocity']),a.edge,c['length_mm'],fraction=a.fraction)
        np.savez_compressed(a.output/'proposal.npz',**{k:v for k,v in p.items() if k!='report'})
    def periodic(X,t,V):
        obj=Periodic(X,c['length_mm']);obj.prescribed_active=False;obj.free=np.arange(obj.ndof);m=lumped_mass(t,V,c['density_tonne_mm3'],len(X));obj.mass(m);return obj,m
    p0,m0=periodic(old['reference_points'],old['tets'],old['reference_volume']);p1,m1=periodic(p['X'],p['t'],p['V0']);q=p1.remap_positions(p['x'],old['generalized_position'][-6:]);oldv=p0.velocities(old['generalized_velocity']);weights=p.get('weights',np.full((len(p['supports']),2),.5));p['velocity']=weights[:,0,None]*oldv[p['supports'][:,0]]+weights[:,1,None]*oldv[p['supports'][:,1]];v=p1.remap_velocity(p1.force(m1[:,None]*p['velocity']),0.);p['velocity']=p1.velocities(v)
    def contact(X,t,bc):return AreaContact(X,t,bc,c['contact_penalty_mpa_mm'],c['friction'],c['contact_skin_mm'],thickness=c['barrier_thickness_mm'],max_candidates=c['contact_candidate_budget'],multiple_patches=c['multiple_patches'],edge_stationarity_filter=c['edge_stationarity_filter'],closest_tie_rule=c['closest_tie_rule'])
    oldcontact=contact(old['reference_points'],old['tets'],p0);newcontact=contact(p['X'],p['t'],p1)
    oldcontact.history={tuple(item['key']):{k:np.array(v) if isinstance(v,list) else v for k,v in item['state'].items()} for item in capture['committed_history']}
    h=np.loadtxt(a.source/'history.csv',delimiter=',',skiprows=1)[-1];before=dict(normal_energy=h[10],tangent_energy=h[11],max_coulomb_ratio=h[17]);result=dict(scope=__doc__,source_proposal=str(files[-1]) if a.edge is None else 'fresh_weighted_proposal',edge=a.edge,fraction=a.fraction,geometry_transfer_error_mm=float(np.max(abs(p1.positions(q)-p['x']))),minimum_quality=float(quality(p['x'],p['t']).min()),time_advanced=False,physical_solution_verified=False)
    try:
        after=transfer_contact(oldcontact,newcontact,p,deformation(q[-6:])*c['length_mm'])
        report=audit((old['reference_points'],old['points'],old['tets'],old['elastic_b'],old['peeq'],old['reference_volume'],oldv),p,c,before,after[-1],c['remesh_energy_tolerance'],old_response=(np.array(capture['accepted']['force']),np.array(capture['accepted']['extra'])),new_response=after)
        result.update(transaction_audit=report,accepted=report['accepted'],old_active_histories=len(oldcontact.history),new_active_histories=len(newcontact.history),normal_energy_before=before['normal_energy'],normal_energy_after=after[-1]['normal_energy'],tangent_energy_before=before['tangent_energy'],tangent_energy_after=after[-1]['tangent_energy'])
    except ValueError as exc:result.update(accepted=False,rejection=str(exc))
    (a.output/'summary.json').write_text(json.dumps(plain(result),indent=2)+'\n');(a.output/'audit_rve_remesh_proposal.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(plain(result),indent=2))


if __name__=='__main__':main()
