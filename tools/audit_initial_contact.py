"""Measure original-law contact preload at the stress-free CAD starting geometry."""
import argparse,csv,gc,hashlib,json
from pathlib import Path
import numpy as np
from packing_fem.rigid_inclusion import make_periodic,make_contact


def audit(study,out):
    study=Path(study);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    cases=json.loads((study/'assessment.json').read_text())['cases'];rows=[]
    for case in cases:
        run=Path(case['run']);c=json.loads((run/'config.json').read_text());path=run/'state-0000.npz'
        with np.load(path) as state:X=state['X'];t=state['t'];q=state['q'];ep=state['ep'];b=state['b']
        if np.any(q!=0) or np.any(ep!=0) or not np.array_equal(b,np.broadcast_to(np.eye(3),b.shape)):
            raise ValueError('Expected original stress-free material geometry without prestrain')
        p=make_periodic(X,c['length_mm'],c);contact=make_contact(X,t,p,c)
        value,g,_,count=contact.evaluate_coupled(q,q)
        force,torque=contact.rigid_resultant(q)
        rows.append(dict(level=case['level'],tetrahedra=len(t),contact_stencils=count,
            contact_potential_Nmm=float(value),maximum_free_contact_force_N=float(abs(g[:p.nmicro]).max()),
            contact_force_l2_N=float(np.linalg.norm(g[:p.nmicro])),
            initial_contact_macro_pressure_MPa=float(-g[p.prescribed]/p.L**3),
            initial_to_loaded_pressure_ratio=float((-g[p.prescribed]/p.L**3)/case['pressure_MPa']),
            rigid_force_max_N=float(abs(force).max()),rigid_torque_over_radius_max_N=float(abs(torque).max()/p.rigid_radius),
            starting_geometry_balanced_under_original_contact_law=bool(abs(g[:p.nmicro]).max()<c['implicit_force_tolerance_N'] and contact.force_balance_ok(q,c['implicit_force_tolerance_N'])),
            initial_state_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        del contact,p;gc.collect()
    report=dict(scope='Initial geometry diagnostic, not an equilibrium solution. Material is stress-free; contact barrier forces can be nonzero.',
        source_study=str(study),rows=rows,parameters_changed=False,
        initial_state_is_geometry_not_equilibrium=True,unique_cause_of_mesh_difference_established=False)
    (out/'assessment.json').write_text(json.dumps(report,indent=2)+'\n')
    with (out/'contact-preload.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(rows),flush=True);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();audit(a.study,a.out)
