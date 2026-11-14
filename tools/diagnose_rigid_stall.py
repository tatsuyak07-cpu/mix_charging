"""Replay the actual stopping point, including the zero-load-increment limit."""
import argparse,json
from pathlib import Path
import numpy as np
from packing_fem.rigid_inclusion import make_periodic,make_contact
from packing_fem.explicit_material import geometry
from packing_fem.implicit_material import evaluate
from tools.audit_rigid_compaction import load


def diagnose(directory,out):
    directory=Path(directory);out=Path(out)
    if out.exists():raise FileExistsError(out)
    c=json.loads((directory/'original-config.json').read_text());accepted=load(directory/'accepted.npz');previous=load(directory/'previous.npz');stalled=load(directory/'stalled.npz');p=make_periodic(accepted['X'],c['length_mm'],c);contact=make_contact(accepted['X'],accepted['t'],p,c)
    rows=[]
    for name,q,old in [('accepted_replay',accepted['q'],previous),('zero_macro_increment',accepted['q'],accepted),('stalled_trial',stalled['q'],accepted)]:
        _,gm,_,state=evaluate(p.positions(q),old['t'],old['b'],old['ep'],np.linalg.inv(geometry(old['x'],old['t'])[2]),old['V0'],c,parent=old['parent'])
        _,gc,_,count=contact.evaluate_coupled(q,old['q']);solid=np.asarray(p.A.T@gm.ravel()).ravel();x=contact.positions(q);normal=np.asarray(contact.jacobian(q).T@contact.barrier.gradient(contact.last_normal,contact.mesh,x)).ravel();friction=gc-normal;total=solid+gc;force,torque=contact.rigid_resultant(q);slips=[];u=contact.displacement(q,old['q'])
        for kind in ['vv_collisions','ev_collisions','ee_collisions','fv_collisions']:
            for item in getattr(contact.lag,kind):
                local=item.dof(u,contact.edges,contact.faces);slips.append(float(np.linalg.norm(item.tangent_basis.T@item.relative_velocity_jacobian()@local)))
        row=dict(case=name,compression=float(-q[p.prescribed]),strain_increment=float(old['q'][p.prescribed]-q[p.prescribed]),free_maximum_N=float(abs(total[:p.nmicro]).max()),free_l2_N=float(np.linalg.norm(total[:p.nmicro])),solid_dof_maximum_N=float(abs(total[:p.body_nmicro]).max()),rigid_dof_maximum_N=float(abs(total[p.rigid_dofs]).max()),material_gradient_maximum_N=float(abs(solid[:p.nmicro]).max()),normal_gradient_maximum_N=float(abs(normal[:p.nmicro]).max()),friction_gradient_maximum_N=float(abs(friction[:p.nmicro]).max()),rigid_force_N=force.tolist(),rigid_torque_over_radius_N=(torque/p.rigid_radius).tolist(),contacts=count,slip_regularization_mm=c['ipc_slip_regularization_mm'],tangential_stencils=len(slips),slips_below_regularization=sum(s<c['ipc_slip_regularization_mm'] for s in slips),slip_min_median_max_mm=[float(v) for v in np.quantile(slips,[0,.5,1])] if slips else [])
        rows.append(row);print(json.dumps(row),flush=True)
    q=stalled['q'];_,_,K,_=contact.evaluate_coupled(q,accepted['q'],True);direction=np.zeros(p.ndof);direction[p.rigid_dofs]=[.3,-.4,.2,.5,.2,-.1];analytic=K@direction;checks=[]
    for h in [1e-7,1e-8]:
        plus=contact.evaluate_coupled(q+h*direction,accepted['q'])[1];minus=contact.evaluate_coupled(q-h*direction,accepted['q'])[1];fd=(plus-minus)/(2*h)
        checks.append(dict(h_mm=h,relative_l2_error=float(np.linalg.norm(fd-analytic)/np.linalg.norm(fd)),maximum_error=float(abs(fd-analytic).max())))
    report=dict(accepted_force_replay_passed=rows[0]['free_maximum_N']<c['implicit_force_tolerance_N'],same_material_geometry_friction_and_tolerances=True,interpretation='The original regularized incremental slip law rebuilds friction from q-oldq; no elastic tangential traction history is stored. Zero-increment residual is diagnostic, not an accepted solution.',cases=rows,rigid_direction_jacobian_checks=checks)
    out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--out',required=True);args=parser.parse_args();diagnose(args.input,args.out)
