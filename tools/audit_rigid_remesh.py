"""Replay an archived contact remesh and its actual zero-macro re-equilibrium."""
import argparse,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
from packing_fem.rigid_inclusion import make_periodic,make_contact
from packing_fem.explicit_periodic import deformation
from packing_fem.explicit_material import geometry
from packing_fem.implicit_material import evaluate
from packing_fem.friction_audit import friction_report
from tools.audit_rigid_compaction import load


def audit(run,step,event,out):
    run=Path(run);out=Path(out)
    if out.exists():raise FileExistsError(out)
    c=json.loads((run/'config.json').read_text());old=load(run/f'state-{step:04d}.npz');d=load(run/f'remesh-{event:03d}.npz')
    source=run/'source/implicit_remesh.py';manifest=json.loads((run/'source-sha256.json').read_text())
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if digest!=manifest['packing_fem/implicit_remesh.py']:raise ValueError('Archived remesh source hash mismatch')
    # The archived policy is deliberately replayed, even if the current policy
    # correctly rejects a roundoff-only quality gain.
    spec=importlib.util.spec_from_file_location('packing_fem.archived_rigid_remesh',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    try:nx,nt,nb,ne,nv,parent,report=m.refine_deformed_edges(old['x'],old['t'],old['b'],old['ep'],old['V0'],c,old['parent'],old['X'],c['length_mm'])
    except ValueError:nx,nt,nb,ne,nv,parent,report=m.adaptive_remesh(old['x'],old['t'],old['b'],old['ep'],old['V0'],c,parent=old['parent'])
    newX=report.pop('_reference_points',None);macro=old['q'][-6:];op=make_periodic(old['X'],c['length_mm'],c)
    if newX is None:newX=nx@np.linalg.inv(deformation(macro)).T
    p=make_periodic(newX,c['length_mm'],c);q=p.remap_positions(nx,macro);q[p.rigid_dofs]=old['q'][op.rigid_dofs]
    topology=all(np.array_equal(a,b) for a,b in [(nt,d['t']),(parent,d['parent']),(nv,d['V0'])]);chart=float(abs(newX-d['X']).max())
    center,(R,_,_)=p.pose(q);oldcenter,(oldR,_,_)=op.pose(old['q']);pose_error=max(float(abs(center-oldcenter).max()),float(abs(R-oldR).max()))
    _,gm,_,state=evaluate(d['x'],nt,nb,ne,np.linalg.inv(geometry(nx,nt)[2]),nv,c,parent=parent)
    contact=make_contact(newX,nt,p,c);_,gc,_,count=contact.evaluate_coupled(d['q'],q);gradient=np.asarray(p.A.T@gm.ravel()).ravel()+gc
    force,torque=contact.rigid_resultant(d['q']);friction=friction_report(contact,d['q'],q,c['friction'])
    quality_gain=float(report['quality_after']-report['quality_before']);required=max(c.get('remesh_quality_gain_absolute_tolerance',1e-12),c.get('remesh_quality_gain_relative_tolerance',1e-8)*report['quality_before'])
    result=dict(archived_policy_replayed=True,archived_remesh_source_sha256=digest,topology_and_quadrature_identical=topology,chart_error_mm=chart,transfer_rigid_pose_error=pose_error,b_replay_error=float(abs(state['b']-d['b']).max()),ep_replay_error=float(abs(state['ep']-d['ep']).max()),free_residual_N=float(abs(gradient[:p.nmicro]).max()),rigid_force_N=force.tolist(),rigid_torque_Nmm=torque.tolist(),rigid_contacts=contact.rigid_contact_count(),contacts=count,coulomb=friction,quality_gain=quality_gain,required_meaningful_quality_gain=required,meaningful_quality_improvement=bool(quality_gain>required),transfer_report=report)
    # The archived successful transfer is audited separately from whether its
    # old quality gate should have admitted it. Never certify roundoff as gain.
    result['passed']=bool(topology and chart<1e-10 and pose_error<1e-10 and result['b_replay_error']<1e-7 and result['ep_replay_error']<1e-7 and result['free_residual_N']<c['implicit_force_tolerance_N'] and contact.force_balance_ok(d['q'],c['implicit_force_tolerance_N']) and friction['passed'])
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);return result['passed']


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',required=True);a.add_argument('--step',required=True,type=int);a.add_argument('--event',required=True,type=int);a.add_argument('--out',required=True);args=a.parse_args()
    if not audit(args.run,args.step,args.event,args.out):raise SystemExit(1)
