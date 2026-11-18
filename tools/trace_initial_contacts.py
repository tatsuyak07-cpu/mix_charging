"""Trace every signed initial barrier stencil, retaining exact force cancellation."""
import argparse,json
from pathlib import Path
import numpy as np
from packing_fem.rigid_inclusion import make_periodic,make_contact


def trace(run,out):
    run=Path(run);out=Path(out)
    if out.exists():raise FileExistsError(out)
    c=json.loads((run/'config.json').read_text())
    with np.load(run/'state-0000.npz') as f:X=f['X'];t=f['t'];q=f['q']
    p=make_periodic(X,c['length_mm'],c);contact=make_contact(X,t,p,c)
    value,g,_,_=contact.evaluate_coupled(q,q);normal=contact.last_normal;x=contact.positions(q)
    rows=[]
    for kind in ['vv_collisions','ev_collisions','ee_collisions','fv_collisions']:
        for item in getattr(normal,kind):
            ids=np.asarray(item.vertex_ids(contact.edges,contact.faces));ids=ids[ids>=0]
            row=dict(kind=kind,weight=float(item.weight),initial_positions_mm=contact.X[ids].tolist(),
                distance_mm=float(np.linalg.norm(item.compute_distance_vector(x,contact.edges,contact.faces))),
                involves_rigid=bool(np.any(ids>=contact.body_vertices)))
            rows.append(row)
    report=dict(rows=rows,original_dhat_mm=c['ipc_dhat_mm'],original_force_max_N=float(abs(g[:p.nmicro]).max()),
        full_contact_energy_Nmm=float(value),
        any_rigid_contact=any(r['involves_rigid'] for r in rows),
        note='Full contact set evaluated together. Feature geometry and signed weights only; no per-stencil force additivity assumed. No filtering or law change.')
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();trace(a.run,a.out)
