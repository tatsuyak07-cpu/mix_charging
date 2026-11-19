"""Revalidate a zero-contact solution at a smaller cutoff without a new solve.

Geometry/material history are copied byte-for-byte. This is explicitly a replay,
not a claim of a new nonlinear calculation or of active-contact insensitivity.
"""
import argparse,csv,gc,hashlib,json,shutil
from pathlib import Path
import numpy as np
from packing_fem.rigid_inclusion import make_periodic,make_contact
from packing_fem.initial_contact import inspect
from tools.audit_rigid_compaction import audit


def replay(source,target,reference,out):
    source=Path(source);out=Path(out);out.mkdir(exist_ok=False,parents=True)
    c=json.loads((source/'config.json').read_text());old_delta=c['ipc_dhat_mm']
    old_audit=json.loads((source/'audit.json').read_text())
    if not old_audit['passed'] or not 0<target<old_delta:raise ValueError('Need audited zero-contact solution and smaller cutoff')
    with (source/'history.csv').open() as f:rows=list(csv.DictReader(f))
    if any(int(r['contacts']) for r in rows):raise ValueError('Cannot reuse a contact-active solution')
    manifest=[]
    for path in sorted(source.glob('state-*.npz')):
        shutil.copy2(path,out/path.name);manifest.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if manifest[-1]['sha256']!=old_audit['states'][-1]['state_sha256']:raise ValueError('Source audit is stale')
    shutil.copy2(source/'history.csv',out/'history.csv');c.update(ipc_dhat_mm=target,usage_scope='Independent smaller-cutoff replay of an already converged zero-contact solution; no new nonlinear loading solve.')
    (out/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    original=json.loads(Path(reference).read_text());original['ipc_dhat_mm']=target
    (out/'reference-physics.json').write_text(json.dumps(original,indent=2)+'\n')
    with np.load(out/'state-0000.npz') as f:X=f['X'];t=f['t'];q=f['q']
    p=make_periodic(X,c['length_mm'],c);contact=make_contact(X,t,p,c);preflight=inspect(contact,q,c)
    if not preflight['passed']:raise ValueError('Replayed initial geometry has contact preload')
    (out/'initial-contact-preflight.json').write_text(json.dumps(preflight,indent=2)+'\n');del contact,p;gc.collect()
    if not audit(out,out/'audit.json',out/'reference-physics.json',True):raise ValueError('New cutoff equilibrium replay failed')
    (out/'provenance.json').write_text(json.dumps(dict(run_kind='zero_contact_equilibrium_revalidation',new_nonlinear_solve_performed=False,source=str(source),source_activation_mm=old_delta,target_activation_mm=target,geometry_and_material_history_copied_without_change=True,states=manifest),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--target',type=float,required=True);p.add_argument('--reference',required=True);p.add_argument('--out',required=True);a=p.parse_args();replay(a.source,a.target,a.reference,a.out)
