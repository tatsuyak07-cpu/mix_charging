"""Assemble an accepted restart chain for export, preserving every original run."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
import numpy as np


def merge(segments,out):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    status=json.loads((Path(segments[-1][0])/'status.json').read_text())
    if status.get('status')!='loaded_target_reached' or not status.get('target_reached'):raise ValueError('Final segment has not reached the loaded target')
    physics=['length_mm','diameter_mm','neck_diameter_mm','young_mpa','poisson','yield_mpa','hardening_mpa','friction','transverse_boundary','ipc_dhat_mm','ipc_stiffness_mpa','ipc_slip_regularization_mm','implicit_force_tolerance_N','implicit_macro_tolerance']
    records=[];provenance=[];previous=None;base=None;elapsed=0.;copies=[]
    for run,last in segments:
        run=Path(run);c=json.loads((run/'config.json').read_text())
        if base is None:base=c
        if any(c.get(key)!=base.get(key) for key in physics):raise ValueError('Physical model or acceptance criteria changed across restart chain')
        initial=run/'state-0000.npz'
        if previous is not None:
            with np.load(previous) as a,np.load(initial) as b:
                if set(a.files)!=set(b.files) or any(not np.array_equal(a[key],b[key]) for key in a.files):raise ValueError('Restart chain is not an exact continuation')
        else:copies.append((initial,0))
        with (run/'history.csv').open() as stream:rows=list(csv.DictReader(stream))
        selected=[row for row in rows if last<0 or int(row['step'])<=last]
        if not selected or (last>=0 and int(selected[-1]['step'])!=last):raise ValueError('Requested accepted prefix is unavailable')
        for row in selected:
            original_step=int(row['step']);source=run/f'state-{original_step:04d}.npz'
            row=dict(row,source_run=str(run),source_step=original_step,source_wall_s=row['wall_s'])
            row['step']=len(records)+1;row['wall_s']=elapsed+float(row['wall_s']);records.append(row)
            copies.append((source,len(records)))
        elapsed=float(records[-1]['wall_s']);previous=copies[-1][0]
        provenance.append(dict(run=str(run),last_source_step=int(selected[-1]['step']),config_sha256=hashlib.sha256((run/'config.json').read_bytes()).hexdigest(),exact_restart_check_passed=True))
    if float(records[-1]['porosity'])>c['target_final_porosity']+c['porosity_tolerance']:raise ValueError('Selected final prefix does not contain the loaded target')
    out.mkdir(parents=True,exist_ok=False)
    manifest=[]
    for source,step in copies:
        target=out/f'state-{step:04d}.npz';shutil.copy2(source,target)
        manifest.append(dict(step=step,source=str(source),sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    c['usage_scope']='Merged accepted restart chain; source segments remain unmodified.'
    c['wall_time_interpretation']='Cumulative wall times along the selected restart path; excludes other trial branches.'
    (out/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    (out/'status.json').write_text(json.dumps(dict(status,merged_path=True,validated=False),indent=2)+'\n')
    (out/'provenance.json').write_text(json.dumps(dict(segments=provenance,states=manifest),indent=2)+'\n')
    with (out/'history.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=records[0].keys());writer.writeheader();writer.writerows(records)
    print(json.dumps(dict(states=len(copies),last_porosity=records[-1]['porosity'],out=str(out))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--segment',nargs=2,action='append',required=True,metavar=('RUN','LAST_STEP'));p.add_argument('--out',required=True);a=p.parse_args()
    merge([(path,int(step)) for path,step in a.segment],a.out)
