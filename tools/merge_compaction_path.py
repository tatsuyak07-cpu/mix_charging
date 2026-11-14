"""Assemble an accepted restart chain for export, preserving every original run."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
import numpy as np
from packing_fem.explicit_material import geometry
from packing_fem.explicit_periodic import deformation


def merge(segments,out):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    status=json.loads((Path(segments[-1][0])/'status.json').read_text())
    if status.get('status')!='loaded_target_reached' or not status.get('target_reached'):raise ValueError('Final segment has not reached the loaded target')
    physics=['rigid_inclusion','rigid_diameter_mm','rigid_height_mm','rigid_segments','rigid_center_mm','requested_matrix_porosity','length_mm','diameter_mm','neck_diameter_mm','young_mpa','poisson','yield_mpa','hardening_mpa','friction','transverse_boundary','ipc_dhat_mm','ipc_stiffness_mpa','ipc_slip_regularization_mm','implicit_force_tolerance_N','implicit_macro_tolerance']
    records=[];provenance=[];previous=None;base=None;elapsed=0.;copies=[];initial_porosity=None
    for run,last in segments:
        run=Path(run);c=json.loads((run/'config.json').read_text())
        if base is None:
            base=c
            # A source run may have failed AFTER this accepted prefix, leaving
            # only failure metadata in status.json. Use the actual initial mesh.
            with np.load(run/'state-0000.npz') as initial:
                volume=float(geometry(initial['x'],initial['t'])[0].sum())
                cell=c['length_mm']**3*np.linalg.det(deformation(initial['q'][-6:]))
                if c.get('rigid_inclusion',False):
                    from packing_fem.rigid_inclusion import porosities
                    initial_porosity=porosities(volume,cell,c)[0]
                else:initial_porosity=1-volume/cell
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
    transfers=[]
    for index,(run,_) in enumerate(segments,1):
        for source in sorted(Path(run).glob('remesh-*.npz')):
            target=out/f'remesh-segment-{index:02d}-{source.name.removeprefix("remesh-")}'
            shutil.copy2(source,target)
            record=dict(source=str(source),file=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest())
            if source.with_suffix('.json').exists():
                report=target.with_suffix('.json');shutil.copy2(source.with_suffix('.json'),report)
                record['report_file']=report.name;record['report_sha256']=hashlib.sha256(report.read_bytes()).hexdigest()
            transfers.append(record)
    c['usage_scope']='Merged accepted restart chain; source segments remain unmodified.'
    c['wall_time_interpretation']='Cumulative wall times along selected accepted segments; excludes other trial branches and initial-guess preparation outside those segments.'
    (out/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    (out/'status.json').write_text(json.dumps(dict(status,initial_porosity=initial_porosity,merged_path=True,validated=False),indent=2)+'\n')
    (out/'provenance.json').write_text(json.dumps(dict(segments=provenance,states=manifest,local_remesh_predecessors=transfers),indent=2)+'\n')
    with (out/'history.csv').open('w',newline='') as stream:
        columns=list(dict.fromkeys(key for row in records for key in row))
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(records)
    print(json.dumps(dict(states=len(copies),last_porosity=records[-1]['porosity'],out=str(out))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--segment',nargs=2,action='append',required=True,metavar=('RUN','LAST_STEP'));p.add_argument('--out',required=True);a=p.parse_args()
    merge([(path,int(step)) for path,step in a.segment],a.out)
