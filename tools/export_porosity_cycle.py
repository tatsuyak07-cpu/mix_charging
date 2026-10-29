"""Publish actual porosity-cycle diagnostics, never an invented target state."""
import argparse,csv,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.export_pressure_stages import render
from packing_fem.checkpoint import load


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=False);run=a.source/'rve-cycle'
    summary=json.loads((run/'summary.json').read_text());rows=summary['attempts'][-1]['cycle']['records']
    fields=['time_s','axial_compression','porosity','pressure_MPa','force_ratio','kinetic_ratio','energy_ratio','contacts','minimum_quality','passed']
    with (root/'equilibrated-holds.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:
            out={k:row[k] for k in fields if k!='pressure_MPa'};out['pressure_MPa']=-row['cauchy_stress_MPa'][2][2];writer.writerow(out)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    initial_phi=json.loads((run/'geometry.json').read_text())['geometry']['initial_porosity']
    strain=np.r_[0,[r['axial_compression'] for r in rows]];phi=np.r_[initial_phi,[r['porosity'] for r in rows]]
    pressure=np.r_[0,[-r['cauchy_stress_MPa'][2][2] for r in rows]]
    axes[0].plot(strain,100*phi,'o-');axes[0].axhline(10,color='red',ls='--',label='Target (NOT reached)');axes[0].legend()
    axes[0].set(xlabel='Axial engineering compression',ylabel='Porosity (%)',ylim=(0,45))
    axes[1].plot(strain,pressure,'o-');axes[1].set(xlabel='Axial engineering compression',ylabel='Held axial Cauchy pressure (MPa)')
    fig.suptitle('Actual equilibrated holds; no contact yet; spatial convergence unverified');fig.savefig(root/'equilibrated-holds.png',dpi=170);plt.close(fig)
    checkpoints=list(run.glob('attempt-*/checkpoint-*'));last=rows[-1]
    chosen=None
    for checkpoint in checkpoints:
        payload=json.loads((checkpoint/'state.json').read_text());time=dict(payload['state']['dict']).get('time')
        if isinstance(time,(int,float)) and abs(time-last['time_s'])<1e-13:chosen=checkpoint;break
    if chosen is None:raise ValueError('Accepted hold checkpoint missing')
    state,metadata=load(chosen);render(root/'last-equilibrated-mesh.png',state,f"Actual held mesh: porosity {100*last['porosity']:.4f}%",metadata['config']['length_mm'])
    with np.load(run/'model.npz') as z:initial=dict(x=z['X'],t=z['t'],q=np.zeros(6),ep=np.zeros(len(z['t'])))
    render(root/'initial-mesh.png',initial,f'Actual initial voxel mesh: porosity {100*initial_phi:.5f}%',metadata['config']['length_mm'])
    # Keep every failure's diagnostics; full original directories remain local.
    for attempt in sorted(run.glob('attempt-*')):
        dst=root/'rve-attempts'/attempt.name;dst.mkdir(parents=True)
        for name in ('summary.json','config.json','history.csv','contact-audit.json','surface-resolution-attempts.json','porosity-cycle.json','accepted-step-audit.csv.gz','last-state.npz','last-mesh.vtu','embedded-quadrature-rules.json','contact-quadrature.npz'):
            if (attempt/name).exists():shutil.copy2(attempt/name,dst/name)
    shutil.copytree(chosen,root/'last-equilibrated-checkpoint')
    shutil.copytree(Path(summary['last_certified_noncontact_checkpoint']),root/'last-certified-noncontact-checkpoint')
    for folder in ('cad-mesh','cad-mesh-refined'):
        dst=root/folder;dst.mkdir()
        for name in ('metadata.json','mesh.npz','gmsh.log'):
            shutil.copy2(a.source/folder/name,dst/name)
    for folder in ('rve-cycle','rve-persistent-cycle','persistent-frozen-poses','convergence-gate-002'):
        if (a.source/folder/'summary.json').exists():
            dst=root/folder;dst.mkdir(exist_ok=True);shutil.copy2(a.source/folder/'summary.json',dst/'summary.json')
    for folder in ('full-tests-source','final-tests-source','persistent-frozen-poses'):
        if (a.source/folder).exists():shutil.copytree(a.source/folder,root/folder,dirs_exist_ok=True)
    for folder in ('rve-cycle','rve-persistent-cycle'):
        shutil.copytree(a.source/folder/'source',root/folder/'source')
        for path in (a.source/folder).glob('replay-rules-*.json'):shutil.copy2(path,root/folder/path.name)
    logs=root/'test-logs';logs.mkdir()
    for path in a.source.glob('*tests*.log'):shutil.copy2(path,logs/path.name)
    for path in a.source.glob('*regression*.log'):shutil.copy2(path,logs/path.name)
    if (a.source/'elastic-void-control').exists():shutil.copytree(a.source/'elastic-void-control',root/'elastic-void-control')
    (root/'README.md').write_text(Path('docs/porosity-cycle-development.md').read_text().replace('../exports/porosity-cycle-development-001/','./').replace('../exports/porosity-cycle-development-001.zip','../porosity-cycle-development-001.zip'))
    manifest={str(path.relative_to(root)):dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sorted(root.rglob('*')) if path.is_file()}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(str(root)+'.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for path in root.rglob('*'):
            if path.is_file():archive.write(path,str(path.relative_to(root.parent)))


if __name__=='__main__':main()
