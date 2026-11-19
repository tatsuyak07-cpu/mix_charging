"""Export verified prototype repairs while retaining the failed coarser screen."""
import csv,hashlib,json,shutil
from pathlib import Path
import numpy as np


def export(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    report=json.loads((root/'repair-assessment-002.json').read_text())
    periodic=json.loads((root/'candidate-periodic-audit.json').read_text())
    if not report['prototype_repairs_verified'] or not periodic['passed']:raise ValueError('Repair verification incomplete')
    launches=[]
    for path in root.rglob('launch.json'):
        record=json.loads(path.read_text())
        if record['state']!='process_exited' or not record['runtime_hashes_unchanged']:raise ValueError('Live or modified calculation source')
        launches.append(dict(file=str(path),return_code=record['child_return_code']))
    for manifest in root.rglob('manifest.json'):
        for name,digest in json.loads(manifest.read_text()).items():
            if hashlib.sha256((manifest.parent/name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen runtime changed')
    with np.load(root/'prepared-mesh-001/mesh.npz') as a,np.load(root/'prepared-model-002/mesh.npz') as b:
        if any(not np.array_equal(a[k],b[k]) for k in ['points','tets']):raise ValueError('Delivered mesh differs from verified geometry')
    copy={
        'assessment.json':'repair-assessment-002.json','original-plan.json':'repair-plan.json',
        'extended-plan.json':'extended-contact-plan.json','periodic-audit.json':'candidate-periodic-audit.json',
        'guard-rejection.json':'guard-rejection-assessment.json','intentional-interruptions.json':'intentional-interruptions-001.json',
        'repaired-model-config.json':'prepared-model-002/config.json','repaired-reference-mesh.npz':'prepared-model-002/mesh.npz',
        'mesh-preparation.json':'prepared-model-002/metadata.json','original-contact-config.json':'input/original-contact-config.json'}
    for target,source in copy.items():shutil.copy2(root/source,out/target)
    cases=report['initial_cases']+report['active_cases']+report['adoption_equilibrium_replays']+report['original_coarser_family']['cases'][:1]
    for i,case in enumerate(cases):
        for name in ['config.json','history.csv','audit.json','initial-contact-preflight.json','provenance.json']:
            source=Path(case['run'])/name
            if source.exists():shutil.copy2(source,out/f'case-{i:02d}-{name}')
    with (out/'comparison.csv').open('x',newline='') as f:
        keys=['run','dhat_mm','compression','pressure_MPa','porosity','contacts','rigid_contacts','maximum_ep','free_residual_N','state_sha256']
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(cases)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    initial=report['initial_cases'];active=report['original_coarser_family']['cases'][:1]+report['active_cases']
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    x=[1000*r['dhat_mm'] for r in active];ax.plot(x,[r['pressure_MPa'] for r in active],'o-')
    ax.set_xscale('log',base=2);ax.set_xticks(x,[f'{a:g}' for a in x]);ax.invert_xaxis()
    ax.set(xlabel='Contact activation (micrometres)',ylabel='Nominal pressure (MPa)',title='Contact-active matched-history screen')
    ax.grid(alpha=.2);fig.savefig(out/'contact-sensitivity.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for axis,strain,title in zip(axes,[.0001,.005],['0.01% compression','0.5% compression']):
        pair=[next(r for r in initial if r['run'].endswith('coarse-small' if strain==.0001 else 'coarse-005')),
              next(r for r in initial if r['run'].endswith('refined-small' if strain==.0001 else 'refined-005'))]
        axis.bar(['65,407 T4','523,256 T4'],[r['pressure_MPa'] for r in pair],color=['steelblue','darkorange'])
        axis.set(ylabel='Nominal pressure (MPa)',title=title);axis.grid(axis='y',alpha=.2)
    fig.savefig(out/'mesh-comparison.png',dpi=180);plt.close(fig)
    from tools.export_pressure_stages import render
    from tools.export_rigid_compaction import sections
    from packing_fem.rigid_inclusion import make_periodic,cylinder_surface
    case=root/'candidate-replay-0025/refined-005';c=json.loads((case/'config.json').read_text())
    with np.load(case/'state-0000.npz') as f:start=dict(f)
    with np.load(case/'state-0001.npz') as f:end=dict(f)
    p=make_periodic(end['X'],c['length_mm'],c);center,(R,_,_)=p.pose(end['q']);local,faces=cylinder_surface(c)
    render(out/'refined-compression-mesh.png',end,'Refined prototype: axial compression0.5%\n523,256 T4; matrix porosity40.1287%',c['length_mm'],footer='16-grain prototype; actual positions1x. Zero-contact equilibrium revalidated at activation0.0025mm.\nNot a full40% to10% compression result; mesh/increment convergence unverified.',rigid_surface=(local@R.T+center,faces))
    sections(start,end,c,out/'refined-sections.png')
    (out/'execution-summary.json').write_text(json.dumps(dict(launches=launches,all_snapshots_unchanged=True,completed_loading_solves=sum(r['return_code']==0 for r in launches),expected_guard_rejection=1,intentionally_interrupted_duplicate_searches=2,zero_contact_equilibrium_replays=2,delivered_mesh_arrays_equal_verified_mesh=True,full_path_convergence_verified=False),indent=2)+'\n')


if __name__=='__main__':export('results/rigid-contact-mesh-repair-001','exports/rigid-contact-mesh-repair-001')
