"""Export near-zero edge classification evidence and its remaining limitation."""
import argparse,csv,hashlib,json,re,shutil,zipfile
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    diagnostic=json.loads((a.input/'diagnostic-003/summary.json').read_text())
    log=(a.input/'full-tests.log').read_text();count=re.search(r'(\d+) passed in ',log)
    assert diagnostic['passed'] and count and 'failed' not in log and 'skipped' not in log
    a.output.mkdir(parents=True,exist_ok=False);shutil.copytree(a.input,a.output/'validation')
    source=Path('exports/closest-selection-001/validation/pressure-run/calculation')
    original=a.output/'input-state';original.mkdir()
    for name in ('config.json','summary.json','last-state.npz','contact-audit.json'):shutil.copy2(source/name,original/name)
    code=a.output/'source';code.mkdir()
    for f in Path('packing_fem').glob('*.py'):shutil.copy2(f,code/f.name)
    for name in ('tools/verify_near_edge.py','tools/export_near_edge.py','tests/test_near_edge.py'):shutil.copy2(name,code/Path(name).name)
    result=dict(scope=__doc__,whole_suite_passed=int(count.group(1)),diagnostic=diagnostic,
                roundoff_screening_corrected=True,endpoint_contact_coverage_fixed=False,
                pressure_integration_performed=False,physical_compaction_verified=False)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    rows=[dict(case=i,kind=r['kind'],legacy_screened=r['legacy']['dominated'],legacy_cosine=r['legacy']['maximum_descent_cosine'],corrected_screened=r['corrected']['dominated'],corrected_cosine=r['corrected']['maximum_descent_cosine']) for i,r in enumerate(diagnostic['variants'],1)]
    with (a.output/'classification-variants.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    variants=diagnostic['variants'];fig,ax=plt.subplots(1,2,figsize=(11,4.5))
    for label,key in (('Archived screening','legacy'),('Corrected off-edge screening','corrected')):
        ax[0].plot(np.arange(1,len(variants)+1),[r[key]['maximum_descent_cosine'] for r in variants],'.-',label=label)
    ax[0].set(yscale='symlog',xlabel='Rigid-motion / coordinate-noise case',ylabel='Maximum candidate descent cosine');ax[0].axhline(0,color='black',lw=.7);ax[0].legend(fontsize=8);ax[0].grid(alpha=.25)
    ax[0].set_yscale('symlog',linthresh=1e-8)
    endpoint=diagnostic['endpoint_contacts'][0]
    ax[1].bar(['Closest EE gap','Endpoint PT gap','Endpoint to EE closest'],[float(diagnostic['decimal_distance_mm']),endpoint['point_triangle_gap_mm'],endpoint['source_distance_to_EE_closest_point_mm']],color=['#bb5544','#4d8caf','#4b9878'])
    ax[1].set(yscale='log',ylabel='Distance [mm]');ax[1].tick_params(axis='x',labelrotation=15);ax[1].grid(axis='y',alpha=.25)
    fig.suptitle('Saved transient geometry — classification evidence, NOT a compaction solution');fig.tight_layout();fig.savefig(a.output/'near-edge-checks.png',dpi=180);plt.close(fig)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from packing_fem.contact_geometry import closest_triangle
    with np.load(source/'last-state.npz') as z:x=z['points']
    fig=plt.figure(figsize=(7,6));ax=fig.add_subplot(111,projection='3d')
    for ids,color,label in (([175,384],'#bb5544','First edge'),([385,401],'#4d8caf','Second edge')):
        points=x[ids];ax.plot(*points.T,'o-',color=color,label=label)
        for node in ids:ax.text(*x[node],str(node))
    tri=x[endpoint['target_face']];ax.add_collection3d(Poly3DCollection([tri],alpha=.3,facecolor='orange',edgecolor='orange'))
    q=closest_triangle(x[401],tri)[0];ax.plot(*np.array([x[401],q]).T,'--',color='orange',label='Endpoint point contact')
    s=float(diagnostic['decimal_s']);ee=(1-s)*x[175]+s*x[384];ax.scatter(*ee,color='black',s=45,label='EE closest pair (~3e-10 mm gap)')
    ax.set(xlabel='x [mm]',ylabel='y [mm]',zlabel='z [mm]',title='Saved local geometry — NOT equilibrium');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(a.output/'near-edge-geometry.png',dpi=180);plt.close(fig)
    hashes={str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.output.rglob('*')) if f.is_file()}
    (a.output/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    with zipfile.ZipFile(a.output.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(a.output.rglob('*')):
            if f.is_file():z.write(f,str(Path(a.output.name)/f.relative_to(a.output)))
    print(json.dumps(dict(output=str(a.output),files=len(hashes),whole_suite_passed=int(count.group(1)))))


if __name__=='__main__':main()
