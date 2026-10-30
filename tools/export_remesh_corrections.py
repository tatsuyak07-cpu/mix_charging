"""Publish component corrections with explicit incomplete-packing status."""
import argparse,csv,hashlib,json,re,shutil,zipfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=False)
    cases={name:json.loads((a.source/name/'summary.json').read_text()) for name in ('moving-final-020','moving-final-010','moving-final-005','moving-l2-control','moving-stop-control')}
    packing=json.loads((a.source/'packing-eligibility/summary.json').read_text())
    test_log=(a.source/'final-current-tests.log').read_text();match=re.search(r'(\d+) passed',test_log)
    if match is None or re.search(r'\d+ failed',test_log):raise ValueError('Final current-source tests are not all passed')
    assessment=dict(component_corrections_verified=all(cases[n]['passed'] for n in ('moving-final-020','moving-final-010','moving-final-005')),
                    stage3_completed=False,full_compaction_completed=False,final_current_tests_passed=int(match.group(1)),solver_cases=cases,packing_eligibility=packing,
                    limits=['Three bounded steps on manufactured planar contact, not quasi-static compression.',
                            'CFL levels test transfer invariants, not same-time response convergence.',
                            'Packing still has incompatible child areas/sample masses; no history reset or reweighting allowed.',
                            'No general sample insertion/coarsening, retained-partition clipping, quality recovery, loaded 10% equilibrium or unload result.'])
    (root/'assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
    shutil.copytree(a.source,root/'diagnostics')
    source=root/'implementation';source.mkdir()
    for name in ('packing_fem','tools','tests'):
        target=source/name;target.mkdir()
        for path in Path(name).glob('*.py'):shutil.copy2(path,target/path.name)
    logs=root/'test-logs';logs.mkdir()
    for path in a.source.glob('*.log'):shutil.copy2(path,logs/path.name)
    rows=[]
    for name,result in cases.items():
        event=result['results']['enabled']['events'][0]
        errors=event.get('scaled_errors',{})
        rows.append(dict(case=name,passed=result['passed'],remesh_accepted=event['accepted'],steps=result['results']['enabled']['steps'],
                         angular_momentum_error=errors.get('angular_momentum'),kinetic_energy_error=errors.get('kinetic_energy'),
                         contact_macro_force_error=errors.get('contact_macro_force'),stop_reason=result['results']['enabled']['failure']))
    with (root/'comparison.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    old=next(r for r in rows if r['case']=='moving-l2-control');new=next(r for r in rows if r['case']=='moving-final-020')
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained');x=np.arange(2)
    ax.bar(x-.18,[old['angular_momentum_error'],old['kinetic_energy_error']],width=.36,label='L2 velocity transfer',color='#b95c42')
    ax.bar(x+.18,[new['angular_momentum_error'],new['kinetic_energy_error']],width=.36,label='Constrained transfer',color='#267ba6');ax.axhline(.001,ls='--',color='red',label='Unchanged remesh criterion')
    ax.set_yscale('log');ax.set_xticks(x,['Angular momentum','Kinetic energy']);ax.set(ylabel='Scaled transfer error',title='Moving contact / same geometry and physical parameters');ax.legend();fig.savefig(root/'conservation-errors.png',dpi=170);plt.close(fig)
    (root/'README.md').write_text(Path('docs/remesh-corrections.md').read_text().replace('../exports/remesh-corrections-001/','./').replace('../exports/remesh-corrections-001.zip','../remesh-corrections-001.zip'))
    manifest={str(path.relative_to(root)):dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in root.rglob('*') if path.is_file()}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(str(root)+'.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for path in root.rglob('*'):
            if path.is_file():archive.write(path,str(path.relative_to(root.parent)))


if __name__=='__main__':main()
