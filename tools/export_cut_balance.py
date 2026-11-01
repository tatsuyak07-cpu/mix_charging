"""Export a negative refinement-eligibility result without claiming compaction."""
import argparse,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=False)
    result=json.loads((a.source/'rve-analysis/summary.json').read_text());separations=[]
    pairs=[e for e in result['edges'] if not e['missing_rule_faces']]
    fig,axes=plt.subplots(1,len(pairs),figsize=(12,4),layout='constrained');axes=np.atleast_1d(axes)
    for ax,edge in zip(axes,pairs):
        for face in edge['faces']:
            s=face['positions'];cdf=face['cumulative_fraction'];x=np.r_[0,s,1];y=np.r_[cdf,cdf[-1]]
            ax.step(x,y,where='post',label=f"Face {face['face']}")
            for f in face['roots']:ax.plot(f,f,'o',ms=4,color=ax.lines[-1].get_color())
        ax.plot([0,1],[0,1],'k--',label='Required child area f A / A');ax.axvline(.5,color='#999999',ls=':')
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Edge split fraction f',ylabel='Inherited reference-area weight / parent area',title=f"Edge {edge['edge'][0]}-{edge['edge'][1]}: no common interval");ax.legend(fontsize=8)
        left,right=[f['intervals'] for f in edge['faces']]
        gap=min(max(x[0]-y[1],y[0]-x[1],0.) for x in left for y in right)
        separations.append(dict(edge=edge['edge'],minimum_parameter_interval_gap=gap))
    fig.suptitle('Fixed samples and weights: necessary area conditions only');fig.savefig(root/'cut-balance.png',dpi=170);plt.close(fig)
    robustness=json.loads((a.source/'boundary-robustness/summary.json').read_text())
    assessment=dict(boundary_robustness=robustness,analysis_completed=True,unchanged_sample_refinement_available=result['eligible_candidates']>0,packing_remesh_completed=False,full_compaction_completed=False,
                    interval_separations=separations,result=result)
    (root/'assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
    for name in ['face-balance.csv','edge-outcomes.csv']:shutil.copy2(a.source/'rve-analysis'/name,root/name)
    shutil.copytree(a.source,root/'diagnostics')
    source=root/'implementation';source.mkdir()
    for name in ['tools/cut_balance.py','tools/verify_packing_cut_balance.py','tools/export_cut_balance.py','tools/check_cut_boundaries.py','tests/test_cut_balance.py']:
        shutil.copy2(name,source/Path(name).name)
    logs=root/'test-logs';logs.mkdir()
    for path in a.source.glob('*.log'):shutil.copy2(path,logs/path.name)
    (root/'README.md').write_text(Path('docs/cut-balance-verification.md').read_text().replace(f'../exports/{root.name}/','./').replace(f'../exports/{root.name}.zip',f'../{root.name}.zip'))
    manifest={str(path.relative_to(root)):dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in root.rglob('*') if path.is_file()}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(str(root)+'.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for path in root.rglob('*'):
            if path.is_file():archive.write(path,str(path.relative_to(root.parent)))


if __name__=='__main__':main()
