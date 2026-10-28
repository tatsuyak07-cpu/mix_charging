"""Publish conservative local-material contact screening and unresolved guards."""
import argparse,csv,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
from packing_fem.material_neighborhood import MaterialNeighborhood
from packing_fem.barrier_contact import feature
from tools.run_barrier_compaction import fixture


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('preflight','evidence','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    shutil.copytree(a.preflight,a.output/'preflight');shutil.copytree(a.evidence,a.output/'evidence')
    old=json.loads(Path('exports/contact-mesh-preflight-002/summary.json').read_text())
    current=json.loads((a.preflight/'summary.json').read_text());candidates=old['levels'][3]['initial']['diagnostic']['candidates']
    X,t=fixture(4);sensitivity=[]
    for radius in (.05,.1,.2):
        c=MaterialNeighborhood(X,t,4.,radius);decisions=[]
        for row in candidates:
            ids=np.array(row['ids']);points=X[ids];_,w=feature(points,'pt')
            decisions.append(c.local(('corner',*map(int,ids)),ids,'pt',np.zeros(3),points,w,X,np.eye(3)*4))
        sensitivity.append(dict(reference_radius_mm=radius,material_connected_candidates=decisions))
    testlog=(a.evidence/'material-neighborhood-full-tests-001.log').read_text()
    focused=(a.evidence/'material-neighborhood-tests-final.log').read_text()
    summary=dict(scope='Local point/face material-neighbour screening; not complete self-contact or mesh convergence',reference_radius_sensitivity=sensitivity,old_error=old['levels'][3]['initial']['error'],new_initial_error=current['levels'][3]['initial']['error'],new_mapped_error=current['levels'][3]['mapped']['error'],all_old_point_candidates_material_connected=all(all(v['material_connected_candidates']) for v in sensitivity),full_test_result=testlog.strip().splitlines()[-1],focused_test_result=focused.strip().splitlines()[-1],refined_preflight_passed=current['preflight_passed'],geometry_checks_passed=current['checks']['geometry_preserved'],pressure_cycle_mesh_convergence_verified=False,particle_compaction_verified=False,decision='Point candidates are screened conservatively; edge-only approach and uncertified deformed material bridge still stop. Do not accept a refined pressure-cycle solution.')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (a.output/'error-comparison.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['refinement','state','before','after'])
        for before,after in zip(old['levels'],current['levels']):
            for state in ('initial','mapped'):w.writerow([before['refinement'],state,before[state].get('error','resolved'),after[state].get('error','resolved')])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig=plt.figure(figsize=(12,5));ax=fig.add_subplot(1,2,1,projection='3d')
    triangles=[np.array(v['reference_points'])[1:] for v in candidates];point=np.array(old['levels'][3]['initial']['diagnostic']['source_reference'])
    ax.add_collection3d(Poly3DCollection(triangles,facecolor='steelblue',edgecolor='navy',alpha=.4));ax.scatter(*point,color='red',s=40)
    allpoints=np.concatenate(triangles+[point[None]]);lo=allpoints.min(0);hi=allpoints.max(0)
    ax.set(xlim=(lo[0]-.02,hi[0]+.02),ylim=(lo[1]-.02,hi[1]+.02),zlim=(lo[2]-.005,hi[2]+.005),title='Historical PT candidates: material-connected',xlabel='x (mm)',ylabel='y (mm)',zlabel='z (mm)')
    ax=fig.add_subplot(1,2,2,projection='3d');hit=current['levels'][3]['initial']['failure_feature'];points=np.array(hit['points'])
    for edge,color in zip((points[:2],points[2:]),('orange','red')):ax.plot(*edge.T,'o-',color=color)
    lo=points.min(0);hi=points.max(0);ax.set(xlim=(lo[0]-.02,hi[0]+.02),ylim=(lo[1]-.02,hi[1]+.02),zlim=(lo[2]-.005,hi[2]+.005),title=f"EE guard retained: gap {hit['distance_mm']:.8f} mm",xlabel='x (mm)',ylabel='y (mm)',zlabel='z (mm)')
    fig.suptitle('Local contact candidate verification: partial correction\nRefined geometry preflight still FAILED; no pressure-cycle solution');fig.tight_layout();fig.savefig(a.output/'local-material-candidates.png',dpi=160);plt.close(fig)
    for path in (Path(__file__),Path('tests/test_material_neighborhood.py'),Path('tests/test_contact_mesh_preflight.py')):
        target=a.output/'source'/path.resolve().relative_to(Path.cwd());target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    (a.output/'README.md').write_text('Conservative material-neighbour screening; unresolved refined geometry guards remain FAILED. No refined-cycle or packing result. Units mm, N, MPa. See summary.json and preflight/summary.json. Failed preliminary trials retained under evidence/.\n')
    hashes={str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest() for f in a.output.rglob('*') if f.is_file()};(a.output/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    with zipfile.ZipFile(a.output.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as z:
        for f in a.output.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(a.output.parent))
    with zipfile.ZipFile(a.output.with_suffix('.zip')) as z:
        if z.testzip():raise ValueError('ZIP CRC failure')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
