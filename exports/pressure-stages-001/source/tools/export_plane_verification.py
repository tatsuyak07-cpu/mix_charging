"""Export the flat-interface verification, including failed physical checks."""
import argparse,json,shutil,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from tools.verify_plane_contact import fixture,evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    for name in ('plane-contact-verification-001','plane-contact-verification-002','plane-contact-phase-path-001'):
        src=Path('results')/name
        if not (src/'summary.json').exists():raise RuntimeError(f'Incomplete verification: {src}')
        shutil.copytree(src,a.output/name)
    matrix=np.genfromtxt(a.output/'plane-contact-verification-002/plane-contact.csv',names=True,delimiter=',')
    path=np.genfromtxt(a.output/'plane-contact-phase-path-001/plane-contact.csv',names=True,delimiter=',')
    fig,axes=plt.subplots(1,3,figsize=(14,4));gap=.015
    for phase in (0.,.25,.5):
        r=matrix[np.isclose(matrix['gap_mm'],gap)&np.isclose(matrix['phase_fraction'],phase)]
        axes[0].plot(r['refinement'],r['force_N'],'o-',label=f'phase {phase:g} h')
    axes[0].set(xlabel='Subdivisions per in-plane axis',ylabel='Normal force [N]',title='Same flat interface, gap 0.015 mm')
    axes[1].plot(path['phase_fraction'],path['force_N'],'o-',label='normal force')
    axes[1].plot(path['phase_fraction'],path['tangent_resultant_N'],'o-',label='unintended tangential resultant')
    axes[1].set(xlabel='Independent mesh shift / h',ylabel='Force [N]',title='Fresh friction history at every sample')
    axes[2].plot(path['phase_fraction'],path['point_triangle_force_N'],'o-',label='point-triangle')
    axes[2].plot(path['phase_fraction'],path['edge_edge_force_N'],'o-',label='edge-edge')
    axes[2].set(xlabel='Independent mesh shift / h',ylabel='Normal force contribution [N]',title='Constraint-sum decomposition, n=1')
    for ax in axes:ax.grid(True);ax.legend(fontsize=8)
    fig.suptitle('FAILED flat-interface mesh-position invariance; not a packing compaction result')
    fig.tight_layout();fig.savefig(a.output/'plane-contact-dependence.png',dpi=170);plt.close(fig)
    fig=plt.figure(figsize=(12,8))
    for k,(n,phase) in enumerate(((1,0.),(1,.25),(4,0.),(4,.25))):
        X,t,x,H,upper=fixture(n,.015,phase);_,_,_,contact,_=evaluate(X,t,x,H,20.,.02)
        directory=a.output/f'actual-mesh-n{n}-phase{phase:g}';directory.mkdir();np.savez_compressed(directory/'mesh.npz',reference_points=X,points=x,tets=t,periodic_cell=H)
        import meshio
        meshio.write(directory/'mesh.vtu',meshio.Mesh(x,[('tetra',t)]))
        ax=fig.add_subplot(2,2,k+1,projection='3d')
        for high,color in ((False,'royalblue'),(True,'crimson')):
            faces=contact.faces[np.all(upper[contact.faces]==high,axis=1)]
            ax.add_collection3d(Line3DCollection(x[faces][:,[0,1,2,0]],colors=color,linewidths=.8))
        ax.set(xlim=(-.05,1.3),ylim=(-.05,1.15),zlim=(.397,.418),xlabel='x [mm]',ylabel='y [mm]',zlabel='z [mm]',title=f'n={n}, mesh shift={phase:g} h')
        ax.set_box_aspect((1,1,.25));ax.view_init(25,-65)
    fig.suptitle('Actual periodic interface meshes; lower blue / upper red; gap 0.015 mm\nCoordinates unscaled; axes have different scales; periodic images are omitted')
    fig.tight_layout();fig.savefig(a.output/'plane-interface-meshes.png',dpi=170);plt.close(fig)
    report={'physical_check':'FAILED: flat interface depends on mesh position','cases':len(matrix),
        'max_balance_relative':float(matrix['force_balance_relative'].max()),'max_gap_gradient_relative':float(matrix['gap_gradient_relative'].max()),
        'max_macro_work_relative':float(matrix['macro_work_relative'].max()),
        'max_phase_path_tangent_force_N':float(path['tangent_resultant_N'].max()),
        'max_initial_tangent_energy_Nmm':float(path['initial_tangent_energy_Nmm'].max()),
        'max_initial_friction_increment_Nmm':float(path['initial_friction_increment_Nmm'].max())}
    table='|n|Aligned force [N]|Shifted 0.5 h [N]|Ratio|\n|---:|---:|---:|---:|\n'
    for n in (1,2,4,8):
        r=matrix[(matrix['refinement']==n)&np.isclose(matrix['gap_mm'],gap)];f0=float(r[np.isclose(r['phase_fraction'],0)]['force_N'][0]);f1=float(r[np.isclose(r['phase_fraction'],.5)]['force_N'][0]);table+=f'|{n}|{f0:.9f}|{f1:.9f}|{f0/f1:.6f}|\n'
    (a.output/'comparison.md').write_text(table);(a.output/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    logdir=a.output/'logs';logdir.mkdir()
    for f in Path('/tmp').glob('plane-contact-*.log'):shutil.copy2(f,logdir/f.name)
    (a.output/'README.md').write_text('''# Flat periodic contact verification: FAILED physical invariance\n\nThis is a prescribed **contact-only** test of two flat periodic slab faces, each of area 1 mm². It is not the 100 mm particle packing, a material loading calculation, or a 40% to 10% compaction result. Geometry volume, gap and periodic identification are independently tested. Mu remains 0.3; every mesh phase is initialized with fresh friction history. The phase sweep is not a sliding-friction trajectory.\n\nChanging the in-plane mesh shift must leave the flat physical interface unchanged. The current fixed sum of point-triangle and edge-edge barriers fails this test. Normal force balance and energy/periodic-work derivatives pass; they do not establish physical contact consistency. The area-spring reference columns are diagnostic comparison values, not a claimed exact solution of the implemented fixed constraint sum.\n\nAll failed runs and logs are retained. Original matrix 001 precedes component decomposition; matrix 002 adds refinement 8 and force components; the dense phase path records initial friction energy. Missing columns in older records are not inferred as passed checks. Mesh NPZ/VTU files are the actual prescribed interface geometries. Production solver/material/geometry were not changed. See docs/plane-contact-verification.md in the repository for conclusions and required redesign.\n''')
    zpath=a.output.with_suffix('.zip')
    if zpath.exists():raise FileExistsError(zpath)
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
        for f in a.output.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(a.output.parent))
    with zipfile.ZipFile(zpath) as z:assert z.testzip() is None
    print(zpath)
if __name__=='__main__':main()
