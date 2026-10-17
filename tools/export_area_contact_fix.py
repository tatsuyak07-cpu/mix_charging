"""Collect reproducible area-contact verification and retained failed trials."""
import argparse,csv,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    for src in sorted(Path('results').glob('area-*')):
        if src.is_dir():shutil.copytree(src,a.output/src.name)
    src=Path('exports/plane-contact-verification-001/plane-contact-verification-002');shutil.copy2(src/'plane-contact.csv',a.output/'before-plane-contact.csv')
    before=np.genfromtxt(a.output/'before-plane-contact.csv',names=True,delimiter=',');after=np.genfromtxt(a.output/'area-plane-verification-006/plane-contact.csv',names=True,delimiter=',')
    arc=np.genfromtxt(a.output/'area-friction-arc-006/arc-slip.csv',names=True,delimiter=',');sliding=np.genfromtxt(a.output/'area-sliding-002/sliding.csv',names=True,delimiter=',')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for phase,label in [(0.,'old, aligned'),(.5,'old, shifted 0.5 h')]:
        row=before[np.isclose(before['gap_mm'],.015)&np.isclose(before['phase_fraction'],phase)]
        axes[0].plot(row['refinement'],row['force_N'],'o--',label=label)
    row=after[np.isclose(after['gap_mm'],.015)&np.isclose(after['phase_fraction'],0.)]
    axes[0].plot(row['refinement'],row['force_N'],'o-',label='area integration, all tested shifts');axes[0].set(xlabel='In-plane subdivisions',ylabel='Normal force [N]',title='Same plane area and 0.015 mm gap')
    axes[1].plot(sliding['slip_mm'],-sliding['tangent_force_N']/sliding['normal_force_N'],label='sliding traction / normal force');axes[1].axhline(.3,color='black',ls=':',label='Coulomb coefficient 0.3');axes[1].set(xlabel='Prescribed slip [mm]',ylabel='Force ratio',title='0.8 mm sliding across facets/images')
    axes[2].loglog(arc['intervals'],arc['relative_dissipation_error'],'o-',label='arc-slip dissipation error');axes[2].set(xlabel='Time intervals',ylabel='Relative error',title='Independent constant-radius arc reference')
    for ax in axes:ax.grid(True);ax.legend(fontsize=7)
    fig.suptitle('Restricted area contact: plane consistency and friction verification; not a particle-packing solution');fig.tight_layout();fig.savefig(a.output/'area-contact-verification.png',dpi=170);plt.close(fig)
    pressure=a.output/'area-pressure-layered-002';r=json.loads((pressure/'summary.json').read_text());h=np.genfromtxt(pressure/'history.csv',names=True,delimiter=',');reference=json.loads((pressure/'analytic-comparison.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(14,4));axes[0].plot(h['nominal_compression_MPa'],100*h['porosity']);axes[0].plot(.02,100*reference['porosity'],'kx',label='diagonal reference branch');axes[0].set(xlabel='Nominal pressure [MPa]',ylabel='Porosity [%]');axes[0].legend(fontsize=8)
    axes[1].plot(h['nominal_compression_MPa'],100*h['axial_compression']);axes[1].plot(.02,100*reference['axial_compression'],'kx');axes[1].set(xlabel='Nominal pressure [MPa]',ylabel='Axial compression [%]')
    loaded=h['time_s']>.002;axes[2].semilogy(1000*h['time_s'][loaded],np.maximum(h['kinetic_Nmm'][loaded]/np.maximum(abs(h['internal_Nmm'][loaded]-h['internal_Nmm'][0]),1e-15),1e-16));axes[2].axhline(.01,color='black',ls=':');axes[2].set(xlabel='Numerical time [ms]',ylabel='Kinetic / internal change')
    for ax in axes:ax.grid(True)
    fig.suptitle('Development-stage 1 mm layered cell, initial porosity 20%; DIAGNOSTICS ONLY\nLoaded kinetic screen FAILED; endpoint differs from diagonal reference branch');fig.tight_layout();fig.savefig(a.output/'pressure-diagnostic.png',dpi=170);plt.close(fig)
    with np.load(pressure/'last-state.npz') as m:X,x,t=m['reference_points'],m['points'],m['tets'];peak=float(m['peeq'].max());macro=m['generalized_position'][-6:]
    edges=np.unique(np.sort(np.concatenate([t[:,[i,j]] for i,j in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]]),axis=1),axis=0)
    fig=plt.figure(figsize=(11,5))
    for k,(points,title) in enumerate([(X,'Initial mesh'),(x,'Last accepted mesh at 0.02 MPa')]):
        ax=fig.add_subplot(1,2,k+1,projection='3d');ax.add_collection3d(Line3DCollection(points[edges],colors='steelblue',linewidths=.8));ax.auto_scale_xyz(points[:,0],points[:,1],points[:,2]);ax.set(xlabel='x [mm]',ylabel='y [mm]',zlabel='z [mm]',title=title);ax.set_box_aspect(np.ptp(points,axis=0))
    fig.suptitle('1 mm layered verification geometry; displacement scale 1; DIAGNOSTICS ONLY');fig.tight_layout();fig.savefig(a.output/'pressure-diagnostic-mesh.png',dpi=170);plt.close(fig)
    codes=a.output/'core-source';codes.mkdir();hashes={}
    for name in ('area_contact.py','barrier_contact.py','explicit_solver.py'):
        src=Path('packing_fem')/name;shutil.copy2(src,codes/name);hashes[name]=hashlib.sha256(src.read_bytes()).hexdigest()
    retained=a.output/'retained-designs';retained.mkdir();shutil.copy2('/tmp/area-contact-lagged-master-frame.py',retained/'lagged-master-frame.py')
    logs=a.output/'logs';logs.mkdir()
    for src in Path('/tmp').glob('area-*.log'):shutil.copy2(src,logs/src.name)
    eq=np.genfromtxt(a.output/'area-equilibrium-003/equilibrium.csv',names=True,delimiter=',')
    meta=dict(scope='restricted_area_contact_correction_not_40_to_10_packing',source_sha256=hashes,
        max_plane_force_gain_error=float(np.max(abs(after['force_gain']-1))),max_plane_macro_work_relative=float(after['macro_work_relative'].max()),
        arc_finest_dissipation_error=float(arc['relative_dissipation_error'][-1]),manufactured_force_residual_relative=float(eq['force_residual_relative'].max()),
        transient_pressure_status=r['status'],transient_peak_peeq=peak,transient_kinetic_ratio=r['kinetic_internal_ratio_loaded'],
        transient_macro_shear_xy_xz_yz=[float(v) for v in macro[3:]],transient_diagonal_reference_axial_difference=abs(r['final_axial_compression']-reference['axial_compression']),full_0_1_MPa_explicit_solution=False)
    (a.output/'verification-summary.json').write_text(json.dumps(meta,indent=2)+'\n')
    (a.output/'README.md').write_text('''# Area-contact correction and verification\n\nNew backend: `area_guarded`. Two-pass reference nodal area, one connected nearest target patch per source point. Old `distance_barrier` results remain unchanged. Edge-edge distances and linear-step non-crossing certification remain; unresolved edge-only or disjoint multi-patch contact stops for surface refinement/multi-contact treatment. There is no physical edge-only spring response or surface remeshing.\n\nFinal contact-only matrix: area-plane-verification-006 (32 cases). Final friction arc: area-friction-arc-006. Finite sliding: area-sliding-002. Coupled manufactured equilibrium: area-equilibrium-003. Manufactured states test FEM/contact residuals, not an explicitly computed pressure-loading solution.\n\nThe development-stage transient area-pressure-layered-002 uses a 1 mm layered cell with initial porosity 20%, reaches 0.02 MPa and FAILS the loaded kinetic screen. Its endpoint differs from the diagonal reference branch; nonzero macro shear means this difference alone does not diagnose a contact-force error. Its source snapshot is preserved. It is not a validated compaction curve or the final friction-frame implementation. The 0.1 MPa transient 001 was deliberately stopped as a partial-load cost trial and must not be reported as complete. No 100 mm / 40% to 10% packing solution is supplied.\n\nFailed design/test/output-serialization trials are retained; failure JSON and logs explain them. CSV uses mm, N, MPa, tonne, s; porosity fractions in history, percent on figures. Mesh PNG/VTU/NPZ use actual last accepted coordinates with displacement scale 1. Use the repository tools to reproduce; core-source is a version snapshot, not a standalone installed package.\n''')
    zpath=a.output.with_suffix('.zip')
    if zpath.exists():raise FileExistsError(zpath)
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
        for f in a.output.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(a.output.parent))
    with zipfile.ZipFile(zpath) as z:assert z.testzip() is None
    print(zpath)
if __name__=='__main__':main()
