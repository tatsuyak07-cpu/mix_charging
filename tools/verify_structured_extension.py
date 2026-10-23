"""Verify geometry-only periodic injectivity conditions and preserve contact.

Canonical tensor-grid diagnostics, not a pressure solve, contact replacement,
material model for voids, or a certificate for arbitrary packing/remeshed grids.
"""
import argparse,csv,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
from packing_fem.structured_extension import StructuredExtension
from packing_fem.material_neighborhood import MaterialNeighborhood,segment_covered
from packing_fem.explicit_material import hex_to_tet
from packing_fem.explicit_periodic import deformation,Periodic
from packing_fem.area_contact import AreaContact
from packing_fem.barrier_contact import feature
from packing_fem.sintered import mesh
from packing_fem.adaptive_tet import boundary
from packing_fem.contact_audit import plain
from tools.verify_plane_contact import fixture


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    root=Path('exports/edge-stationarity-002/preflight');old=json.loads((root/'summary.json').read_text())
    plan=dict(scope=__doc__,source=str(root),levels=[1,2,3,4,6],condition='min eig sym(A^-1 F) > 1e-10 on EVERY canonical background tet, plus exact material-node matching and periodic-boundary tolerance',macro_frame='A=H/columnwise_reference_lengths',geometric_extension='reference P1 Laplacian for missing void-node displacement; never material stiffness',margin_tolerance=1e-10,margin_sensitivity=[1e-12,1e-10,1e-8],contact_exclusion_authorized=False,pressure_integration=False)
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');rows=[];local=[];states={}
    with np.load(root/'shape-source/last-state.npz') as z:H=4*deformation(z['generalized_position'][-6:])
    for row in old['levels']:
        n=row['refinement']
        with np.load(root/f'mesh-{n}.npz') as z:X=z['reference_points'];t=z['tets'];x=z['points']
        extension=StructuredExtension(X,t);initial,_=extension.evaluate(X,np.eye(3)*4);r,data=extension.evaluate(x,H)
        V=extension.reference_volume;mask=extension.material_mask;cell=float(np.linalg.det(H));physical_current=float(V[mask]@data['current_J'][mask]);void_current=float(V[~mask]@data['current_J'][~mask]);phi=1-physical_current/cell;void_phi=void_current/cell
        rows.append(dict(refinement=n,initial=initial,mapped=r,initial_porosity=float(1-V[mask].sum()/64),mapped_porosity_from_physical=phi,mapped_porosity_from_void_extension=void_phi,volume_fraction_error=abs(phi-void_phi),same_physical_positions=bool(np.array_equal(data['points'][extension.ids],x))))
        states[n]=(X,t,x,H,extension,r,data);np.savez_compressed(a.output/f'extension-{n}.npz',**data)
        if n in (4,6):
            hit=row['mapped']['failure_feature'];ids=np.array(hit['ids']);shift=np.array(hit['shift']);m=MaterialNeighborhood(X,t,4.,.1);points=x[ids].copy();points[1:]+=H@shift;_,w=feature(points,'pt');left,right=m.endpoints(points,w)
            assert not np.any(shift)  # Archived interior-cell examples.
            local.append(dict(refinement=n,reference_material_neighbour=m.witness(('stopped',*ids),ids,'pt',shift) is not None,current_full_material_chord_covered=segment_covered(left,right,x[t]),geometry_certificate_passed=r['certificate_passed'],candidate_exclusion_authorized=False,old_guard_remains=row['mapped']['error'],feature=hit))
    controls=[]
    # Overlap while ALL physical-material tetrahedra retain positive Jacobians.
    X,t,x,G,upper=fixture(2,.015,0.);x[upper,2]-=.02;G[2,2]-=.02;overlap,data=StructuredExtension(X,t).evaluate(x,G);controls.append(dict(name='overlapping_slabs',result=overlap));np.savez_compressed(a.output/'overlapping-slabs.npz',**data)
    # An invertible periodic shear deliberately outside the sufficient bound.
    X,h,_=mesh(np.ones((2,2,2),bool),.5);t=hex_to_tet(X,h);x=X.copy();x[np.isclose(X[:,2],.5),0]+=2.;shear,data=StructuredExtension(X,t).evaluate(x,np.eye(3));controls.append(dict(name='large_invertible_shear_uncertified',result=shear));np.savez_compressed(a.output/'large-shear.npz',**data)
    # Positive margin alone is insufficient if the periodic boundary is broken.
    X,t,x,G,_,_,_=states[1];x=X.copy();x[np.isclose(X[:,0],4.),0]+=.001;mismatch,data=StructuredExtension(X,t).evaluate(x,np.eye(3)*4);controls.append(dict(name='periodic_boundary_mismatch',result=mismatch));np.savez_compressed(a.output/'boundary-mismatch.npz',**data)
    force_controls=[]
    for name in ('same_body_pore','wavy_interface'):
        if name=='same_body_pore':X,t,x,G,_,_,_=states[1];L=4.
        else:
            X,t,x,G,_=fixture(4,.015,0.);interface=np.isclose(X[:,2],.6);x[interface,2]+=.002*np.cos(2*np.pi*X[interface,0]+.23);L=1.
        periodic=Periodic(X,L);c=AreaContact(X,t,periodic,20.,.3,.1,thickness=.02,edge_stationarity_filter=True);c.capture_audit=True;first=c.evaluate(x,x*0,G,G*0,0.);relations=[]
        for rec in c.trial_audit['records']:
            ids=np.array(rec['support'][:4]);shift=np.array(rec['images'][1],int);relations.append(dict(same_raw_material_component=len(set(c.material_neighborhood.components[ids]))==1,reference_material_neighbour=c.material_neighborhood.witness(('force',*ids,*shift),ids,'pt',shift) is not None,normal_force_N=rec['normal_force']))
        certificate,data=StructuredExtension(X,t).evaluate(x,G);second=c.evaluate(x,x*0,G,G*0,0.);force_controls.append(dict(name=name,certificate=certificate,relations=relations,normal_force_sum_N=float(sum(v['normal_force_N'] for v in relations)),force_arrays_preserved=bool(np.array_equal(first[0],second[0]) and np.array_equal(first[1],second[1])),energy_preserved=first[-1]['normal_energy']==second[-1]['normal_energy'],contacts=first[-1]['contacts']))
        np.savez_compressed(a.output/f'{name}.npz',**data,physical_contact_force=first[0])
    checks=dict(all_initial_and_mapped_geometry_conditions=all(v['initial']['certificate_passed'] and v['mapped']['certificate_passed'] for v in rows),margin_sensitivity=all(v['mapped']['normalized_margin']>tol for v in rows for tol in plan['margin_sensitivity']),physical_field_and_volume_preserved=all(v['same_physical_positions'] and v['volume_fraction_error']<1e-10 for v in rows),overlap_rejected_with_material_positive_J=overlap['minimum_material_J']>0 and not overlap['certificate_passed'] and overlap['minimum_void_extension_J']<0,positive_J_shear_conservatively_uncertified=shear['minimum_extension_J']>0 and not shear['certificate_passed'],periodic_mismatch_rejected=mismatch['normalized_margin']>0 and not mismatch['certificate_passed'],reference_local_chord_failure_is_distinct=all(v['reference_material_neighbour'] and not v['current_full_material_chord_covered'] and v['geometry_certificate_passed'] for v in local),valid_contact_forces_kept=all(v['certificate']['certificate_passed'] and v['force_arrays_preserved'] and v['energy_preserved'] and v['normal_force_sum_N']>0 and not any(r['reference_material_neighbour'] for r in v['relations']) for v in force_controls),same_body_pore_still_nonlocal_contact=all(v['same_raw_material_component'] for v in force_controls[0]['relations']),no_contact_exclusion_authorized=all(not v['mapped']['candidate_exclusion_authorized'] for v in rows))
    result=dict(scope=__doc__,checks=plain(checks),geometry_checks_passed=bool(all(checks.values())),levels=rows,stopped_local_pairs=local,negative_controls=controls,contact_controls=force_controls,test_result=(a.evidence/'structured-extension-tests-002.log').read_text().strip().splitlines()[-1],geometric_void_extension_is_material=False,production_contact_modified=False,refined_pressure_cycle_verified=False,mesh_convergence_verified=False,particle_compaction_verified=False)
    (a.output/'summary.json').write_text(json.dumps(plain(result),indent=2)+'\n')
    with (a.output/'geometry-certificate.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['refinement','ghost_nodes','normalized_margin','minimum_extension_J','geometry_passed','physical_porosity','extended_void_porosity'])
        for v in rows:w.writerow([v['refinement'],v['mapped']['ghost_nodes'],v['mapped']['normalized_margin'],v['mapped']['minimum_extension_J'],v['mapped']['certificate_passed'],v['mapped_porosity_from_physical'],v['mapped_porosity_from_void_extension']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig=plt.figure(figsize=(12,9));ax=fig.add_subplot(2,2,1);ax.plot([v['refinement'] for v in rows],[v['mapped']['normalized_margin'] for v in rows],'o-');ax.axhline(0,color='black',lw=.7);ax.set(xlabel='Mesh refinement',ylabel='min eig sym(A^-1 F)',ylim=(0,.7),title='Sufficient geometry condition: positive margin');ax.grid(alpha=.3)
    ax=fig.add_subplot(2,2,2,projection='3d');X,t,x,G,extension,r,data=states[4];void_faces=boundary(extension.tets[~extension.material_mask]);tri=data['points'][void_faces];ax.add_collection3d(Poly3DCollection(tri,facecolor='steelblue',edgecolor='none',alpha=.1));hit=local[0]['feature'];points=np.array(hit['points']);_,w=feature(points,'pt');left,right=MaterialNeighborhood.endpoints(points,w);ax.plot(*np.array([left,right]).T,'ro-');allpoints=tri.reshape(-1,3);lo=allpoints.min(0);hi=allpoints.max(0);ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),xlabel='x (mm)',ylabel='y (mm)',zlabel='z (mm)',title='Mapped void boundary + stopped chord (n4)')
    ax=fig.add_subplot(2,2,3);names=['Overlap','Invertible shear','Boundary mismatch'];margins=[v['result']['normalized_margin'] for v in controls];ax.bar(names,margins,color='tab:orange');ax.axhline(0,color='black',lw=.7);ax.set(ylabel='Normalized margin',title='All three controls are NOT certified');ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.3)
    ax=fig.add_subplot(2,2,4);ax.bar(['Same-body pore','Wavy interface'],[v['normal_force_sum_N'] for v in force_controls]);ax.set(ylabel='Sum of normal quadrature forces (N)',title='Positive geometry condition does not remove force');ax.grid(axis='y',alpha=.3)
    fig.suptitle('Geometry-only extension: no material added in voids\nContact guards and refined pressure-cycle failures remain unchanged');fig.tight_layout();fig.savefig(a.output/'geometry-extension.png',dpi=160);plt.close(fig)
    shutil.copytree(a.evidence,a.output/'evidence')
    for path in [Path(__file__),Path('tests/test_structured_extension.py'),*Path('packing_fem').glob('*.py')]:
        target=a.output/'source'/path.resolve().relative_to(Path.cwd());target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    (a.output/'README.md').write_text('Geometry-only harmonic extension and numerical strong-monotonicity condition on a canonical tensor grid. Missing void nodes have NO material stiffness, mass or constitutive state. Contact forces/guards are unchanged. Positive geometric condition does NOT authorize force exclusion. No refined pressure-cycle or 40-to-10-percent packing result. Units mm, N, MPa.\n')
    hashes={str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest() for f in a.output.rglob('*') if f.is_file()};(a.output/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    with zipfile.ZipFile(a.output.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as z:
        for f in a.output.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(a.output.parent))
    with zipfile.ZipFile(a.output.with_suffix('.zip')) as z:
        if z.testzip():raise ValueError('ZIP CRC failure')
    print(json.dumps(plain(dict(checks=checks,geometry_checks_passed=result['geometry_checks_passed'])),indent=2));raise SystemExit(0 if result['geometry_checks_passed'] else 2)
if __name__=='__main__':main()
