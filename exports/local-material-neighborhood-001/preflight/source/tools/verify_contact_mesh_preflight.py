"""Fixed-geometry mesh/contact preflight; never a refined pressure solution.

Interpolate an archived accepted shape, reset tangential history for a normal
contact probe, and test unloaded and mapped geometries before any cycle solve.
"""
import argparse,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
from packing_fem.explicit_material import geometry
from packing_fem.explicit_periodic import Periodic,deformation
from packing_fem.area_contact import AreaContact
from packing_fem.barrier_contact import feature,relative
from packing_fem.contact_audit import plain
from tools.run_barrier_compaction import fixture


def prolongate(X,t,x,Y):
    """Containing-tetra P1 interpolation, with an explicit outside-domain error."""
    inv=np.linalg.inv(np.stack([X[t[:,k]]-X[t[:,0]] for k in (1,2,3)],axis=2))
    owners=[];weights=[];ambiguity=0.
    for y in Y:
        v=np.einsum('tij,tj->ti',inv,y-X[t[:,0]])
        w=np.column_stack((1-v.sum(1),v));valid=np.flatnonzero(w.min(1)>=-1e-10)
        if not len(valid):raise ValueError('Refined point outside source solid mesh')
        values=np.einsum('ni,nij->nj',w[valid],x[t[valid]])
        ambiguity=max(ambiguity,float(np.max(abs(values-values[0]))))
        owners.append(valid[0]);weights.append(w[valid[0]])
    return np.einsum('ni,nij->nj',np.array(weights),x[t[np.array(owners)]]),ambiguity


def patch_diagnostic(contact,x,H):
    """Record ambiguous active candidates; do not bypass the production guard."""
    selected={}
    for key,ids,kind,shift,_,_ in contact.guard.neighbors:
        if kind!='pt':continue
        z=x[ids].copy();z[1:]+=H@shift
        branch,_=feature(z,'pt');offset=relative(z,'pt',branch);d=float(np.linalg.norm(offset))
        if d<contact.thickness:selected.setdefault(int(ids[0]),[]).append((d,key,ids,shift,z,offset))
    for node,rows in selected.items():
        groups=[]
        for row in rows:
            vertices=set(contact.target_vertices(row[1][2],row[3]));linked=[i for i,g in enumerate(groups) if g.intersection(vertices)]
            for i in reversed(linked):vertices.update(groups.pop(i))
            groups.append(vertices)
        best=min(rows,key=lambda r:(r[0],r[1]));d=best[0];tol=64*np.finfo(float).eps*max(contact.L,d)
        ptol=max(contact.L*1e-10,4*np.sqrt(2*d*tol));ties=[r for r in rows if abs(r[0]-d)<=tol]
        ambiguous=any(np.linalg.norm(r[-1]-best[-1])>ptol for r in ties)
        if len(groups)>1 or ambiguous:
            candidates=[]
            for row in rows:
                ids=row[2];ref=contact.X[ids].copy();ref[1:]+=contact.L*np.asarray(row[3]);normal=np.cross(ref[2]-ref[1],ref[3]-ref[1]);normal/=np.linalg.norm(normal)
                candidates.append(dict(distance_mm=row[0],target_face=int(row[1][2]),ids=ids.tolist(),reference_points=ref.tolist(),current_points=row[4].tolist(),offset_mm=row[-1].tolist(),reference_source_distance_to_target_plane_mm=float(abs((ref[0]-ref[1])@normal))))
            return dict(source_node=node,source_reference=contact.X[node].tolist(),source_current=x[node].tolist(),active_patch_groups=len(groups),distinct_equal_closest=ambiguous,candidates=candidates)
    return None


def probe(X,t,x,H,c):
    contact=AreaContact(X,t,Periodic(X,c['length_mm']),c['contact_penalty_mpa_mm'],c['friction'],c['contact_skin_mm'],thickness=c['barrier_thickness_mm'],max_candidates=c['contact_candidate_budget'],local_material_filter=c.get('local_material_filter',True))
    contact.capture_audit=True
    try:
        f,extra,_,r=contact.evaluate(x,x*0,H,H*0,0.)
    except ValueError as e:
        return dict(resolved=False,error_type=type(e).__name__,error=str(e),diagnostic=patch_diagnostic(contact,x,H) if not c.get('local_material_filter',True) else None,failure_feature=getattr(e,'feature',None))
    normal_force=sum(k['normal_force'] for k in contact.trial_audit['records'])
    moment=sum((np.cross(k['z'],k['local_force']).sum(0) for k in contact.trial_audit['records']),start=np.zeros(3))
    return plain(dict(resolved=True,report=r,total_normal_force_N=normal_force,contact_resultant_N=f.sum(0),unwrapped_moment_Nmm=moment,active_source_nodes=[int(k['support'][0]) for k in contact.trial_audit['records']]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=Path('exports/contact-cycle-cfl-001/fine/run'));p.add_argument('--output',type=Path,required=True);p.add_argument('--evidence',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    c=json.loads((a.source/'config.json').read_text());c.setdefault('local_material_filter',True)
    with np.load(a.source/'last-state.npz') as z:X=z['reference_points'];x=z['points'];t=z['tets'];F=deformation(z['generalized_position'][-6:])
    plan=dict(scope=__doc__,levels=[1,2,3,4,6],solver_config=c,shape_source=str(a.source),reset_tangent_history=True,plastic_history_transferred=False,pressure_trajectory_computed=False,force_or_material_changes=False,gate='All mesh levels must resolve initial and mapped geometry before claiming a refined-cycle comparison. No failed guard is bypassed.',geometry_tolerance=1e-10)
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');rows=[];meshes={};baseV=geometry(x,t)[0].sum()
    for n in plan['levels']:
        Y,T=fixture(n);y,ambiguity=prolongate(X,t,x,Y);V0=geometry(Y,T)[0];V=geometry(y,T)[0];periodic=Periodic(Y,4.)
        micro=y-Y@F.T;errors=[]
        for cl in np.unique(periodic.classes):
            group=micro[periodic.classes==cl];errors.append(float(np.max(abs(group-group[0]))))
        row=dict(refinement=n,nodes=len(Y),tetrahedra=len(T),initial_porosity=float(1-V0.sum()/64),mapped_porosity=float(1-V.sum()/(64*np.linalg.det(F))),volume_relative_difference=float(abs(V.sum()-baseV)/baseV),interpolation_shared_face_error_mm=ambiguity,periodic_displacement_error_mm=max(errors),minimum_volume_mm3=float(V.min()),initial=probe(Y,T,Y,np.eye(3)*4,c),mapped=probe(Y,T,y,F*4,c))
        row['geometry_checks']=dict(positive_elements=bool(V.min()>0),same_initial_porosity=bool(abs(row['initial_porosity']-.0025)<1e-10),same_mapped_volume=row['volume_relative_difference']<1e-10,continuous_transfer=ambiguity<1e-10,periodic_transfer=max(errors)<1e-10)
        rows.append(row);meshes[n]=(Y,T,y);np.savez_compressed(a.output/f'mesh-{n}.npz',reference_points=Y,tets=T,points=y)
        (a.output/'progress.json').write_text(json.dumps(plain(rows),indent=2)+'\n');print(n,row['initial']['resolved'],row['mapped']['resolved'],flush=True)
    resolved=[r for r in rows if r['mapped']['resolved']];energies=[r['mapped']['report']['normal_energy'] for r in resolved]
    checks=dict(geometry_preserved=all(all(r['geometry_checks'].values()) for r in rows),all_initial_geometries_resolved=all(r['initial']['resolved'] for r in rows),all_mapped_geometries_resolved=all(r['mapped']['resolved'] for r in rows))
    result=dict(scope=plan['scope'],levels=rows,checks=checks,preflight_passed=all(checks.values()),normal_energy_ratios_to_level1={str(r['refinement']):e/energies[0] for r,e in zip(resolved,energies)},pressure_trajectory_computed=False,mesh_convergence_verified=False,particle_compaction_verified=False,decision='Geometry preflight passed; pressure-cycle mesh convergence still unverified.' if all(checks.values()) else 'Contact guard still fails at refined levels; retain diagnostics before refined pressure-cycle claims.')
    (a.output/'summary.json').write_text(json.dumps(plain(result),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from packing_fem.adaptive_tet import boundary
    fig=plt.figure(figsize=(12,10))
    for index,n in enumerate((1,2)):
        ax=fig.add_subplot(2,2,index+1);Y,T,y=meshes[n];faces=boundary(T);faces=faces[np.all(abs(Y[faces,2]-2.02)<1e-10,axis=1)];faces=faces[np.all((Y[faces,:2]>=1-1e-10)&(Y[faces,:2]<=3+1e-10),axis=(1,2))]
        for face in faces:ax.plot(*Y[np.r_[face,face[0]],:2].T,color='gray',lw=.7)
        ids=np.array(rows[index]['mapped']['active_source_nodes'],int);ids=ids[abs(Y[ids,2]-2.02)<1e-10];ax.scatter(Y[ids,0],Y[ids,1],color='red',label='Active upper-wall nodes');ax.set(xlim=(.9,3.1),ylim=(.9,3.1),xlabel='Reference x (mm)',ylabel='Reference y (mm)',title=f'Refinement {n}: mapped pore-wall mesh');ax.set_aspect('equal');ax.legend(fontsize=8)
    ax=fig.add_subplot(2,2,3);ax.plot([r['refinement'] for r in resolved],energies,'o-');ax.set(xlabel='Refinement',ylabel='Normal contact energy (N mm)',title='Same geometry, fresh tangential history');ax.grid(alpha=.3)
    ax=fig.add_subplot(2,2,4,projection='3d');d=rows[3]['initial'].get('diagnostic');
    if d is None:
        old=AreaContact(meshes[4][0],meshes[4][1],Periodic(meshes[4][0],4.),c['contact_penalty_mpa_mm'],c['friction'],c['contact_skin_mm'],thickness=c['barrier_thickness_mm'],max_candidates=c['contact_candidate_budget'],local_material_filter=False)
        try:old.evaluate(meshes[4][0],meshes[4][0]*0,np.eye(3)*4,np.zeros((3,3)),0.)
        except ValueError:pass
        d=patch_diagnostic(old,meshes[4][0],np.eye(3)*4)
    targets=[np.array(v['reference_points'])[1:] for v in d['candidates']];ax.add_collection3d(Poly3DCollection(targets,alpha=.4,facecolor='steelblue',edgecolor='navy'));point=np.array(d['source_reference']);ax.scatter(*point,color='red',s=35)
    allpoints=np.concatenate(targets+[point[None]]);lo=allpoints.min(0);hi=allpoints.max(0);ax.set(xlim=(lo[0]-.02,hi[0]+.02),ylim=(lo[1]-.02,hi[1]+.02),zlim=(lo[2]-.005,hi[2]+.005),xlabel='x (mm)',ylabel='y (mm)',zlabel='z (mm)',title='Refinement 4: historical initial candidates')
    fig.suptitle(('Mesh preflight PASSED' if result['preflight_passed'] else 'Mesh preflight FAILED')+': fixed-geometry diagnostics\nNo refined pressure trajectory or packing compaction solution');fig.tight_layout();fig.savefig(a.output/'mesh-preflight.png',dpi=160);plt.close(fig)
    import csv
    with (a.output/'mesh-comparison.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['refinement','nodes','tetrahedra','initial_resolved','mapped_resolved','initial_porosity','mapped_porosity','normal_energy_Nmm','normal_force_N'])
        for r in rows:w.writerow([r[k] for k in ('refinement','nodes','tetrahedra')]+[r['initial']['resolved'],r['mapped']['resolved'],r['initial_porosity'],r['mapped_porosity'],r['mapped'].get('report',{}).get('normal_energy'),r['mapped'].get('total_normal_force_N')])
    for path in [Path(__file__),Path('tools/run_barrier_compaction.py'),*Path('packing_fem').glob('*.py')]:
        target=a.output/'source'/path.resolve().relative_to(Path.cwd());target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    for name in ('last-state.npz','config.json','summary.json'):
        target=a.output/'shape-source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a.source/name,target)
    (a.output/'README.md').write_text('Mesh/contact preflight; see summary.json for pass/fail. Diagnostic geometry probes only. Tangential history reset, plastic history not transferred. No refined pressure solution. See plan.json and summary.json. Mesh units mm, force N, energy N mm. Contact law/settings unchanged.\n')
    if a.evidence:shutil.copytree(a.evidence,a.output/'evidence')
    hashes={str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest() for f in a.output.rglob('*') if f.is_file()};(a.output/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    with zipfile.ZipFile(a.output.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as z:
        for f in a.output.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(a.output.parent))
    print(json.dumps(checks,indent=2));raise SystemExit(0 if result['preflight_passed'] else 2)
if __name__=='__main__':main()
