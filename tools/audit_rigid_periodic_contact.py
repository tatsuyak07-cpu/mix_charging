"""Replay nonlinear rigid/body contact with successively enlarged periodic tiles."""
import argparse,gc,json,hashlib
from collections import Counter
from pathlib import Path
import numpy as np
import ipctk as ipc
from packing_fem.rigid_inclusion import make_periodic,make_contact
from packing_fem.explicit_periodic import deformation
from packing_fem.ipc_periodic import free_surface
from tools.audit_rigid_compaction import load,predecessor


def stencil_catalogue(contact,normal):
    def point(index):return (int(index>=contact.body_vertices),*np.rint(contact.X[index]/(contact.p.L*1e-9)).astype(np.int64).tolist())
    def edge(index):return tuple(sorted(point(int(i)) for i in contact.edges[index]))
    result={}
    for kind in ['vv_collisions','ev_collisions','ee_collisions','fv_collisions']:
        for item in getattr(normal,kind):
            if kind=='vv_collisions':feature=tuple(sorted([point(item.vertex0_id),point(item.vertex1_id)]))
            elif kind=='ev_collisions':feature=(point(item.vertex_id),edge(item.edge_id))
            elif kind=='ee_collisions':feature=tuple(sorted([edge(item.edge0_id),edge(item.edge1_id)]))
            else:feature=(point(item.vertex_id),tuple(sorted(point(int(i)) for i in contact.faces[item.face_id])))
            result.setdefault(repr((kind,feature)),[]).append((kind,item))
    return result


def mesh_digest(contact):
    digest=hashlib.sha256()
    for array in [contact.X,contact.edges,contact.faces]:digest.update(array.tobytes())
    return digest.hexdigest()


def stencil_group_contribution(contact,q,oldq,items):
    normal=ipc.NormalCollisions();normal.use_area_weighting=True;normal.collision_set_type=ipc.NormalCollisions.IMPROVED_MAX_APPROX
    for kind in ['vv_collisions','ev_collisions','ee_collisions','fv_collisions']:setattr(normal,kind,[item for name,item in items if name==kind])
    x=contact.positions(q);R=contact.jacobian(q)
    energy=float(contact.barrier(normal,contact.mesh,x));gradient=np.asarray(R.T@contact.barrier.gradient(normal,contact.mesh,x)).ravel()
    lag=ipc.TangentialCollisions();lag.build(contact.mesh,x,normal,contact.barrier,contact.c['friction']);u=contact.displacement(q,oldq)
    friction_energy=float(contact.friction(lag,contact.mesh,u));spatial_friction=contact.friction.gradient(lag,contact.mesh,u);friction_gradient=np.asarray(R.T@spatial_friction).ravel()
    weights=[float(item.weight) for _,item in items]
    row=dict(allocated=len(items),weights=weights,weight_sum=sum(weights),normal_energy_Nmm=energy,friction_energy_Nmm=friction_energy,physical_friction_work_Nmm=float(-spatial_friction@u.ravel()),normal_generalized_gradient_max=float(abs(gradient).max()),friction_generalized_gradient_max=float(abs(friction_gradient).max()))
    row['exactly_zero']=bool(energy==0 and friction_energy==0 and not np.any(gradient) and not np.any(friction_gradient))
    row['inert']=bool(abs(energy)<1e-18 and abs(friction_energy)<1e-18 and row['normal_generalized_gradient_max']<1e-12 and row['friction_generalized_gradient_max']<1e-12)
    return row,gradient,friction_gradient


def stencil_contribution(contact,q,oldq,kind,item):
    row,_,_=stencil_group_contribution(contact,q,oldq,[(kind,item)])
    return dict(row,kind=kind,weight=float(item.weight))


def audit(run,step,out):
    run=Path(run);out=Path(out)
    if out.exists():raise FileExistsError(out)
    c=json.loads((run/'config.json').read_text());d=load(run/f'state-{step:04d}.npz')
    old,origin=predecessor(run,step,d,load(run/f'state-{step-1:04d}.npz'))
    p=make_periodic(d['X'],c['length_mm'],c);q=d['q'];inverse=np.linalg.inv(deformation(q[-6:]));records=[];forces=[]
    required=None;catalogues=[];original_stencils=[];mesh_hashes=[]
    for image_radius in [1,2]:
        contact=make_contact(d['X'],d['t'],p,c,radius=image_radius)
        x=contact.positions(q);value,force,_,count=contact.evaluate_coupled(q,old['q'])
        n=contact.last_normal;barrier=float(contact.barrier(n,contact.mesh,x));f,t=contact.rigid_resultant(q)
        catalogue=stencil_catalogue(contact,n);catalogues.append(Counter({key:len(items) for key,items in catalogue.items()}));original_stencils.append(catalogue);mesh_hashes.append(mesh_digest(contact));del catalogue
        if image_radius==1:
            # Every replicated surface has identical deformed fractional span;
            # inspect one body image and one unstrained rigid image separately.
            body=d['x'][np.unique(free_surface(d['X'],d['t'],p))]
            center,(rotation,_,_)=p.pose(q)
            rigid=contact.local[:contact.rigid_local_count]@rotation.T+center
            widths=np.ptp(np.vstack([body,rigid])@inverse.T/p.L,axis=0)
            widths+=c['ipc_dhat_mm']*np.linalg.norm(inverse,axis=1)/p.L
            required=max(1,int(np.floor(widths.max())))
            if required>1:raise ValueError('Production radius1 lacks geometric coverage; enlarged audit alone cannot repair the saved trajectory')
        displacement=contact.displacement(q,old['q']);friction_gradient=contact.friction.gradient(contact.lag,contact.mesh,displacement)
        friction_work=float(-friction_gradient@displacement.ravel());del displacement,friction_gradient
        records.append(dict(image_radius=image_radius,images=(2*image_radius+1)**3,contacts=count,rigid_contacts=contact.rigid_contact_count(),barrier_energy_Nmm=barrier,contact_potential_Nmm=float(value),physical_friction_work_Nmm=friction_work,rigid_force_N=f.tolist(),rigid_torque_Nmm=t.tolist(),intersections=bool(ipc.has_intersections(contact.mesh,x))))
        forces.append(force.copy());del contact,x,n;gc.collect()
    different={key for key in catalogues[0].keys()|catalogues[1].keys() if catalogues[0][key]!=catalogues[1][key]};group_values=[{},{}]
    for index,image_radius in enumerate([1,2]):
        if not different:continue
        contact=make_contact(d['X'],d['t'],p,c,radius=image_radius)
        if mesh_digest(contact)!=mesh_hashes[index]:raise ValueError('Rebuilt diagnostic mesh changed its indexing')
        # Retain the ORIGINAL allocated collision objects/weights, rather than
        # rebuilding a possibly differently consolidated collision list.
        for key in sorted(different):group_values[index][key]=stencil_group_contribution(contact,q,old['q'],original_stencils[index].get(key,[]))
        del contact;gc.collect()
    comparisons=[]
    for key in sorted(different):
        a,ga,fa=group_values[0][key];b,gb,fb=group_values[1][key]
        row=dict(feature=key,tiles=[a,b],weight_sum_difference=abs(a['weight_sum']-b['weight_sum']),normal_energy_difference_Nmm=abs(a['normal_energy_Nmm']-b['normal_energy_Nmm']),friction_energy_difference_Nmm=abs(a['friction_energy_Nmm']-b['friction_energy_Nmm']),physical_friction_work_difference_Nmm=abs(a['physical_friction_work_Nmm']-b['physical_friction_work_Nmm']),normal_generalized_gradient_max_difference=float(abs(ga-gb).max()),friction_generalized_gradient_max_difference=float(abs(fa-fb).max()))
        row['passed']=bool(row['weight_sum_difference']<1e-10 and max(row['normal_energy_difference_Nmm'],row['friction_energy_difference_Nmm'],row['physical_friction_work_difference_Nmm'])<1e-12 and max(row['normal_generalized_gradient_max_difference'],row['friction_generalized_gradient_max_difference'])<1e-10)
        comparisons.append(row)
    allocation_equivalent=all(row['passed'] for row in comparisons)
    difference=float(abs(forces[1]-forces[0]).max());energy_error=abs(records[1]['barrier_energy_Nmm']-records[0]['barrier_energy_Nmm'])/max(abs(records[1]['barrier_energy_Nmm']),1e-15)
    potential_error=abs(records[1]['contact_potential_Nmm']-records[0]['contact_potential_Nmm'])/max(abs(records[1]['contact_potential_Nmm']),1e-15)
    force_error=float(abs(np.asarray(records[1]['rigid_force_N'])-records[0]['rigid_force_N']).max())
    torque_error=float(abs(np.asarray(records[1]['rigid_torque_Nmm'])-records[0]['rigid_torque_Nmm']).max())
    work_error=abs(records[1]['physical_friction_work_Nmm']-records[0]['physical_friction_work_Nmm'])
    passed=not any(r['intersections'] for r in records) and allocation_equivalent and difference<1e-6 and energy_error<1e-7 and potential_error<1e-7 and force_error<1e-6 and torque_error<1e-6 and work_error<1e-10
    result=dict(passed=bool(passed),state=str(run/f'state-{step:04d}.npz'),predecessor=origin,required_relative_shift_bound=required,nonlinear_actual_displacement=True,raw_candidate_counts_equal=records[0]['contacts']==records[1]['contacts'],allocation_equivalent=bool(allocation_equivalent),allocation_group_comparisons=comparisons,generalized_force_max_difference_N=difference,barrier_energy_relative_difference=energy_error,contact_potential_relative_difference=potential_error,physical_friction_work_difference_Nmm=work_error,rigid_force_max_difference_N=force_error,rigid_torque_max_difference_Nmm=torque_error,tiles=records)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);return passed


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',required=True);a.add_argument('--step',required=True,type=int);a.add_argument('--out',required=True);args=a.parse_args()
    if not audit(args.run,args.step,args.out):raise SystemExit(1)
