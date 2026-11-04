"""Parent-material-face contact audit, including every refined source DOF.

Area belongs to the parent partition; child rules may be empty or carry a
non-area-matching subset. Every child is still integrated independently for
energy and source/target nodal forces. No force component is discarded.
"""
import numpy as np
from . import surface_patch
from .barrier_contact import barrier


def groups(X,faces,rules,L):
    result={};owners={}
    for key,rule in rules.items():
        if 'parent_vertices' not in rule:continue
        face=int(key);vertices=tuple(map(int,rule['parent_vertices']))
        if len(vertices)!=3 or len(set(vertices))!=3 or min(vertices)<0 or max(vertices)>=len(X):raise ValueError('Invalid material parent vertices')
        result.setdefault(vertices,[]).append(face)
    parsed={}
    for vertices,members in result.items():
        tri=X[list(vertices)];E=(tri[1:]-tri[0]).T;partition=[]
        for name in ('parent_atlas','parent_controls'):
            if any(rules[str(f)].get(name,[])!=rules[str(members[0])].get(name,[]) for f in members):raise ValueError('Parent group loses historical/control poses')
        for face in members:
            uv=(X[faces[face]]-tri[0])@np.linalg.pinv(E).T;B=np.c_[1-uv.sum(1),uv]
            if np.max(abs(B@tri-X[faces[face]]))>L*1e-10:raise ValueError('Parent group is not a reference-face partition')
            partition.append(B)
        surface_patch.validate_partition(partition)
        area=float(np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))/2)
        weight=sum(sum(rules[str(f)]['weights']) for f in members)
        if abs(weight-area)>max(1e-12,area*1e-10):raise ValueError('Parent quadrature area mismatch')
        leader=min(members);parsed[leader]=dict(vertices=vertices,faces=tuple(sorted(members)),area=area)
        for f in members:owners[f]=leader
    return parsed,owners


def certify(contact,face):
    leader=contact.parent_owner[face];group=contact.parent_groups[leader];members=group['faces']
    contact._candidates(members)
    if contact._branch_index is None:
        contact._branch_index={}
        for _,branches in contact.inner.trial_patch_branches.items():contact._branch_index.setdefault(int(branches[0][2][0]),[]).append(branches)
    child_keys={}
    for child in members:
        nodes=contact.face_points[child][0]
        actual={(v[1][2],tuple(v[3])) for node in nodes for branches in contact._branch_index.get(int(node),[]) for v in branches}
        child_keys[child]=sorted(set(contact._face_candidates.get(child,[]))|actual)
    all_keys=sorted(set(k for keys in child_keys.values() for k in keys));key=(leader,tuple(all_keys))
    if key in contact._certificates:return
    source_nodes=sorted(set(int(n) for child in members for n in contact.faces[child]));source_lookup={n:i for i,n in enumerate(source_nodes)}
    identities={(int(contact.periodic.classes[n]),*np.rint((contact.X[n]+contact.L*np.asarray(shift))/(contact.L*1e-10)).astype(np.int64)) for j,shift in all_keys for n in contact.faces[j]}
    identities=sorted(identities);identity_lookup={v:i for i,v in enumerate(identities)}
    target_map={}
    for j,shift in all_keys:
        target_map[j,shift]=[identity_lookup[(int(contact.periodic.classes[n]),*np.rint((contact.X[n]+contact.L*np.asarray(shift))/(contact.L*1e-10)).astype(np.int64))] for n in contact.faces[j]]
    ns=len(source_nodes);size=1+3*ns+3*len(identities);exact=np.zeros(size);nodes=[];weights=[];columns=[]
    EB,FB=contact._face_allocation.get(leader,(contact.absolute_energy_budget,contact.absolute_force_budget))
    for child in members:
        ids=contact.faces[child];tri=contact.X[ids];A=float(np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))/2)
        keys=child_keys[child]
        if keys:
            targets=np.array([contact._x[contact.faces[j]]+contact._H@shift for j,shift in keys]);mapping=np.array([target_map[k] for k in keys])
            components=surface_patch.components(np.array([contact.faces[j] for j,_ in keys]),[s for _,s in keys],contact.X,contact.L)
            reference=surface_patch.integrate(contact._x[ids],targets,A,kn=.5*contact.kn,thickness=contact.thickness,
                tolerance=contact.quadrature_tolerance/4,vertex_map=mapping,groups=components,pointwise=True,error_budget='normalized',
                absolute_energy_budget=contact.absolute_energy_budget/(8*len(contact.faces)),absolute_force_budget=contact.absolute_force_budget/(8*len(contact.faces)),tie_rule=contact.inner.closest_tie_rule)['value']
            exact[0]+=reference[0]
            for local,n in enumerate(ids):exact[1+3*source_lookup[int(n)]:4+3*source_lookup[int(n)]]+=reference[1+3*local:4+3*local]
            exact[1+3*ns:1+3*ns+len(reference)-10]+=reference[10:]
        point_nodes,B,w=contact.face_points[child]
        for node,bary,weight in zip(point_nodes,B,w):
            column=np.zeros(size)
            for branches in contact._branch_index.get(int(node),[]):
                E,Fn,_=barrier(branches[0][0],.5*contact.kn,contact.thickness);column[0]+=E
                for v in branches:
                    force=Fn/len(branches)*v[-1]/v[0]
                    for local,n in enumerate(ids):column[1+3*source_lookup[int(n)]:4+3*source_lookup[int(n)]]+=bary[local]*force
                    for local,n in enumerate(target_map[v[1][2],tuple(v[3])]):column[1+3*ns+3*n:4+3*ns+3*n]+=v[6][local+1]*force
            nodes.append(int(node));weights.append(weight);columns.append(column)
    matrix=np.asarray(columns).T if columns else np.empty((size,0));weights=np.asarray(weights);applied=matrix@weights
    force_scale=max(float(np.linalg.norm(exact[1:1+3*ns].reshape(-1,3).sum(0))),float(np.linalg.norm(exact[1+3*ns:].reshape(-1,3),axis=1).sum()),1e-16)
    scales=np.r_[max(abs(exact[0]),1e-18),np.full(size-1,force_scale)]
    difference=abs(applied-exact);tol=contact.quadrature_tolerance
    e=max(0.,float(difference[0]-tol*scales[0]));f=float(np.maximum(difference[1:]-tol*scales[1:],0.).sum())
    passed=e<=EB/2 and f<=FB/2
    record=dict(energy_error=float(difference[0]/scales[0]),force_error=float(max(difference[1:],default=0.)/force_scale),absolute_energy_excess_Nmm=e,absolute_force_excess_L1_N=f)
    contact._certificates[key]=record
    contact.parent_contexts[leader]=dict(nodes=np.asarray(nodes,int),weights=weights,matrix=matrix,exact=exact,scales=scales,energy_budget=EB/2,force_budget=FB/2,passed=passed,
                                        source_nodes=source_nodes,reference_area=group['area'],report=record)
    if not passed and not contact._parent_transfer_probe:
        from .area_contact import SurfaceResolutionError
        exc=SurfaceResolutionError('Parent contact audit requires conservative sample redistribution: '+str(record))
        exc.feature=dict(kind='parent_quadrature',group=leader,**record);raise exc


def source_on_child(C,pose):
    """Evaluate a retained piecewise source geometry on a material child."""
    if 'child_sources' not in pose:return C@np.asarray(pose['source'])
    if len(pose.get('source_cells',[]))!=len(pose['child_sources']):raise ValueError('Retained curved source loses material-cell lineage')
    for cell,source in zip(pose['source_cells'],pose['child_sources']):
        local=C@np.linalg.inv(np.asarray(cell))
        if np.min(local)>=-1e-10:return local@np.asarray(source)
    raise ValueError('Retained curved pose does not cover refined material child')


def historical(contact,leader,pose):
    """Fine source-node responses on an inherited pre-refinement parent pose."""
    group=contact.parent_groups[leader];vertices=list(group['vertices']);parent=contact.X[vertices];E=(parent[1:]-parent[0]).T
    source_nodes=sorted(set(int(n) for f in group['faces'] for n in contact.faces[f]));lookup={n:i for i,n in enumerate(source_nodes)}
    targets=np.asarray(pose['targets']);mapping=np.asarray(pose['mapping'],int);nt=int(mapping.max())+1;ns=len(source_nodes);size=1+3*ns+3*nt
    exact=np.zeros(size);nodes=[];weights=[];columns=[]
    for child in group['faces']:
        ids=contact.faces[child];uv=(contact.X[ids]-parent[0])@np.linalg.pinv(E).T;C=np.c_[1-uv.sum(1),uv];source=source_on_child(C,pose)
        tri=contact.X[ids];area=float(np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))/2)
        ref=surface_patch.integrate(source,targets,area,kn=.5*contact.kn,thickness=contact.thickness,tolerance=contact.quadrature_tolerance/16,
            vertex_map=mapping,groups=pose['groups'],pointwise=True,error_budget='normalized',tie_rule=pose.get('tie_rule','stop'))['value']
        exact[0]+=ref[0]
        for local,n in enumerate(ids):exact[1+3*lookup[int(n)]:4+3*lookup[int(n)]]+=ref[1+3*local:4+3*local]
        exact[1+3*ns:]+=ref[10:]
        point_nodes,B,w=contact.face_points[child]
        if len(B):
            field=surface_patch.grouped_field(B,source,targets,pose['groups'],.5*contact.kn,contact.thickness,mapping,True,pose.get('tie_rule','stop'))[0]
            for node,weight,value in zip(point_nodes,w,field):
                column=np.zeros(size);column[0]=value[0]
                for local,n in enumerate(ids):column[1+3*lookup[int(n)]:4+3*lookup[int(n)]]+=value[1+3*local:4+3*local]
                column[1+3*ns:]=value[10:];nodes.append(int(node));weights.append(weight);columns.append(column)
    force_scale=max(float(np.linalg.norm(exact[1:1+3*ns].reshape(-1,3).sum(0))),float(np.linalg.norm(exact[1+3*ns:].reshape(-1,3),axis=1).sum()),1e-16)
    return dict(nodes=np.asarray(nodes,int),weights=np.asarray(weights),matrix=np.asarray(columns).T,exact=exact,scales=np.r_[max(abs(exact[0]),1e-18),np.full(size-1,force_scale)],energy_budget=0.,force_budget=0.)
