"""Boundary-preserving TetGen remeshing with conservative T4 overlap transfer.

A remesh is accepted only if volume, reference mass and material energy pass
separate checks. Boundary triangles, hence the periodic/contact surface, stay
unchanged. This cannot repair an already degenerate boundary triangle.
"""
from itertools import combinations
import numpy as np
from scipy.spatial import cKDTree,ConvexHull,QhullError
from scipy.sparse import coo_matrix
from .explicit_material import geometry,oriented,spectral_log,response
from .adaptive_tet import boundary,quality

EDGES=np.array(list(combinations(range(4),2)))


def barycentric(points,tet):
    u=np.linalg.solve((tet[1:]-tet[0]).T,(points-tet[0]).T).T
    return np.column_stack([1-u.sum(axis=1),u])


def intersection_volume(a,b):
    ba=barycentric(a,b);ab=barycentric(b,a)
    points=[a[np.min(ba,axis=1)>=-1e-10],b[np.min(ab,axis=1)>=-1e-10]]
    for vertices,bc in ((a,ba),(b,ab)):
        first=bc[EDGES[:,0]];delta=bc[EDGES[:,1]]-first
        s=np.divide(-first,delta,out=np.full_like(first,-1),where=abs(delta)>1e-14)
        coords=first[:,None,:]+s[:,:,None]*delta[:,None,:]
        valid=(s>=0)&(s<=1)&(coords.min(axis=2)>=-1e-10)
        pos=vertices[EDGES[:,0],None,:]+s[:,:,None]*(vertices[EDGES[:,1]]-vertices[EDGES[:,0]])[:,None,:]
        points.append(pos[valid])
    points=np.concatenate(points)
    if len(points)<4:return 0.
    points=np.unique(np.round(points,11),axis=0)
    if len(points)<4:return 0.
    try:return float(ConvexHull(points).volume)
    except QhullError:return 0.


def overlap(oldx,oldt,newx,newt):
    old=oldx[oldt];new=newx[newt]
    center=old.mean(axis=1);radius=np.linalg.norm(old-center[:,None],axis=2).max(axis=1)
    tree=cKDTree(center);lo=old.min(axis=1);hi=old.max(axis=1)
    rows=[];cols=[];values=[]
    for i,tet in enumerate(new):
        middle=tet.mean(axis=0);r=np.linalg.norm(tet-middle,axis=1).max()
        ids=np.asarray(tree.query_ball_point(middle,r+radius.max()),int)
        ids=ids[np.all(hi[ids]>=tet.min(axis=0)-1e-9,axis=1)&np.all(lo[ids]<=tet.max(axis=0)+1e-9,axis=1)]
        for j in ids:
            v=intersection_volume(tet,old[j])
            if v>1e-13:rows.append(i);cols.append(j);values.append(v)
    return coo_matrix((values,(rows,cols)),shape=(len(newt),len(oldt))).tocsr()


def remesh(x,t,b,ep,V0,c,parent=None):
    import tetgen
    faces=boundary(t);nodes=np.unique(faces);lookup=np.full(len(x),-1);lookup[nodes]=np.arange(len(nodes))
    generator=tetgen.TetGen(x[nodes],lookup[faces].astype(np.int32))
    # PLC exterior is removed by TetGen. Closed internal voids need hole seeds.
    # Every outward surface seed is outside the solid; duplicates are harmless.
    tri=x[faces];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal=cross/np.linalg.norm(cross,axis=1)[:,None]
    for point in tri.mean(axis=1)+normal*1e-6:generator.add_hole(point)
    nx,nt,*_=generator.tetrahedralize(nobisect=True,minratio=1.5,mindihedral=8.,steinerleft=50000,quiet=True)
    nt=oriented(nx,nt);ov=geometry(x,t)[0];nv=geometry(nx,nt)[0]
    volume_error=abs(nv.sum()-ov.sum())/ov.sum()
    if volume_error>1e-8:raise ValueError(f'Remesh changed surface volume: {volume_error}')
    # -Y must preserve the original oriented boundary, including paired cuts.
    def signature(points,faces):return sorted(tuple(sorted(map(tuple,np.round(points[f],9)))) for f in faces)
    if signature(x,faces)!=signature(nx,boundary(nt)):raise ValueError('TetGen changed boundary triangles')
    W=overlap(x,t,nx,nt)
    row_error=float(np.max(abs(np.asarray(W.sum(axis=1)).ravel()-nv)/nv))
    col_error=float(np.max(abs(np.asarray(W.sum(axis=0)).ravel()-ov)/ov))
    if max(row_error,col_error)>1e-6:raise ValueError(f'Incomplete overlap partition {row_error}, {col_error}')
    # Preserve each constitutive subcell instead of averaging plastic history.
    # This preserves energy and all history values, at the cost of more points.
    if parent is None:parent=np.arange(len(t))
    qp=coo_matrix((V0,(parent,np.arange(len(parent)))),shape=(len(t),len(parent))).tocsr()
    transfer=(W.multiply((1/ov)[None,:])@qp).tocoo()
    newparent=transfer.row;source=transfer.col;newV0=transfer.data
    newb=b[source].copy();newep=ep[source].copy()
    oldenergy=float(V0@response(b,ep,c)[1]);newenergy=float(newV0@response(newb,newep,c)[1])
    energy_error=abs(newenergy-oldenergy)/max(abs(oldenergy),1e-12)
    report=dict(current_volume_relative_error=volume_error,reference_volume_relative_error=float(abs(newV0.sum()-V0.sum())/V0.sum()),overlap_row_error=row_error,overlap_column_error=col_error,energy_relative_error=energy_error,quality_before=float(quality(x,t).min()),quality_after=float(quality(nx,nt).min()),elements_before=len(t),elements_after=len(nt),history_points_before=len(parent),history_points_after=len(newparent),boundary_preserved=True)
    if energy_error>c['remesh_energy_tolerance']:raise ValueError(f'Remesh state transfer energy error {energy_error}; report={report}')
    gain=report['quality_after']-report['quality_before'];required=max(c.get('remesh_quality_gain_absolute_tolerance',1e-12),c.get('remesh_quality_gain_relative_tolerance',1e-8)*abs(report['quality_before']))
    report.update(quality_gain=gain,required_quality_gain=required)
    if gain<=required:raise ValueError(f'Remesh does not resolve a worst-quality improvement: {report}')
    return nx,nt,newb,newep,newV0,newparent,report


def adaptive_remesh(x,t,b,ep,V0,c,parent=None):
    """Repair a small cavity; retain all other tets and constitutive points."""
    if parent is None:parent=np.arange(len(t))
    worst=int(np.argmin(quality(x,t)));vertices=np.unique(t[worst]);failures=[]
    for ring in range(1,4):
        region=np.any(np.isin(t,vertices),axis=1);vertices=np.unique(t[region])
        owners=np.flatnonzero(region);oldnodes=np.unique(t[region]);lookup=np.full(len(x),-1);lookup[oldnodes]=np.arange(len(oldnodes))
        qpids=np.flatnonzero(region[parent]);tetmap=np.full(len(t),-1);tetmap[owners]=np.arange(len(owners))
        try:
            nx,nt,nb,ne,nv,np_local,report=remesh(x[oldnodes],lookup[t[region]],b[qpids],ep[qpids],V0[qpids],c,tetmap[parent[qpids]])
        except ValueError as exc:
            failures.append(str(exc));continue
        # Old boundary vertices are unchanged. Reuse their global IDs exactly.
        tree=cKDTree(x);dist,node=tree.query(nx);newnodes=dist>1e-9
        node[newnodes]=len(x)+np.arange(np.count_nonzero(newnodes));xx=np.r_[x,nx[newnodes]]
        tt=np.r_[t[~region],node[nt]]
        keepqp=~region[parent];keepmap=np.full(len(t),-1);keepmap[~region]=np.arange(np.count_nonzero(~region))
        pp=np.r_[keepmap[parent[keepqp]],np_local+np.count_nonzero(~region)]
        bb=np.r_[b[keepqp],nb];ee=np.r_[ep[keepqp],ne];vv=np.r_[V0[keepqp],nv]
        used=np.unique(tt);compact=np.full(len(xx),-1);compact[used]=np.arange(len(used));xx=xx[used];tt=compact[tt]
        report.update(cavity_rings=ring,cavity_elements=len(owners),global_elements_after=len(tt),global_history_points_after=len(pp))
        return xx,tt,bb,ee,vv,pp,report
    raise ValueError('No cavity repair accepted: '+'; '.join(failures))


def periodic_bisection(X,x,t,edge,L):
    edges=np.unique(np.sort(np.concatenate([t[:,[i,j]] for i,j in combinations(range(4),2)]),axis=1),axis=0)
    a,d=X[list(edge)];matches=np.zeros(len(edges),bool)
    for ordered in (edges,edges[:,::-1]):
        shift=(X[ordered[:,0]]-a)/L
        matches|=(np.max(abs(shift-np.rint(shift)),axis=1)<1e-9)&(np.max(abs(X[ordered[:,1]]-d-L*np.rint(shift)),axis=1)<L*1e-9)
    pairs=edges[matches];nx=list(x);nX=list(X);rows=[];parents=[];removed=set()
    for u,v in pairs:
        owners=np.flatnonzero(np.any(t==u,axis=1)&np.any(t==v,axis=1))
        if any(int(i) in removed for i in owners):raise ValueError('Periodic edge images share a tet')
        middle=len(nx);nx.append((x[u]+x[v])/2);nX.append((X[u]+X[v])/2)
        for owner in owners:
            rest=[int(i) for i in t[owner] if i!=u and i!=v]
            rows.extend([[u,middle,*rest],[middle,v,*rest]]);parents.extend([int(owner)]*2);removed.add(int(owner))
    keep=np.array([i for i in range(len(t)) if i not in removed],int)
    nx=np.asarray(nx);nX=np.asarray(nX);nt=oriented(nx,np.r_[t[keep],np.asarray(rows,int)])
    return dict(x=nx,X=nX,t=nt,parents=np.r_[keep,parents],report=dict(operation='periodic_edge_bisection',periodic_edges=len(pairs),new_nodes=len(nx)-len(x)))


def refine_deformed_edges(x,t,b,ep,V0,c,parent,X,L):
    """Periodic longest-edge subdivision, exact geometry and history transfer."""
    original=quality(x,t).min();events=[]
    oldmass=float(V0.sum());oldvolume=float(geometry(x,t)[0].sum());oldenergy=float(V0@response(b,ep,c)[1])
    for count in range(16):
        qualities=quality(x,t);worst=int(np.argmin(qualities))
        if qualities[worst]>=c['implicit_remesh_quality']:break
        best=None
        for pair in combinations(t[worst],2):
            try:
                proposal=periodic_bisection(X,x,t,pair,L)
                newq=quality(proposal['x'],proposal['t'])
                owners=proposal['parents'];counts=np.bincount(owners,minlength=len(t));changed=counts>1
                before=qualities[changed].min();after=newq[changed[owners]].min()
                if after<=before*1.02:continue
                if best is None or after>best[0]:best=(after,proposal,changed)
            except ValueError:continue
        if best is None:break
        after,proposal,changed=best;owners=proposal['parents']
        ratio=geometry(proposal['x'],proposal['t'])[0]/geometry(x,t)[0][owners]
        mapping=coo_matrix((ratio,(np.arange(len(owners)),owners)),shape=(len(owners),len(t))).tocsr()
        source=coo_matrix((V0,(parent,np.arange(len(parent)))),shape=(len(t),len(parent))).tocsr()
        transfer=(mapping@source).tocoo()
        b=b[transfer.col];ep=ep[transfer.col];V0=transfer.data;parent=transfer.row
        events.append(dict(edge_subdivision=proposal['report'],affected_quality_after=float(after)))
        X=proposal['X'];x=proposal['x'];t=proposal['t']
    if not events:raise ValueError('No deformed edge bisection improves the affected minimum quality')
    mass_error=abs(V0.sum()-oldmass)/oldmass;volume_error=abs(geometry(x,t)[0].sum()-oldvolume)/oldvolume
    energy_error=abs(float(V0@response(b,ep,c)[1])-oldenergy)/max(abs(oldenergy),1e-12)
    J=response(b,ep,c)[2];integrated=np.bincount(parent,weights=V0*J,minlength=len(t))
    density_error=float(np.max(abs(integrated-geometry(x,t)[0])/geometry(x,t)[0]))
    if max(mass_error,volume_error,energy_error,density_error)>1e-8:raise ValueError('Nested refinement conservation failed')
    return x,t,b,ep,V0,parent,dict(reference_volume_relative_error=float(mass_error),current_volume_relative_error=float(volume_error),energy_relative_error=float(energy_error),element_volume_history_error=density_error,operation='periodic_deformed_edge_refinement',events=events,quality_before=float(original),quality_after=float(quality(x,t).min()),boundary_geometry_preserved=True,history_values_preserved=True,history_points_after=len(parent),_reference_points=X)
