"""Closest connected surface patch integration in a specified periodic image.

The caller owns image enumeration, topology exclusions and history. Facets of a
patch locate one closest surface; their potentials are never added together.
"""
import numpy as np
from .triangle_quadrature import density_batch,split,RULES


class IntegrationRefinement(ValueError):
    pass


def separation_bound(source,target):
    """Conservative support-interval distances along separating axes."""
    es=np.roll(source,-1,axis=0)-source;et=np.roll(target,-1,axis=0)-target
    ns=np.cross(es[0],es[1]);nt=np.cross(et[0],et[1])
    axes=np.r_[ns[None],nt[None],np.cross(es[:,None,:],et[None,:,:]).reshape(-1,3),np.cross(ns,es),np.cross(nt,et)]
    norms=np.linalg.norm(axes,axis=1);axes=axes[norms>1e-25]/norms[norms>1e-25,None]
    if not len(axes):raise ValueError('Degenerate contact surface')
    ps=source@axes.T;pt=target@axes.T
    gap=np.maximum(ps.min(0)-pt.max(0),pt.min(0)-ps.max(0))
    pad=128*np.finfo(float).eps*max(1.,float(abs(source).max()),float(abs(target).max()))
    return max(0.,float(gap.max())-pad)


def density_fast(bary,source,target,kn,thickness):
    point=bary@source;edges=(target[1:]-target[0]).T
    uv=np.linalg.solve(edges.T@edges,edges.T@(point-target[0]).T).T
    weights=np.c_[1-uv.sum(1),uv];interior=np.all(weights>=0,axis=1)
    best=np.full(len(point),np.inf);edge_weights=np.zeros_like(weights)
    for i,j in ((0,1),(1,2),(2,0)):
        edge=target[j]-target[i];parameter=np.clip((point-target[i])@edge/(edge@edge),0,1)
        candidate=np.zeros_like(weights);candidate[:,i]=1-parameter;candidate[:,j]=parameter
        difference=point-candidate@target;distance=np.einsum('ij,ij->i',difference,difference)
        take=distance<best;best[take]=distance[take];edge_weights[take]=candidate[take]
    weights[~interior]=edge_weights[~interior];projection=weights@target;offset=point-projection
    normal=np.cross(edges[:,0],edges[:,1]);normal/=np.linalg.norm(normal)
    offset[interior]=((point[interior]-target[0])@normal)[:,None]*normal
    gap=np.linalg.norm(offset,axis=1)
    if np.any(gap<=0):raise IntegrationRefinement('Zero patch separation')
    E=np.zeros(len(bary));Fn=np.zeros(len(bary));active=gap<thickness
    r=(gap[active]/thickness)**2;log=np.log(r)
    E[active]=-kn*thickness**2/8*(1-r)**2*log
    Fn[active]=-kn*gap[active]/4*(2*(1-r)*log-(1-r)**2/r)
    force=Fn[:,None]*offset/gap[:,None]
    values=np.c_[E,(bary[:,:,None]*force[:,None,:]).reshape(-1,9),(-weights[:,:,None]*force[:,None,:]).reshape(-1,9)]
    return values,gap,point-offset


def field(bary,source,targets,kn,thickness,vertex_map=None):
    bary=np.asarray(bary);targets=np.asarray(targets)
    results=[density_fast(bary,source,target,kn,thickness) for target in targets]
    distances=np.array([r[1] for r in results]);selected=np.argmin(distances,axis=0)
    count=len(bary)
    if vertex_map is None:vertex_map=np.arange(3*len(targets)).reshape(-1,3)
    values=np.zeros((count,10+3*(int(np.max(vertex_map))+1)))
    for k,(density,gaps,_) in enumerate(results):
        take=selected==k;values[take,:10]=density[take,:10]
        for j,node in enumerate(vertex_map[k]):values[take,10+3*node:13+3*node]+=density[take,10+3*j:13+3*j]
    # Distinct equally closest points are not silently turned into one plane.
    for k,(_,gaps,projection) in enumerate(results):
        tie=128*np.finfo(float).eps*max(1.,float(abs(source).max()))
        same=abs(gaps-distances.min(0))<=tie
        other=same&(selected!=k)&(gaps<thickness)
        if np.any(other):
            chosen_projection=np.array([results[selected[i]][2][i] for i in np.flatnonzero(other)])
            error=np.linalg.norm(projection[other]-chosen_projection,axis=1)
            if np.any(error>np.maximum(1e-10,4*np.sqrt(2*gaps[other]*tie))):raise IntegrationRefinement('Distinct tied closest patch projections require subgradient integration')
    return values,distances.min(0),selected


def grouped_field(bary,source,targets,groups,kn,thickness,vertex_map,pointwise=False):
    size=10+3*(int(np.max(vertex_map))+1);total=np.zeros((len(bary),size));distances=[];selections=[]
    for group in groups:
        group=np.asarray(group,int)
        if not pointwise:
            values,gaps,selected=field(bary,source,targets[group],kn,thickness,vertex_map[group])
            total[:,:values.shape[1]]+=values;distances.append(gaps);selections.append(group[selected]);continue
        gap_matrix=np.array([density_fast(bary,source,targets[k],kn,thickness)[1] for k in group])
        distances.append(gap_matrix.min(0));active=(gap_matrix<thickness).T
        patterns,labels=np.unique(active,axis=0,return_inverse=True)
        adjacency=np.any(vertex_map[group][:,None,:,None]==vertex_map[group][None,:,None,:],axis=(2,3))
        for label,pattern in enumerate(patterns):
            rows=np.flatnonzero(labels==label);remaining=set(map(int,np.flatnonzero(pattern)))
            while remaining:
                first=min(remaining);remaining.remove(first);connected=[first];queue=[first]
                while queue:
                    i=queue.pop();linked=sorted(j for j in remaining if adjacency[i,j])
                    remaining.difference_update(linked);connected.extend(linked);queue.extend(linked)
                facets=group[connected]
                values,_,selected=field(bary[rows],source,targets[facets],kn,thickness,vertex_map[facets])
                total[rows,:values.shape[1]]+=values
                choice=np.full(len(bary),-1,int);choice[rows]=facets[selected];selections.append(choice)
    if not distances:return total,np.full(len(bary),np.inf),[]
    return total,np.min(distances,axis=0),selections


def feature_children(B,source,targets,selected,distances):
    # Finite-element shape functions change across projected target edges.
    # These cuts only select integration cells, never exclude contact.
    candidates=[];points=B@source
    for k in np.unique(selected):
        if k<0:continue
        target=targets[int(k)];edges=(target[1:]-target[0]).T
        uv=np.linalg.solve(edges.T@edges,edges.T@(points-target[0]).T).T
        coordinates=np.c_[1-uv.sum(1),uv]
        for values in coordinates.T:
            low,high=float(min(values)),float(max(values))
            if low< -1e-10 and high>1e-10:
                candidates.append((min(-low,high)/(high-low),values))
    if not candidates:return split(B,distances)
    _,values=max(candidates,key=lambda item:item[0]);children=[]
    for sign in (1.,-1.):
        polygon=[]
        for i in range(3):
            j=(i+1)%3;a=B[i];b=B[j];va=sign*values[i];vb=sign*values[j]
            if va>=0:polygon.append(a)
            if va*vb<0:polygon.append(a+(b-a)*va/(va-vb))
        for i in range(1,len(polygon)-1):children.append(np.array([polygon[0],polygon[i],polygon[i+1]]))
    return children


def validate_partition(partition):
    """Validate a nonoverlapping cover of the reference triangle.

    Area alone cannot detect balanced holes and overlaps. A bounding-box
    sweep and convex polygon clipping check every potentially overlapping pair.
    """
    cells=np.asarray(partition,float)
    if cells.ndim!=3 or cells.shape[1:]!=(3,3) or not len(cells) or not np.isfinite(cells).all():
        raise ValueError('Invalid material integration partition')
    det=np.linalg.det(cells)
    if np.min(cells)<-1e-12 or np.max(abs(cells.sum(2)-1))>1e-12 or np.any(abs(det)<1e-25) or abs(abs(det).sum()-1)>1e-10:
        raise ValueError('Partition does not cover the reference triangle')
    xy=cells[:,:,1:];low=xy.min(1);high=xy.max(1);order=np.argsort(low[:,0]);active=[]
    def cross(a,b):return a[0]*b[1]-a[1]*b[0]
    for index in order:
        active=[j for j in active if high[j,0]>low[index,0]+1e-14]
        for j in active:
            if min(high[index,1],high[j,1])-max(low[index,1],low[j,1])<=1e-14:continue
            clip=xy[j] if det[j]>0 else xy[j,::-1];poly=list(xy[index])
            for a,b in zip(clip,np.roll(clip,-1,axis=0)):
                new=[]
                for u,v in zip(poly,poly[1:]+poly[:1]):
                    du=cross(b-a,u-a);dv=cross(b-a,v-a)
                    if du>=0:new.append(u)
                    if du*dv<0:new.append(u+(v-u)*du/(du-dv))
                poly=new
                if not poly:break
            if len(poly)>2:
                area=abs(sum(cross(u-poly[0],v-poly[0]) for u,v in zip(poly[1:],poly[2:])))/2
                if area>1e-13:raise ValueError('Overlapping material integration partition')
        active.append(index)
    return cells


def integrate(source,targets,reference_area,kn=20.,thickness=.02,tolerance=1e-3,max_depth=28,max_cells=20000,vertex_map=None,groups=None,compress_rule=True,pointwise=False,error_budget='component',initial_partition=None):
    """Adaptive reference-area rule, including points on inactive subdomains."""
    source=np.asarray(source);targets=np.asarray(targets)
    if vertex_map is None:vertex_map=np.arange(3*len(targets)).reshape(-1,3)
    size=10+3*(int(np.max(vertex_map))+1)
    if error_budget not in ('component','normalized'):raise ValueError('Unknown patch error budget')
    if groups is None:groups=[list(range(len(targets)))]
    absolute=np.r_[1e-18,np.full(size-1,1e-16)];inactive=[];cells=0;evaluations=0
    def allowance(value,relative):
        if error_budget=='component':return absolute+relative*abs(value)
        scale=max(float(np.linalg.norm(value[1:10].reshape(3,3).sum(0))),float(np.linalg.norm(value[10:].reshape(-1,3),axis=1).sum()),1e-16)
        return relative*np.r_[max(abs(value[0]),1e-18),np.full(size-1,scale)]
    def area(B):return reference_area*abs(float(np.linalg.det(B)))
    def rule(B,order):
        nonlocal evaluations
        nodes,w=RULES[order];bary=nodes@B
        values,gaps,_=grouped_field(bary,source,targets,groups,kn,thickness,vertex_map,pointwise);evaluations+=len(bary)
        return area(B)*(w@values),bary,area(B)*w,gaps
    def build(B,depth):
        nonlocal cells
        cells+=1;bound=min(separation_bound(B@source,targets[k]) for group in groups for k in group)
        if bound>=thickness:
            inactive.append(B);return None
        coarse,_,_,gaps=rule(B,4)
        _,vertex_gaps,vertex_selected=grouped_field(B,source,targets,groups,kn,thickness,vertex_map,pointwise)
        _,_,middle_selected=grouped_field(RULES[4][0]@B,source,targets,groups,kn,thickness,vertex_map,pointwise)
        selected=np.concatenate(vertex_selected+middle_selected) if vertex_selected+middle_selected else np.empty(0,int)
        children=feature_children(B,source,targets,selected,vertex_gaps);fine=np.zeros(size);samples=[];minimum=min(gaps)
        for child in children:
            value,bary,w,distances=rule(child,8);fine+=value;samples.append((bary,w,child,value));minimum=min(minimum,min(distances))
        resolved=bound>0 and fine[0]>0 and minimum<=2*bound
        if resolved:error=abs(fine-coarse)
        elif bound>0:
            from .barrier_contact import barrier
            E,F,_=barrier(bound,kn,thickness);error=(len(targets) if pointwise else len(groups))*area(B)*np.r_[E,np.full(size-1,F)]+abs(fine)
        else:error=np.full(size,np.inf)
        return dict(value=fine,error=error,children=children,samples=samples,depth=depth)
    seeds=validate_partition(initial_partition) if initial_partition is not None else np.eye(3)[None]
    leaves=[]
    for B in seeds:
        leaf=build(B,0)
        if leaf is not None:leaves.append(leaf)
    while leaves:
        total=sum((leaf['value'] for leaf in leaves),np.zeros(size));errors=np.array([leaf['error'] for leaf in leaves])
        budget=allowance(total,tolerance)
        if np.all(errors.sum(0)<=budget):break
        k=int(np.argmax(np.max(errors/budget,axis=1)));leaf=leaves.pop(k)
        if cells>=max_cells or leaf['depth']>=max_depth:raise IntegrationRefinement('Surface patch quadrature subdivision limit')
        for child in leaf['children']:
            new=build(child,leaf['depth']+1)
            if new is not None:leaves.append(new)
    else:total=np.zeros(size)
    points=[];weights=[];partition=[];compressed=0;compression_error=np.zeros(size)
    compression_budget=allowance(total,.1*tolerance)
    degree2=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
    for leaf in leaves:
        for bary,w,B,value in leaf['samples']:
            partition.append(B)
            low_bary=degree2@B;low_weights=np.full(3,area(B)/3)
            low_values,_,_=grouped_field(low_bary,source,targets,groups,kn,thickness,vertex_map,pointwise)
            low=low_weights@low_values
            error=abs(low-value)
            permitted=(np.all(compression_error+error<=compression_budget) if error_budget=='normalized' else np.all(error<=.1*tolerance*abs(value)+absolute*area(B)/reference_area))
            if compress_rule and permitted:
                points.extend(low_bary);weights.extend(low_weights);compressed+=1
                compression_error+=error
            else:points.extend(bary);weights.extend(w)
    for B in inactive:
        partition.append(B)
        if compress_rule:points.extend(degree2@B);weights.extend(np.full(3,area(B)/3))
        else:
            nodes,w=RULES[4];points.extend(nodes@B);weights.extend(area(B)*w)
    return dict(value=total,bary=np.asarray(points).reshape(-1,3),weights=np.asarray(weights),cells=cells,evaluations=evaluations,
                partition=np.asarray(partition),compressed_cells=compressed,area_error=abs(sum(weights)-reference_area))


def components(faces,shifts,X,L):
    """Connected facets in actual reference images, not merely class IDs."""
    groups=[]
    for i,(face,shift) in enumerate(zip(faces,shifts)):
        vertices=set(map(tuple,np.rint((X[face]+L*np.asarray(shift))/(L*1e-10)).astype(np.int64)))
        linked=[j for j,(keys,_) in enumerate(groups) if keys&vertices];members=[i]
        for j in reversed(linked):
            keys,old=groups.pop(j);vertices|=keys;members.extend(old)
        groups.append((vertices,members))
    return [sorted(members) for _,members in groups]
