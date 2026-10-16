"""Experimental fixed-constraint distance barrier contact.

All node-triangle and edge-edge candidates retain their own continuous distance
potential; no transfer between incompatible penalty energies. This is NOT the
published IPC solver and has no general convergence claim. Contact thickness,
quadrature weights and parallel-edge mollification require sensitivity tests.
"""
from itertools import product
import numpy as np
from scipy.spatial import cKDTree
from .periodic_contact import Contact
from .contact_geometry import closest_triangle
from .contact_tracking import closest_segments
from .explicit_periodic import affine_basis


class ContactResolutionError(ValueError):
    pass


def barrier(distance,kn,thickness):
    if distance>=thickness:return 0.,0.,0.
    if distance<=0:raise ContactResolutionError('Zero nonlocal surface separation')
    s=(distance/thickness)**2;l=np.log(s)
    bp=2*(1-s)*l-(1-s)**2/s
    bpp=-2*l+4*(1-s)/s+(1-s)**2/s**2
    return (-kn*thickness**2/8*(1-s)**2*l,
            -kn*distance/4*bp,kn/4*(bp+2*s*bpp))


def feature(points,kind):
    if kind=='pt':
        q,w=closest_triangle(points[0],points[1:]);ids=np.flatnonzero(w>1e-12)
        return tuple(int(i+1) for i in ids),np.r_[1.,-w]
    hit=closest_segments(*points);s,t=hit['s'],hit['t']
    return (int(1e-12<s<1-1e-12),int(1e-12<t<1-1e-12),int(s>=.5),int(t>=.5)),np.array([1-s,s,t-1,-t])


def segment_distances(points):
    """Vectorized independent endpoint/interior candidates for broad phase."""
    a,b,c,d=np.moveaxis(points,1,0);u=b-a;v=d-c
    uu=np.einsum('ij,ij->i',u,u);vv=np.einsum('ij,ij->i',v,v)
    uv=np.einsum('ij,ij->i',u,v);r=a-c
    ru=np.einsum('ij,ij->i',r,u);rv=np.einsum('ij,ij->i',r,v)
    distances=[]
    for fixed in (0.,1.):
        t=np.clip((rv+fixed*uv)/vv,0,1);delta=r+fixed*u-t[:,None]*v
        distances.append(np.einsum('ij,ij->i',delta,delta))
        s=np.clip((fixed*uv-ru)/uu,0,1);delta=r+s[:,None]*u-fixed*v
        distances.append(np.einsum('ij,ij->i',delta,delta))
    det=uu*vv-uv*uv;valid=det>1e-14*uu*vv
    ss=np.divide(uv*rv-vv*ru,det,out=np.zeros_like(det),where=valid)
    tt=np.divide(uu*rv-uv*ru,det,out=np.zeros_like(det),where=valid)
    delta=r+ss[:,None]*u-tt[:,None]*v;squared=np.einsum('ij,ij->i',delta,delta)
    distances.append(np.where(valid&(ss>=0)&(ss<=1)&(tt>=0)&(tt<=1),squared,np.inf))
    return np.sqrt(np.min(distances,axis=0))


def relative(points,kind,branch):
    if kind=='pt':
        ids=list(branch);a=points[ids[0]]
        if len(ids)==1:q=a
        else:
            E=(points[ids[1:]]-a).T
            q=a+E@np.linalg.solve(E.T@E,E.T@(points[0]-a))
        return points[0]-q
    free_s,free_t,fixed_s,fixed_t=branch
    a,b,c,d=points;u=b-a;v=d-c
    if free_s and free_t:
        A=np.array([[u@u,-u@v],[-u@v,v@v]])
        s,t=np.linalg.solve(A,np.array([-u@(a-c),v@(a-c)]))
    elif free_s:t=fixed_t;s=((c+t*v-a)@u)/(u@u)
    elif free_t:s=fixed_s;t=((a+s*u-c)@v)/(v@v)
    else:s,t=fixed_s,fixed_t
    return a+s*u-c-t*v


def mollifier(points,threshold):
    if threshold is None:return 1.
    cross=np.cross(points[1]-points[0],points[3]-points[2]);s=(cross@cross)/threshold
    return s*(2-s) if np.real(s)<1 else 1.


def tangent_frame(points,kind,branch,aux):
    r=relative(points,kind,branch);n=r/np.sqrt(r@r)
    edge=(points[2]-points[1]) if kind=='pt' and aux==0 else ((points[3]-points[1]) if kind=='pt' else points[2*aux+1]-points[2*aux])
    v=edge-(edge@n)*n;length=np.sqrt(v@v)
    if np.real(length)<1e-12:raise ContactResolutionError('Degenerate friction frame; reduce step or revise surface')
    t=v/length
    return np.stack([t,np.cross(n,t)],axis=1)


def tangent_gradient_batch(points,kind,branch,state,plastic,kn,threshold):
    # The twelve complex perturbations share the same real closest-feature branch.
    z=np.repeat(points.astype(complex)[None],12,axis=0)
    for k in range(12):z[k,k//3,k%3]+=1j*1e-30
    if kind=='pt':
        ids=list(branch);a=z[:,ids[0]]
        if len(ids)==1:q=a
        else:
            E=np.swapaxes(z[:,ids[1:]]-a[:,None],1,2);Et=np.swapaxes(E,1,2)
            rhs=np.einsum('nij,nj->ni',Et,z[:,0]-a)
            uv=np.linalg.solve(Et@E,rhs[:,:,None])[:,:,0]
            q=a+np.einsum('nij,nj->ni',E,uv)
        rel=z[:,0]-q
    else:
        a,b,c,d=np.moveaxis(z,1,0);u=b-a;v=d-c;r=a-c
        dot=lambda x,y:np.einsum('ij,ij->i',x,y)
        fs,ft,ss,tt=branch
        if fs and ft:
            uu=dot(u,u);vv=dot(v,v);uv=dot(u,v)
            A=np.stack([np.stack([uu,-uv],axis=1),np.stack([-uv,vv],axis=1)],axis=1)
            rhs=np.stack([-dot(u,r),dot(v,r)],axis=1)
            st=np.linalg.solve(A,rhs[:,:,None])[:,:,0];ss,tt=st[:,0],st[:,1]
        elif fs:ss=dot(c+tt*v-a,u)/dot(u,u)
        elif ft:tt=dot(a+ss*u-c,v)/dot(v,v)
        ss=np.broadcast_to(ss,(12,));tt=np.broadcast_to(tt,(12,))
        rel=a+ss[:,None]*u-c-tt[:,None]*v
    dot=lambda x,y:np.einsum('ij,ij->i',x,y)
    normal=rel/np.sqrt(dot(rel,rel))[:,None]
    aux=state['aux']
    edge=(z[:,2]-z[:,1]) if kind=='pt' and aux==0 else ((z[:,3]-z[:,1]) if kind=='pt' else z[:,2*aux+1]-z[:,2*aux])
    tangent=edge-dot(edge,normal)[:,None]*normal;tangent/=np.sqrt(dot(tangent,tangent))[:,None]
    anchor=np.einsum('i,nij->nj',state['anchor'],z)
    coordinate=np.stack([dot(tangent,anchor),dot(np.cross(normal,tangent),anchor)],axis=1)
    spring=coordinate-plastic
    m=np.ones(12,dtype=complex)
    if threshold is not None:
        cross=np.cross(z[:,1]-z[:,0],z[:,3]-z[:,2]);value=dot(cross,cross)/threshold
        m=np.where(value.real<1,value*(2-value),1.)
    energy=.25*kn*m*dot(spring,spring)
    return -energy.imag.reshape(4,3)/1e-30


def constraint(points,kind,kn,thickness,mu,old=None,threshold=None,include_inactive=False):
    branch,w=feature(points,kind);r=relative(points,kind,branch);distance=float(np.linalg.norm(r))
    energy,fn,curvature=barrier(distance,kn,thickness);m=float(mollifier(points,threshold))
    if distance>=thickness or m==0:return {'inactive':True,'distance_mm':distance} if include_inactive else None
    normal=r/distance;force=fn*m*w[:,None]*normal
    if threshold is not None and m<1:
        for i in range(4):
            for j in range(3):
                z=points.astype(complex);z[i,j]+=1j*1e-30
                force[i,j]-=energy*np.imag(mollifier(z,threshold))/1e-30
    if old is None:
        scores=[]
        for aux in (0,1):
            e=(points[2]-points[1]) if kind=='pt' and aux==0 else ((points[3]-points[1]) if kind=='pt' else points[2*aux+1]-points[2*aux])
            scores.append(float(np.linalg.norm(e-(e@normal)*normal)))
        old=dict(anchor=w.copy(),aux=int(np.argmax(scores)),plastic=np.zeros(2),spring=np.zeros(2),coordinate=np.zeros(2),energy=0.)
    kt=.5*kn*m;spring=np.zeros(2);plastic=old['plastic'];coordinate=old['coordinate'];friction=correction=0.
    if mu>0:
        T=tangent_frame(points,kind,branch,old['aux']);coordinate=T.T@(old['anchor']@points)
        trial=coordinate-old['plastic'];length=np.linalg.norm(trial)
        spring=trial*min(1.,mu*fn*m/(kt*max(length,1e-300)));plastic=coordinate-spring
        friction=float(kt*spring@(plastic-old['plastic']))
        du=coordinate-old['coordinate']
        correction=float(.5*kt*(old['spring']+spring)@du-.5*kt*(spring@spring-old['spring']@old['spring'])-friction)
        if np.linalg.norm(spring)>1e-20:
            force+=tangent_gradient_batch(points,kind,branch,old,plastic,kn,threshold)
    tang=.5*kt*float(spring@spring)
    state={**old,'spring':spring,'plastic':plastic,'coordinate':coordinate,'energy':tang}
    return dict(force=force,normal_energy=m*energy,tangent_energy=tang,state=state,
        friction_increment=friction,algorithmic_increment=correction,distance_mm=distance,
        coulomb_ratio=float(kt*np.linalg.norm(spring)/(m*fn)),stiffness=4*(curvature+fn/distance+kt))


class BarrierContact(Contact):
    def __init__(self,*args,thickness=.02,max_candidates=500000,**kwargs):
        super().__init__(*args,**kwargs)
        if not 0<thickness<=self.skin:raise ValueError('Barrier thickness must lie in (0, contact skin]')
        if not isinstance(max_candidates,int) or max_candidates<1:raise ValueError('Positive integer candidate budget required')
        self.max_candidates=max_candidates;self.search_counts={};self.neighbor_builds=0
        self.thickness=thickness;f=self.faces
        self.edges=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
        self.minimum_separation=self.skin;self.diagonal=np.zeros(len(self.X));self.trial_diagonal=self.diagonal.copy()
        self.local_edges={}

    def update_topology(self,t):
        # Interior reconnection changes neither the physical surface nor its
        # reference neighbourhood: retain the initial exclusion graph.
        if not hasattr(self,'exclude'):super().update_topology(t)

    def accept_geometry(self,x,H):
        self.accepted_geometry=(x.copy(),H.copy());self.accepted_shifts=tuple(getattr(self,'trial_shifts',[(0,0,0)]));self.minimum_separation=self.trial_separation;self.diagonal=self.trial_diagonal.copy()

    def edge_stiffness_diagonal(self):return self.diagonal

    def local_edge(self,i,j,shift):
        key=(i,j,*shift)
        if key not in self.local_edges:
            local=False
            for node in self.edges[i]:
                ids=self.edges[j];d=np.rint((self.X[ids]+np.asarray(shift)*self.L-self.X[node])/self.tolerance).astype(np.int64)
                if any((int(cl),*map(int,q)) in self.exclude[int(self.classes[node])] for cl,q in zip(self.classes[ids],d)):local=True;break
            self.local_edges[key]=local
        return self.local_edges[key]

    def build_neighbors(self,x,H,shifts):
        tri=x[self.faces];centers=tri.mean(axis=1);radius=float(np.max(np.linalg.norm(tri-centers[:,None],axis=2)))
        tree=cKDTree(np.concatenate([centers+H@s for s in shifts]));nf=len(self.faces)
        jobs=[]
        point_count=int(np.sum(tree.query_ball_point(x[self.nodes],radius+self.skin,return_length=True)))
        self.search_counts={'point_triangle_broad_candidates':point_count}
        if point_count>self.max_candidates:raise ValueError('Contact candidate budget exceeded before allocating point-triangle lists: '+str(self.search_counts))
        for node,indices in zip(self.nodes,tree.query_ball_point(x[self.nodes],radius+self.skin)):
            for index in indices:
                image,face=divmod(index,nf);shift=shifts[image]
                if self.adjacent(node,face,shift):continue
                ids=np.r_[node,self.faces[face]];points=x[ids].copy();points[1:]+=H@shift
                q,w=closest_triangle(points[0],points[1:]);distance=float(np.linalg.norm(points[0]-q))
                if distance>=self.skin:continue
                jobs.append((('pt',int(node),face,*shift),ids,'pt',shift,.5*self.kn*self.node_area0[node],None))
        edges=x[self.edges];centers=edges.mean(axis=1);radii=np.linalg.norm(edges[:,1]-edges[:,0],axis=1)/2
        ne=len(edges);tree=cKDTree(np.concatenate([centers+H@s for s in shifts]))
        edge_count=int(np.sum(tree.query_ball_point(centers,radii+radii.max()+self.skin,return_length=True)))
        self.search_counts['edge_edge_broad_candidates']=edge_count
        if point_count+edge_count>self.max_candidates:raise ValueError('Contact candidate budget exceeded before allocating edge-edge lists: '+str(self.search_counts))
        edge_candidates=[]
        for i,indices in enumerate(tree.query_ball_point(centers,radii+radii.max()+self.skin)):
            for index in indices:
                image,j=divmod(index,ne);shift=shifts[image]
                if (i,j,*shift)>(j,i,*(-np.asarray(shift))):continue
                if self.local_edge(i,j,shift):continue
                ids=np.r_[self.edges[i],self.edges[j]];points=x[ids].copy();points[2:]+=H@shift
                edge_candidates.append((i,j,shift,ids,points))
        distances=segment_distances(np.array([c[-1] for c in edge_candidates])) if edge_candidates else []
        for (i,j,shift,ids,points),distance in zip(edge_candidates,distances):
            if distance>=self.skin:continue
            ref=self.X[ids];u=ref[1]-ref[0];v=ref[3]-ref[2];threshold=1e-3*float(u@u)*float(v@v)
            jobs.append((('ee',i,j,*shift),ids,'ee',shift,.125*self.kn*self.node_area0[ids].sum(),threshold))
        return jobs

    def evaluate(self,x,velocity,H,Hdot,dt):
        force=np.zeros_like(x);extra=np.zeros(6);hist={};diag=np.zeros(len(x));minimum=self.skin
        report=dict(contacts=0,edge_contacts=0,normal_energy=0.,tangent_energy=0.,friction_increment=0.,algorithmic_increment=0.,released_history_energy=0.,penetration_mm=0.,max_coulomb_ratio=0.,image_count=0,minimum_separation_mm=self.skin)
        self.trial_separation=self.skin;self.trial_diagonal=diag
        if not len(self.faces):return force,extra,hist,report
        inv=np.linalg.inv(H);frac=x[self.nodes]@inv.T;pad=self.skin*np.linalg.norm(inv,axis=1)
        lo=np.ceil(frac.min(axis=0)-frac.max(axis=0)-pad).astype(int);hi=np.floor(frac.max(axis=0)-frac.min(axis=0)+pad).astype(int)
        count=1
        for v in hi-lo+1:count*=int(v)
        if count>729:raise ValueError('Barrier image budget exceeded; no truncation')
        shifts=list(product(*[range(a,b+1) for a,b in zip(lo,hi)]));self.trial_shifts=shifts;report['image_count']=count
        if hasattr(self,'accepted_geometry'):
            old,oldH=self.accepted_geometry
            # Include disappearing images too: their motion still belongs to
            # the accepted-to-trial linear collision certificate.
            certificate_shifts=set(shifts).union(self.accepted_shifts)
            bound=2*float(np.max(np.linalg.norm(x-old,axis=1)))+max(np.linalg.norm((H-oldH)@s) for s in certificate_shifts)
            if bound>=.5*self.minimum_separation:raise ContactResolutionError('Linear-step separation certificate requires a smaller timestep')
        bounds=(*lo,*hi)
        margin=float('inf')
        if hasattr(self,'neighbor_geometry') and bounds==self.neighbor_bounds:
            nx,nH=self.neighbor_geometry
            margin=2*float(np.max(np.linalg.norm(x-nx,axis=1)))+max(np.linalg.norm((H-nH)@q) for q in shifts)
        if margin>min(.25*self.skin,.5*(self.skin-self.thickness)):
            self.neighbors=self.build_neighbors(x,H,shifts)
            self.neighbor_geometry=(x.copy(),H.copy());self.neighbor_bounds=bounds
            self.neighbor_builds=getattr(self,'neighbor_builds',0)+1;margin=0.
        minimum=self.skin-margin
        if minimum<=self.L*1e-12:raise ContactResolutionError('Nonlocal surface separation unresolved')
        for key,ids,kind,shift,kn,threshold in self.neighbors:
            points=x[ids].copy();start=1 if kind=='pt' else 2;points[start:]+=H@shift
            value=constraint(points,kind,kn,self.thickness,self.mu,self.history.get(key),threshold,include_inactive=True)
            minimum=min(minimum,value['distance_mm'])
            if value.get('inactive',False):continue
            np.add.at(force,ids,value['force']);start=1 if kind=='pt' else 2
            master=value['force'][start:]
            extra+=np.einsum('aij,ai->j',affine_basis(np.repeat((np.asarray(shift)*self.L)[None],len(master),axis=0)),master)
            hist[key]=value['state'];np.add.at(diag,ids,value['stiffness'])
            for name in ('normal_energy','tangent_energy','friction_increment','algorithmic_increment'):report[name]+=value[name]
            report['contacts']+=1;report['edge_contacts']+=int(kind=='ee')
            report['max_coulomb_ratio']=max(report['max_coulomb_ratio'],value['coulomb_ratio'])
        report['released_history_energy']=sum(state['energy'] for key,state in self.history.items() if key not in hist)
        report['minimum_separation_mm']=minimum;self.trial_separation=minimum;self.trial_diagonal=diag
        return force,extra,hist,report
