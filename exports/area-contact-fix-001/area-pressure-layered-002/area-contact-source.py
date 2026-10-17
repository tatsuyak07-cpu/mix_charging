"""Two-pass reference nodal-area contact on a unique closest surface patch.

Point/triangle candidates locate the surface; they are not additive contact
springs. Edge/edge distances remain collision guards, without an added energy.
An unresolved edge-only approach stops for surface refinement. This restricted
research backend is not general validated self-contact or surface remeshing.
"""
import numpy as np
from .barrier_contact import BarrierContact,ContactResolutionError,barrier,feature,relative
from .explicit_periodic import affine_basis


class SurfaceResolutionError(ValueError):
    """The current surface discretization cannot resolve this contact."""


def anchor_basis(z):
    edge=z[6]-z[5];edge=edge/np.sqrt(edge@edge)
    normal=np.cross(edge,z[7]-z[5]);normal=normal/np.sqrt(normal@normal)
    return np.stack([edge,np.cross(normal,edge),normal],axis=1)


def frame(z,director):
    normal=anchor_basis(z)@director
    edge=z[4]-z[0];tangent=edge-(edge@normal)*normal
    length=np.sqrt(tangent@tangent)
    if np.real(length)<1e-12:raise ContactResolutionError('Area contact friction frame is degenerate')
    tangent/=length
    return np.stack([tangent,np.cross(normal,tangent)],axis=1)


def friction_force(z,director,bary,plastic,kt):
    """Complete gradient of a material-advected, incrementally lagged frame."""
    count=3*len(z);v=np.repeat(z.astype(complex)[None],count,axis=0)
    for k in range(count):v[k,k//3,k%3]+=1j*1e-30
    dot=lambda a,b:np.einsum('ij,ij->i',a,b)
    e=v[:,6]-v[:,5];e/=np.sqrt(dot(e,e))[:,None]
    n=np.cross(e,v[:,7]-v[:,5]);n/=np.sqrt(dot(n,n))[:,None]
    normal=director[0]*e+director[1]*np.cross(n,e)+director[2]*n
    edge=v[:,4]-v[:,0];tangent=edge-dot(edge,normal)[:,None]*normal
    tangent/=np.sqrt(dot(tangent,tangent))[:,None]
    anchor=v[:,0]-np.einsum('i,nij->nj',bary,v[:,5:8])
    coordinate=np.stack([dot(tangent,anchor),dot(np.cross(normal,tangent),anchor)],axis=1)
    spring=coordinate-plastic;energy=.5*kt*dot(spring,spring)
    return -energy.imag.reshape(len(z),3)/1e-30


def transported_spring(old_normal,new_normal,spring_world):
    cosine=float(np.clip(old_normal@new_normal,-1.,1.))
    if cosine< -1+1e-8:raise ContactResolutionError('Opposed contact normals require a smaller step')
    v=np.cross(old_normal,new_normal)
    return spring_world+np.cross(v,spring_world)+np.cross(v,np.cross(v,spring_world))/(1+cosine)


class AreaContact:
    def __init__(self,X,t,periodic,kn,mu,skin,**kwargs):
        # Share the complete PT/EE distance search and non-crossing certificate.
        # No legacy constraint-sum forces or energies enter this backend.
        self.guard=BarrierContact(X,t,periodic,kn,0.,skin,**kwargs)
        if not np.isfinite(mu) or mu<0:raise ValueError("Nonnegative finite friction required")
        self.history={};self.mu=mu;self.kn=kn
        self.peers={int(node):set() for node in self.guard.nodes}
        for face in self.guard.faces:
            for node in face:self.peers[int(node)].update(int(k) for k in face if k!=node)

    def __getattr__(self,name):return getattr(self.guard,name)
    def update_topology(self,t):self.guard.update_topology(t)
    def accept_geometry(self,x,H):self.guard.accept_geometry(x,H)
    def edge_stiffness_diagonal(self):return self.guard.edge_stiffness_diagonal()

    def adjacent_targets(self,a,sa,b,sb):
        qa=np.rint((self.X[self.faces[a]]+self.L*np.asarray(sa))/self.tolerance).astype(np.int64)
        qb=np.rint((self.X[self.faces[b]]+self.L*np.asarray(sb))/self.tolerance).astype(np.int64)
        return bool(set(map(tuple,qa)).intersection(map(tuple,qb)))

    def evaluate(self,x,velocity,H,Hdot,dt):
        force=np.zeros_like(x);extra=np.zeros(6);hist={};selected={};edges=[]
        self.guard.trial_separation=self.skin;self.guard.trial_diagonal=np.zeros(len(x))
        report=dict(penetration_mm=0.,image_count=0,minimum_separation_mm=self.skin,contacts=0,edge_contacts=0,normal_energy=0.,tangent_energy=0.,friction_increment=0.,algorithmic_increment=0.,released_history_energy=0.,max_coulomb_ratio=0.,edge_guard_pairs=0,integration='two_pass_reference_nodal_area')
        if not len(self.faces):return force,extra,hist,report
        minimum,report['image_count']=self.guard.prepare_geometry(x,H)
        for key,ids,kind,shift,_,_ in self.guard.neighbors:
            if kind=='ee':edges.append((ids,shift));continue
            p=x[ids].copy();p[1:]+=H@shift
            branch,w=feature(p,'pt');r=relative(p,'pt',branch);d=float(np.linalg.norm(r))
            minimum=min(minimum,d)
            if d>=self.thickness:continue
            node=int(ids[0]);candidate=(d,key,ids,shift,p,branch,w,r)
            current=selected.get(node)
            tie_tol=64*np.finfo(float).eps*max(self.L,d)
            if current is not None and abs(d-current[0])<=tie_tol:
                projection_tol=max(self.L*1e-10,4*np.sqrt(2*max(d,current[0])*tie_tol))
                if np.linalg.norm(r-current[-1])>projection_tol:
                    raise SurfaceResolutionError('Multiple distinct closest patches require multi-contact surface integration')
            if current is None or d<current[0] or (d==current[0] and key<current[1]):selected[node]=candidate
        from .barrier_contact import segment_distances
        if edges:
            points=np.array([x[ids]+np.r_[np.zeros((2,3)),np.repeat((H@shift)[None],2,axis=0)] for ids,shift in edges])
            distances=segment_distances(points);minimum=min(minimum,float(distances.min()))
            for (ids,shift),distance in zip(edges,distances):
                if distance<self.thickness:
                    report['edge_guard_pairs']+=1
                    if not any(int(i) in selected for i in ids):
                        raise SurfaceResolutionError('Edge-only approach requires surface refinement; no artificial edge spring applied')
        if minimum<=self.L*1e-12:raise ContactResolutionError('Nonlocal surface separation unresolved')
        report['minimum_separation_mm']=minimum;self.guard.trial_separation=minimum
        for node,(d,key,ids,shift,p,branch,w,r) in selected.items():
            state=self.history.get(('area',node));face=key[2]
            if state is not None and not self.adjacent_targets(state['face'],state['shift'],face,shift):
                raise ContactResolutionError('Closest target changed to a disconnected patch; reduce step or refine surface')
            kn=.5*self.kn*self.node_area0[node];energy,fn,curvature=barrier(d,kn,self.thickness)
            normal=r/d
            if state is None:
                peers=sorted(self.peers[node]);scores=[np.linalg.norm((x[i]-x[node])-((x[i]-x[node])@normal)*normal) for i in peers]
                peer=peers[int(np.argmax(scores))]
                state=dict(peer=peer,anchor_ids=ids[1:].copy(),anchor_shift=np.asarray(shift).copy(),bary=-w[1:].copy(),plastic=np.zeros(2),spring=np.zeros(2),coordinate=np.zeros(2),energy=0.)
            support=np.r_[ids,state['peer'],state['anchor_ids']]
            images=np.zeros((8,3));images[1:4]=shift;images[5:8]=state['anchor_shift']
            z=x[support]+images@H.T
            if 'director' not in state:state=dict(state,director=anchor_basis(z).T@normal)
            local=np.zeros((8,3));local[:4]=fn*w[:,None]*normal
            kt=.5*kn;spring=np.zeros(2);plastic=state['plastic'];coordinate=state['coordinate'];diss=correction=0.
            if self.mu:
                T=frame(z,state['director']);coordinate=T.T@(z[0]-state['bary']@z[5:8]);trial=coordinate-plastic
                spring=trial*min(1.,self.mu*fn/(kt*max(np.linalg.norm(trial),1e-300)))
                plastic=coordinate-spring;diss=float(kt*spring@(plastic-state['plastic']))
                du=coordinate-state['coordinate']
                correction=float(.5*kt*(state['spring']+spring)@du-.5*kt*(spring@spring-state['spring']@state['spring'])-diss)
                if np.linalg.norm(spring)>1e-20:local+=friction_force(z,state['director'],state['bary'],plastic,kt)
            tang=.5*kt*float(spring@spring)
            np.add.at(self.guard.trial_diagonal,support,4*(curvature+fn/d+kt))
            np.add.at(force,support,local)
            extra+=np.einsum('aij,ai->j',affine_basis(self.L*images),local)
            # Update the lagged director only in returned trial history. The
            # solver commits it after accepting the step. Minimal rotation
            # preserves spring length/energy; no friction work is invented.
            director=state['director']
            if self.mu:
                old_normal=anchor_basis(z)@director
                world=transported_spring(old_normal,normal,T@spring)
                director=anchor_basis(z).T@normal
                Tnew=frame(z,director);spring=Tnew.T@world
                coordinate=Tnew.T@(z[0]-state['bary']@z[5:8]);plastic=coordinate-spring
                if abs(.5*kt*(spring@spring)-tang)>1e-10*max(tang,1e-16):raise ContactResolutionError('Friction transport energy mismatch')
            hist[('area',node)]=dict(state,director=director,plastic=plastic,spring=spring,coordinate=coordinate,energy=tang,face=face,shift=np.asarray(shift).copy())
            report['normal_energy']+=energy;report['tangent_energy']+=tang;report['friction_increment']+=diss;report['algorithmic_increment']+=correction;report['contacts']+=1
            report['max_coulomb_ratio']=max(report['max_coulomb_ratio'],float(kt*np.linalg.norm(spring)/fn))
        report['released_history_energy']=sum(v['energy'] for k,v in self.history.items() if k not in hist)
        return force,extra,hist,report
