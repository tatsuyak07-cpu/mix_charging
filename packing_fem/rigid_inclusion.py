"""Floating finite rigid cylinder and conservative periodic nonlinear contact.

Rigid translations and radius-scaled Euler angles are independent unknowns.
The macro lattice moves centers/images, never strains the rigid surface.
"""
from itertools import product
import numpy as np
from scipy.sparse import coo_matrix,hstack,vstack,csr_matrix
import ipctk as ipc
from .explicit_periodic import Periodic,affine_basis,deformation
from .ipc_periodic import PeriodicIPC


def rotation_derivatives(angles):
    mats=[];first=[];second=[]
    for axis,a in enumerate(angles):
        i,j=[k for k in range(3) if k!=axis]
        # Right-handed rotations; y uses the opposite cyclic order.
        sign=-1 if axis==1 else 1
        C=np.eye(3);C[i,i]=C[j,j]=np.cos(a);C[i,j]=-sign*np.sin(a);C[j,i]=sign*np.sin(a)
        D=np.zeros((3,3));D[i,i]=D[j,j]=-np.sin(a);D[i,j]=-sign*np.cos(a);D[j,i]=sign*np.cos(a)
        E=np.zeros((3,3));E[i,i]=E[j,j]=-np.cos(a);E[i,j]=sign*np.sin(a);E[j,i]=-sign*np.sin(a)
        mats.append(C);first.append(D);second.append(E)
    def multiply(replacements):
        factors=[replacements.get(k,mats[k]) for k in range(3)]
        return factors[2]@factors[1]@factors[0]
    R=multiply({});d=np.array([multiply({k:first[k]}) for k in range(3)])
    dd=np.empty((3,3,3,3))
    for i in range(3):
        for j in range(3):dd[i,j]=multiply({i:second[i]}) if i==j else multiply({i:first[i],j:first[j]})
    return R,d,dd


def cylinder_surface(c):
    r=c['rigid_diameter_mm']/2;h=c['rigid_height_mm'];n=c.get('rigid_segments',64)
    if r<=0 or h<=0 or int(n)!=n or n<8:raise ValueError('Cylinder dimensions must be positive and angular divisions an integer>=8')
    theta=np.arange(n)*2*np.pi/n;ring=np.c_[r*np.cos(theta),r*np.sin(theta)]
    vertices=np.vstack([np.c_[ring,np.full(n,-h/2)],np.c_[ring,np.full(n,h/2)],[[0,0,-h/2],[0,0,h/2]]])
    faces=[]
    for i in range(n):
        j=(i+1)%n;faces.extend([[i,j,n+j],[i,n+j,n+i],[2*n,j,i],[2*n+1,n+i,n+j]])
    return vertices,np.array(faces,int)


def rigid_volume(c):
    r=c['rigid_diameter_mm']/2;n=c.get('rigid_segments',64)
    return float(n/2*r*r*np.sin(2*np.pi/n)*c['rigid_height_mm'])


def porosities(volume,cell,c):
    if not c.get('rigid_inclusion',False):return float(1-volume/cell),float(1-volume/cell)
    vr=rigid_volume(c)
    if cell<=vr:raise ValueError('Periodic cell volume is smaller than the rigid inclusion')
    return float(1-volume/(cell-vr)),float(1-(volume+vr)/cell)


class RigidPeriodic(Periodic):
    def __init__(self,X,L,c):
        self.base=Periodic(X,L);self.__dict__.update(self.base.__dict__);self.body_nmicro=self.nmicro
        self.nmicro+=6;self.ndof+=6;self.prescribed+=6;self.free=np.delete(np.arange(self.ndof),self.prescribed)
        self.A=hstack([self.base.A[:,:self.body_nmicro],csr_matrix((3*len(X),6)),self.base.A[:,self.body_nmicro:]],format='csr')
        self.rigid_dofs=np.arange(self.body_nmicro,self.body_nmicro+6);self.rigid_radius=c['rigid_diameter_mm']/2
        self.rigid_center=np.asarray(c.get('rigid_center_mm',[L/2]*3),float)
    def remap_positions(self,x,macro,tolerance=1e-10):
        old=self.base.remap_positions(x,macro,tolerance);q=np.zeros(self.ndof);q[:self.body_nmicro]=old[:self.body_nmicro];q[-6:]=macro;return q
    def predict_loading(self,q,prescribed):
        oldH=deformation(q[-6:]);macro=q[-6:].copy();macro[2]=prescribed;newH=deformation(macro);change=newH@np.linalg.inv(oldH)
        center,_=self.pose(q);x=self.positions(q);fractional=(x-center)@np.linalg.inv(oldH).T/self.L;fractional-=np.rint(fractional);delta=fractional@(oldH*self.L).T
        outer=.45*self.L*min(np.linalg.norm(oldH,axis=0));inner=min(self.rigid_radius*np.sqrt(2)+1.4,.8*outer)
        if inner<=0 or outer<=inner:raise ValueError('Insufficient periodic space for a rigid loading predictor')
        fraction=np.clip((np.linalg.norm(delta,axis=1)-inner)/(outer-inner),0,1);weight=1-3*fraction**2+2*fraction**3
        correction=(delta@(np.eye(3)-change).T)*weight[:,None];gauge=correction[np.flatnonzero(self.classes==0)[0]]
        target=x@change.T+correction-gauge;result=self.remap_positions(target,macro);result[self.rigid_dofs]=q[self.rigid_dofs]
        result[self.rigid_dofs[:3]]=change@q[self.rigid_dofs[:3]]-gauge
        return result
    def pose(self,q):
        angles=q[self.rigid_dofs[3:]]/self.rigid_radius
        if abs(np.cos(angles[1]))<1e-5:raise ValueError('Rigid Euler chart approaches gimbal lock; rechart required')
        return deformation(q[-6:])@self.rigid_center+q[self.rigid_dofs[:3]],rotation_derivatives(angles)


def make_periodic(X,L,c):return RigidPeriodic(X,L,c) if c.get('rigid_inclusion',False) else Periodic(X,L)


def make_contact(X,t,p,c,radius=1):return RigidPeriodicIPC(X,t,p,c,radius) if c.get('rigid_inclusion',False) else PeriodicIPC(X,t,p,c,radius)


class RigidPeriodicIPC(PeriodicIPC):
    def __init__(self,X,t,p,c,radius=1):
        super().__init__(X,t,p,c,radius);self.body_vertices=len(self.X);bodyR=self.R;self.bodyR=bodyR
        local,faces=cylinder_surface(c);self.local=np.tile(local,(self.images,1));self.rigid_local_count=len(local)
        shifts=np.array(list(product(range(-radius,radius+1),repeat=3)))
        self.centers=np.repeat(p.rigid_center+shifts*p.L,len(local),axis=0)
        self.X=np.vstack([self.X,self.centers+self.local]);self.faces=np.asfortranarray(np.vstack([self.faces,np.concatenate([faces+self.body_vertices+j*len(local) for j in range(self.images)])]),dtype=np.int32)
        edges=np.concatenate([self.faces[:,[0,1]],self.faces[:,[1,2]],self.faces[:,[2,0]]]);self.edges=np.asfortranarray(np.unique(np.sort(edges,axis=1),axis=0),dtype=np.int32)
        self.mesh=ipc.CollisionMesh(np.asfortranarray(self.X),self.edges,self.faces);self.mesh.init_adjacencies();self._ccd_q=None;self._box_cache=None
        self.R=self.jacobian(np.zeros(p.ndof))
    def _normal_all(self,x,weighted=True):
        n=ipc.NormalCollisions()
        if weighted:n.use_area_weighting=True;n.collision_set_type=ipc.NormalCollisions.IMPROVED_MAX_APPROX
        cache=self._box_cache;covered=cache is not None and np.all(x>=cache[1]-1e-10) and np.all(x<=cache[2]+1e-10)
        self.last_collision_cache_hit=bool(covered)
        if covered:n.build(cache[0],self.mesh,x,self.c['ipc_dhat_mm'])
        else:n.build(self.mesh,x,self.c['ipc_dhat_mm'])
        return n
    def collisions(self,x):
        n=self._normal_all(x)
        for name in ('vv_collisions','ev_collisions','ee_collisions','fv_collisions'):
            entries=getattr(n,name)
            if not entries:continue
            ids=np.asarray([entry.vertex_ids(self.edges,self.faces) for entry in entries],int);valid=ids>=0
            center=(self.X[np.maximum(ids,0)]*valid[:,:,None]).sum(axis=1)/valid.sum(axis=1)[:,None]
            keep=np.all(center>=-1e-8,axis=1)&np.all(center<self.p.L-1e-8,axis=1)
            setattr(n,name,[entries[i] for i in np.flatnonzero(keep)])
        return n
    def positions(self,q):
        _,(rot,_,_)=self.p.pose(q)
        body=self.X[:self.body_vertices]+(self.bodyR@q).reshape(-1,3)
        rigid=self.centers@deformation(q[-6:]).T+q[self.p.rigid_dofs[:3]]+self.local@rot.T
        return np.asfortranarray(np.vstack([body,rigid]))
    def jacobian(self,q):
        p=self.p;_,(_,d,_)=p.pose(q);N=len(self.local);rows=[];cols=[];data=[]
        for i in range(3):
            rows.extend((3*np.arange(N)+i).tolist());cols.extend([int(p.rigid_dofs[i])]*N);data.extend([1.]*N)
        basis=affine_basis(self.centers).reshape(-1,6);rr,cc=np.nonzero(basis);rows.extend(rr);cols.extend(p.nmicro+cc);data.extend(basis[rr,cc])
        deriv=np.einsum('aij,nj->nia',d,self.local)/p.rigid_radius
        rr,cc=np.nonzero(deriv.reshape(-1,3));rows.extend(rr);cols.extend(p.rigid_dofs[3:][cc]);data.extend(deriv.reshape(-1,3)[rr,cc])
        return vstack([self.bodyR,coo_matrix((data,(rows,cols)),shape=(3*N,p.ndof))],format='csr')
    def displacement(self,q,old):return np.asfortranarray(self.positions(q)-self.positions(old))
    def set_lag(self,q,qold):
        self.R=self.jacobian(q);super().set_lag(q,qold)
    def evaluate(self,*args,**kwargs):
        raise NotImplementedError('Rigid contact requires the nonlinear self-consistent coupled force, not the linear lagged backend')
    def evaluate_coupled(self,q,qold,tangent=False):
        x=self.positions(q);n=self.collisions(x);self.last_normal=n;self.R=self.jacobian(q)
        self.lag=ipc.TangentialCollisions();self.lag.build(self.mesh,x,n,self.barrier,self.c['friction']);self.qold=qold.copy();u=self.displacement(q,qold)
        value=self.barrier(n,self.mesh,x)+self.friction(self.lag,self.mesh,u)
        full=self.barrier.gradient(n,self.mesh,x)+self.friction.gradient(self.lag,self.mesh,u);self.last_full_gradient=full.reshape(-1,3)
        K=None
        if tangent:
            rest=np.asfortranarray(self.X);lagged=np.asfortranarray(x-self.X)
            H=self.barrier.hessian(n,self.mesh,x,ipc.PSDProjectionMethod.NONE)
            for wrt in [ipc.TangentialPotential.DiffWRT.LAGGED_DISPLACEMENTS,ipc.TangentialPotential.DiffWRT.VELOCITIES]:H-=self.friction.force_jacobian(self.lag,self.mesh,rest,lagged,u,self.barrier,wrt)
            K=self.R.T@H@self.R
            _,(_,_,dd)=self.p.pose(q);curvature=np.einsum('ni,abij,nj->ab',self.last_full_gradient[self.body_vertices:],dd,self.local)/self.p.rigid_radius**2
            ids=self.p.rigid_dofs[3:];K+=coo_matrix((curvature.ravel(),(np.repeat(ids,3),np.tile(ids,3))),shape=K.shape).tocsr()
        return value,np.asarray(self.R.T@full).ravel(),K,len(n)
    def ccd(self,q,dq):
        start=self.positions(q);speed=(self.jacobian(q)@dq).reshape(-1,3)
        angles=dq[self.p.rigid_dofs[3:]]/self.p.rigid_radius;omega=float(abs(angles).sum());radius=float(np.linalg.norm(self.local,axis=1).max())
        # Rotation speed bound also covers cancellation in the instantaneous Jacobian.
        center_speed=np.linalg.norm(dq[self.p.rigid_dofs[:3]])+np.max(np.linalg.norm(self.centers@(deformation(dq[-6:])-np.eye(3)).T,axis=1))
        maximum=max(float(np.linalg.norm(speed,axis=1).max()),float(center_speed+radius*omega))
        if not np.isfinite(maximum) or not np.isfinite(omega):raise ValueError('Nonfinite rigid CCD proposal')
        if maximum==0:return 1.
        horizon=min(1.,self.c.get('ipc_ccd_maximum_displacement_mm',.1)/max(maximum,1e-30));curvature=radius*omega**2/8
        if curvature>0:
            if self._ccd_q is None or not np.array_equal(q,self._ccd_q):
                alln=self._normal_all(start,weighted=False)
                self._ccd_minimum=np.sqrt(alln.compute_minimum_distance(self.mesh,start)) if len(alln) else self.c['ipc_dhat_mm'];self._ccd_q=q.copy()
            gap=self._ccd_minimum
            if not np.isfinite(gap) or gap<=0:raise ValueError('Nonpositive starting contact separation')
            horizon=min(horizon,np.sqrt(.1*gap/curvature))
        margin=2*curvature*horizon**2
        self.last_ccd_horizon=horizon;self.last_ccd_max_displacement_mm=maximum
        end=self.positions(q+horizon*dq)
        if self.c.get('ipc_cache_swept_candidates',False):
            candidates=ipc.Candidates();candidates.build(self.mesh,start,end,self.c['ipc_dhat_mm']+margin+1e-8)
            deviation=margin/2
            self._box_cache=(candidates,np.minimum(start,end)-deviation,np.maximum(start,end)+deviation)
            return horizon*candidates.compute_collision_free_stepsize(self.mesh,start,end,min_distance=margin)
        return horizon*ipc.compute_collision_free_stepsize(self.mesh,start,end,min_distance=margin)
    def loading_path_clear(self,start,target):
        current=start.copy()
        for _ in range(128):
            direction=target-current
            if np.max(abs(direction))<1e-10:return True
            bound=self.ccd(current,direction);horizon=self.last_ccd_horizon
            if bound<horizon*(1-1e-9) or bound<1e-12:return False
            current+=bound*direction
        return False
    def rigid_resultant(self,q):
        _,(rotation,_,_)=self.p.pose(q);arms=self.local@rotation.T
        forces=-self.last_full_gradient[self.body_vertices:]
        return forces.sum(axis=0),np.cross(arms,forces).sum(axis=0)
    def rigid_contact_count(self):
        count=0
        for name in ['vv_collisions','ev_collisions','ee_collisions','fv_collisions']:
            for item in getattr(self.last_normal,name):
                ids=np.asarray(item.vertex_ids(self.edges,self.faces));ids=ids[ids>=0]
                count+=int(np.any(ids>=self.body_vertices) and np.any(ids<self.body_vertices))
        return count
    def force_balance_ok(self,q,tolerance):
        force,torque=self.rigid_resultant(q)
        return bool(np.max(abs(force))<tolerance and np.max(abs(torque))/self.p.rigid_radius<tolerance)
