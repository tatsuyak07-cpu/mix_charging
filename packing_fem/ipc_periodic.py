"""IPC Toolkit periodic surface contact, one canonical interaction per cell.

Ghost vertices share exact periodic kinematics. Coincident seam vertices are
merged so identification cuts cannot generate artificial collisions. Friction
uses IPC's lagged Coulomb incremental dissipation, not an elastic spring.
"""
from itertools import product
import numpy as np
from scipy.sparse import coo_matrix
import ipctk as ipc
from .adaptive_tet import boundary
from .explicit_periodic import affine_basis


def free_surface(X,t,p):
    faces=boundary(t)
    keys=np.sort(p.classes[faces],axis=1)
    _,labels,count=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
    keep=np.ones(len(faces),bool)
    for label in np.where(count==2)[0]:
        ids=np.where(labels==label)[0]
        delta=(X[faces[ids[1]]].mean(axis=0)-X[faces[ids[0]]].mean(axis=0))/p.L
        if np.max(abs(delta-np.rint(delta)))<1e-8 and np.any(abs(delta)>.5):keep[ids]=False
    return faces[keep]


class PeriodicIPC:
    def __init__(self,X,t,p,c,radius=1):
        self.p=p;self.c=c;L=p.L
        base_faces=free_surface(X,t,p)
        shifts=np.array(list(product(range(-radius,radius+1),repeat=3)))
        vertices=np.unique(base_faces);coords=np.concatenate([X[vertices]+s*L for s in shifts])
        _,rep,inverse=np.unique(np.rint(coords/(L*1e-9)).astype(np.int64),axis=0,return_index=True,return_inverse=True)
        self.X=coords[rep];self.images=len(shifts)
        lookup=np.full(len(X),-1);lookup[vertices]=np.arange(len(vertices))
        faces=np.concatenate([inverse[j*len(vertices)+lookup[base_faces]] for j in range(len(shifts))])
        _,unique=np.unique(np.sort(faces,axis=1),axis=0,return_index=True)
        self.faces=np.asfortranarray(faces[unique],dtype=np.int32)
        edges=np.concatenate([self.faces[:,[0,1]],self.faces[:,[1,2]],self.faces[:,[2,0]]])
        self.edges=np.asfortranarray(np.unique(np.sort(edges,axis=1),axis=0),dtype=np.int32)
        node=np.tile(vertices,len(shifts))[rep];offset=np.repeat(shifts,len(vertices),axis=0)[rep]*L
        self.R=p.A[(3*node[:,None]+np.arange(3)).ravel()].copy()
        basis=affine_basis(offset).reshape(-1,6);rr,cc=np.nonzero(basis)
        self.R+=coo_matrix((basis[rr,cc],(rr,p.nmicro+cc)),shape=self.R.shape).tocsr()
        self.mesh=ipc.CollisionMesh(np.asfortranarray(self.X),self.edges,self.faces)
        self.mesh.init_adjacencies()
        self.barrier=ipc.BarrierPotential(c['ipc_dhat_mm'],c['ipc_stiffness_mpa'],True)
        self.friction=ipc.FrictionPotential(c['ipc_slip_regularization_mm'])
        self.lag=ipc.TangentialCollisions();self.qold=np.zeros(p.ndof)

    def positions(self,q):return np.asfortranarray(self.X+(self.R@q).reshape(-1,3))

    def collisions(self,x):
        n=ipc.NormalCollisions();n.use_area_weighting=True
        n.collision_set_type=ipc.NormalCollisions.IMPROVED_MAX_APPROX
        n.build(self.mesh,x,self.c['ipc_dhat_mm'])
        # Select using rest coordinates, so ownership never jumps with deformation.
        for name in ('vv_collisions','ev_collisions','ee_collisions','fv_collisions'):
            entries=getattr(n,name)
            if not entries:continue
            ids=np.asarray([z.vertex_ids(self.edges,self.faces) for z in entries],dtype=int)
            valid=ids>=0
            center=(self.X[np.maximum(ids,0)]*valid[:,:,None]).sum(axis=1)/valid.sum(axis=1)[:,None]
            keep=np.all(center>=-1e-8,axis=1)&np.all(center<self.p.L-1e-8,axis=1)
            setattr(n,name,[entries[i] for i in np.flatnonzero(keep)])
        return n

    def set_lag(self,q,qold):
        x=self.positions(q);n=self.collisions(x)
        self.lag=ipc.TangentialCollisions();self.lag.build(self.mesh,x,n,self.barrier,self.c['friction'])
        self.qold=qold.copy()

    def evaluate(self,q,tangent=False,project_hessian=None):
        x=self.positions(q);n=self.collisions(x)
        value=self.barrier(n,self.mesh,x)
        g=self.barrier.gradient(n,self.mesh,x)
        u=np.asfortranarray((self.R@(q-self.qold)).reshape(-1,3))
        value+=self.friction(self.lag,self.mesh,u)
        g+=self.friction.gradient(self.lag,self.mesh,u)
        K=None
        if tangent:
            projected=self.c.get('ipc_project_tangent',True) if project_hessian is None else project_hessian
            projection=ipc.PSDProjectionMethod.CLAMP if projected else ipc.PSDProjectionMethod.NONE
            H=self.barrier.hessian(n,self.mesh,x,projection)
            H+=self.friction.hessian(self.lag,self.mesh,u,projection)
            K=self.R.T@H@self.R
        return value,np.asarray(self.R.T@g).ravel(),K,len(n)

    def evaluate_coupled(self,q,qold,tangent=False):
        """Self-consistent Coulomb force and its nonsymmetric position Jacobian.

        The reported energy is diagnostic; its derivative also differentiates
        normal-force-dependent dissipation and is not this equilibrium force.
        """
        x=self.positions(q);n=self.collisions(x)
        self.lag=ipc.TangentialCollisions();self.lag.build(self.mesh,x,n,self.barrier,self.c['friction']);self.qold=qold.copy()
        u=np.asfortranarray((self.R@(q-qold)).reshape(-1,3))
        value=self.barrier(n,self.mesh,x)+self.friction(self.lag,self.mesh,u)
        g=self.barrier.gradient(n,self.mesh,x)+self.friction.gradient(self.lag,self.mesh,u)
        K=None
        if tangent:
            rest=np.asfortranarray(self.X);lagged=np.asfortranarray((self.R@q).reshape(-1,3))
            H=self.barrier.hessian(n,self.mesh,x,ipc.PSDProjectionMethod.NONE)
            for wrt in [ipc.TangentialPotential.DiffWRT.LAGGED_DISPLACEMENTS,ipc.TangentialPotential.DiffWRT.VELOCITIES]:
                H-=self.friction.force_jacobian(self.lag,self.mesh,rest,lagged,u,self.barrier,wrt)
            K=self.R.T@H@self.R
        return value,np.asarray(self.R.T@g).ravel(),K,len(n)

    def ccd(self,q,dq):
        motion=(self.R@dq).reshape(-1,3)
        maximum=float(np.max(np.linalg.norm(motion,axis=1),initial=0.))
        if not np.isfinite(maximum):raise ValueError('Nonfinite CCD proposal')
        limit=float(self.c.get('ipc_ccd_maximum_displacement_mm',.1))
        if limit<=0:raise ValueError('CCD sweep limit must be positive')
        horizon=limit/maximum if maximum>limit else 1.
        self.last_ccd_horizon=horizon;self.last_ccd_max_displacement_mm=maximum
        if maximum==0:return 1.
        # A conservative horizon avoids enormous swept boxes for poor Newton /
        # quasi-Newton proposals. It bounds the query, not the physical solution.
        return horizon*ipc.compute_collision_free_stepsize(self.mesh,self.positions(q),self.positions(q+horizon*dq))
