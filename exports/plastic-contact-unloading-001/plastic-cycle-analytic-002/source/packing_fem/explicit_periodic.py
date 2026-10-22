"""Exact linear periodic kinematics and consistent reduced mass projection.

x = X + w[class(X)] + (Fbar-I) X, with symmetric six-component Fbar.
A constant mass solve enforces constraints; no tangent-stiffness/Newton solve.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def affine_basis(X):
    result=np.zeros((len(X),3,6))
    for k,(i,j) in enumerate([(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]):
        result[:,i,k]=X[:,j]
        if i!=j:result[:,j,k]=X[:,i]
    return result


def deformation(q):
    return np.eye(3)+np.array([[q[0],q[3],q[4]],[q[3],q[1],q[5]],[q[4],q[5],q[2]]])


class TranslationReducedMass:
    """Sparse mass minus a rank-three rigid-translation Schur complement."""
    def __init__(self,base,B,total):self.base=base;self.B=B;self.total=total;self.shape=base.shape
    def __matmul__(self,v):return self.base@v-self.B@(self.B.T@v)/self.total


class Periodic:
    def __init__(self,X,L,mass_frame='pinned'):
        if mass_frame not in ('pinned','zero_momentum'):raise ValueError('Unknown periodic mass frame')
        self.mass_frame=mass_frame;self.X=X;self.L=L
        canonical=np.mod(X,L);canonical[np.isclose(canonical,L,atol=1e-9,rtol=0)]=0
        keys=np.rint(canonical/(L*1e-10)).astype(np.int64)
        _,self.classes=np.unique(keys,axis=0,return_inverse=True)
        nc=int(self.classes.max())+1;self.nmicro=3*(nc-1);self.ndof=self.nmicro+6
        self.prescribed_active=True;self.prescribed=self.nmicro+2;self.free=np.delete(np.arange(self.ndof),self.prescribed)
        rows=[];cols=[];data=[]
        for node,cl in enumerate(self.classes):
            if cl:
                for axis in range(3):rows.append(3*node+axis);cols.append(3*(cl-1)+axis);data.append(1.)
        basis=affine_basis(X).reshape(-1,6)
        for row,col in zip(*np.nonzero(basis)):
            rows.append(int(row));cols.append(self.nmicro+int(col));data.append(float(basis[row,col]))
        self.A=coo_matrix((data,(rows,cols)),shape=(3*len(X),self.ndof)).tocsr()

    def positions(self,q):return self.X+np.asarray(self.A@q).reshape(-1,3)
    def velocities(self,v):
        velocity=np.asarray(self.A@v).reshape(-1,3)
        if self.mass_frame=='zero_momentum':velocity-=np.sum(self.nodal_mass[:,None]*velocity,axis=0)/self.total_mass
        return velocity
    def force(self,f,extra=None):
        result=np.asarray(self.A.T@f.ravel())
        if self.mass_frame=='zero_momentum':result-=self.B@f.sum(axis=0)/self.total_mass
        if extra is not None:result[-6:]+=extra
        return result

    def mass(self,nodal_mass):
        if np.any(nodal_mass<=0):raise ValueError('Nonpositive nodal mass')
        self.M=self.A.T@self.A.multiply(np.repeat(nodal_mass,3)[:,None])
        self.M=self.M.tocsr();self.Mff=self.M[self.free][:,self.free].tocsc()
        self.Mfp=np.asarray(self.M[self.free,self.prescribed].toarray()).ravel() if self.prescribed_active else np.zeros(len(self.free))
        try:self.factor=splu(self.Mff)
        except RuntimeError as exc:raise ValueError('Periodic mass singular; cell must span all macro directions') from exc
        if self.mass_frame=='zero_momentum':
            self.nodal_mass=np.array(nodal_mass,copy=True);self.total_mass=float(nodal_mass.sum())
            translation=np.tile(np.eye(3),(len(nodal_mass),1))
            self.B=np.asarray(self.A.T@(np.repeat(nodal_mass,3)[:,None]*translation))
            Bf=self.B[self.free];self.U=self.factor.solve(Bf)
            self.core=self.total_mass*np.eye(3)-Bf.T@self.U
            if np.linalg.eigvalsh(.5*(self.core+self.core.T)).min()<=0:raise ValueError('Translation-reduced mass is singular')
            if self.prescribed_active:self.Mfp-=Bf@self.B[self.prescribed]/self.total_mass
            self.M=TranslationReducedMass(self.M,self.B,self.total_mass)
            self.Mff=TranslationReducedMass(self.Mff,Bf,self.total_mass)

    def solve_free_mass(self,rhs):
        base=self.factor.solve(rhs)
        if self.mass_frame=='zero_momentum':base+=self.U@np.linalg.solve(self.core,self.B[self.free].T@base)
        return base

    def acceleration(self,f,v,ap,damping):
        result=np.zeros(self.ndof)
        if self.prescribed_active:result[self.prescribed]=ap
        rhs=f[self.free]-self.Mfp*ap-damping*np.asarray(self.M@v)[self.free]
        result[self.free]=self.solve_free_mass(rhs)
        return result

    def remap_velocity(self,old_momentum,v_prescribed):
        result=np.zeros(self.ndof)
        if self.prescribed_active:result[self.prescribed]=v_prescribed
        result[self.free]=self.solve_free_mass(old_momentum[self.free]-self.Mfp*v_prescribed)
        return result
