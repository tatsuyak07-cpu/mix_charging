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


class Periodic:
    def __init__(self,X,L):
        self.X=X;self.L=L
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
    def velocities(self,v):return np.asarray(self.A@v).reshape(-1,3)
    def force(self,f,extra=None):
        result=np.asarray(self.A.T@f.ravel())
        if extra is not None:result[-6:]+=extra
        return result

    def mass(self,nodal_mass):
        if np.any(nodal_mass<=0):raise ValueError('Nonpositive nodal mass')
        self.M=self.A.T@self.A.multiply(np.repeat(nodal_mass,3)[:,None])
        self.M=self.M.tocsr();self.Mff=self.M[self.free][:,self.free].tocsc()
        self.Mfp=np.asarray(self.M[self.free,self.prescribed].toarray()).ravel() if self.prescribed_active else np.zeros(len(self.free))
        try:self.factor=splu(self.Mff)
        except RuntimeError as exc:raise ValueError('Periodic mass singular; cell must span all macro directions') from exc

    def acceleration(self,f,v,ap,damping):
        result=np.zeros(self.ndof)
        if self.prescribed_active:result[self.prescribed]=ap
        rhs=f[self.free]-self.Mfp*ap-damping*np.asarray(self.M@v)[self.free]
        result[self.free]=self.factor.solve(rhs)
        return result

    def remap_velocity(self,old_momentum,v_prescribed):
        result=np.zeros(self.ndof)
        if self.prescribed_active:result[self.prescribed]=v_prescribed
        result[self.free]=self.factor.solve(old_momentum[self.free]-self.Mfp*v_prescribed)
        return result
