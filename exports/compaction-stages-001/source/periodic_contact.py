"""Two-pass node-to-face penalty contact on exposed, periodic solid surfaces.

Coulomb tangential spring return map. Local topological neighbours are excluded;
remote faces of the SAME connected body are included. Edge-edge contact and
surface regeneration are not implemented; this backend is experimental.
"""
from itertools import product
import numpy as np
from scipy.spatial import cKDTree
from .adaptive_tet import boundary
from .explicit_periodic import affine_basis


class Contact:
    def __init__(self,X,t,periodic,kn,mu,skin):
        if kn<=0 or mu<0 or skin<=0:raise ValueError('Invalid contact parameters')
        self.X=X.copy();self.tolerance=periodic.L*1e-10;self._L=periodic.L;self.faces=boundary(t);self.classes=periodic.classes;self.kn=kn;self.mu=mu;self.skin=skin
        # A cell-cut face is an identification interface, not a free surface.
        artificial=np.zeros(len(self.faces),bool)
        for axis in range(3):
            cut={}
            for side in (0.,periodic.L):
                indices=np.where(np.all(np.isclose(X[self.faces,axis],side,atol=1e-8,rtol=0),axis=1))[0]
                for face in indices:
                    coords=X[self.faces[face]].copy();coords[:,axis]-=side
                    keys=np.rint(coords/(periodic.L*1e-10)).astype(np.int64)
                    key=tuple(sorted(map(tuple,keys)))
                    cut.setdefault(key,[]).append((int(face),side))
            for entries in cut.values():
                if len(entries)==2 and entries[0][1]!=entries[1][1]:
                    artificial[[entry[0] for entry in entries]]=True
        # An unpaired face on a cell plane is a real free surface and remains.
        self.faces=self.faces[~artificial];self.nodes=np.unique(self.faces)
        triangles=X[self.faces];area0=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)/2
        self.node_area0=np.zeros(len(X))
        for k in range(3):np.add.at(self.node_area0,self.faces[:,k],area0/3)
        self.update_topology(t)
        self.history={}

    def update_topology(self,t):
        self.exclude={int(cl):set() for cl in np.unique(self.classes)};self.exclusion_cache={}
        for tet in t:
            for i in tet:
                for j in tet:
                    delta=np.rint((self.X[j]-self.X[i])/self.tolerance).astype(np.int64)
                    self.exclude[int(self.classes[i])].add((int(self.classes[j]),*map(int,delta)))

    def adjacent(self,node,face,shift):
        key=(int(node),int(face),*map(int,shift))
        if key not in self.exclusion_cache:
            ids=self.faces[face];delta=np.rint((self.X[ids]+np.asarray(shift)*self.L-self.X[node])/self.tolerance).astype(np.int64)
            self.exclusion_cache[key]=any((int(cl),*map(int,d)) in self.exclude[int(self.classes[node])] for cl,d in zip(self.classes[ids],delta))
        return self.exclusion_cache[key]

    def evaluate(self,x,velocity,H,Hdot,dt):
        force=np.zeros_like(x);extra=np.zeros(6);history={};friction_loss=0.;normal_energy=0.;tangent_energy=0.;max_depth=0.;max_ratio=0.;active=0
        if not len(self.faces):return force,extra,history,{'contacts':0,'normal_energy':0.,'tangent_energy':0.,'friction_increment':0.,'released_history_energy':0.,'penetration_mm':0.,'max_coulomb_ratio':0.,'image_count':0}
        tri=x[self.faces];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(cross,axis=1)/2
        if np.any(area<1e-15):raise ValueError('Degenerate contact triangle')
        normal=cross/(2*area[:,None]);center=tri.mean(axis=1);radius=np.max(np.linalg.norm(tri-center[:,None],axis=2))
        node_normal=np.zeros_like(x);node_area=np.zeros(len(x))
        for k in range(3):np.add.at(node_normal,self.faces[:,k],cross);np.add.at(node_area,self.faces[:,k],area/3)
        norm=np.linalg.norm(node_normal,axis=1);node_normal/=np.maximum(norm[:,None],1e-30)
        invH=np.linalg.inv(H);frac=x@invH.T;pad=self.skin*np.linalg.norm(invH,axis=1)
        sf=frac[self.nodes];mf=frac[self.faces.ravel()]
        lo=np.ceil(sf.min(axis=0)-mf.max(axis=0)-pad).astype(int);hi=np.floor(sf.max(axis=0)-mf.min(axis=0)+pad).astype(int)
        shifts=np.array(list(product(*[range(a,b+1) for a,b in zip(lo,hi)])),int).reshape(-1,3)
        if len(shifts)>729:raise ValueError('Periodic contact image budget exceeded; no image truncation allowed')
        centers=np.concatenate([center+H@s for s in shifts]);tree=cKDTree(centers)
        lists=tree.query_ball_point(x[self.nodes],radius+self.skin)
        nf=len(self.faces)
        for node,candidates in zip(self.nodes,lists):
            choices=[]
            for index in candidates:
                image,face=divmod(index,nf);shift=shifts[image];ids=self.faces[face]
                if self.adjacent(node,face,shift):continue
                if node_normal[node]@normal[face]>-.05:continue
                vertices=tri[face]+H@shift;n=normal[face];gap=float((x[node]-vertices[0])@n)
                if not -self.skin<gap<0:continue
                projection=x[node]-gap*n;v0=vertices[1]-vertices[0];v1=vertices[2]-vertices[0];v2=projection-vertices[0]
                gram=np.array([[v0@v0,v0@v1],[v0@v1,v1@v1]])
                uv=np.linalg.solve(gram,np.array([v0@v2,v1@v2]));weights=np.r_[1-uv.sum(),uv]
                if weights.min()<-1e-9:continue
                choices.append((abs(gap),int(face),tuple(map(int,shift)),weights,n))
            if not choices:continue
            depth,face,shift,weights,n=min(choices,key=lambda r:(r[0],r[1],r[2]))
            ids=self.faces[face];key=(int(node),face,*shift);A=.5*self.node_area0[node]
            fn=self.kn*A*depth;kt=.5*self.kn*A
            tangent=(tri[face,1]-tri[face,0]);tangent=tangent/np.linalg.norm(tangent)
            frame=np.stack([tangent,np.cross(n,tangent),n],axis=1)
            old=self.history.get(key)
            relative=velocity[node]-weights@velocity[ids]-Hdot@np.array(shift)
            if old is not None and dt>0:relative+=depth*(n-old['normal'])/dt
            vt=relative-(relative@n)*n
            previous=np.zeros(3) if old is None else frame@old['components']
            trial=previous+vt*dt;tangential=-kt*trial;limit=self.mu*fn
            if np.linalg.norm(tangential)>limit:
                tangential*=limit/max(np.linalg.norm(tangential),1e-30)
            spring=-tangential/kt;slip=trial-spring
            friction_loss+=max(0.,float(-tangential@slip));history[key]={'components':frame.T@spring,'normal':n.copy(),'energy':.5*kt*float(spring@spring)}
            f=fn*n+tangential;force[node]+=f
            master_force=-weights[:,None]*f;np.add.at(force,ids,master_force)
            extra+=np.einsum('aij,ai->j',affine_basis(np.repeat((np.asarray(shift)*self.L)[None],3,axis=0)),master_force)
            normal_energy+=.5*self.kn*A*depth**2;tangent_energy+=.5*kt*float(spring@spring)
            max_depth=max(max_depth,depth);max_ratio=max(max_ratio,float(np.linalg.norm(tangential))/max(fn,1e-30));active+=1
        return force,extra,history,{'contacts':active,'normal_energy':normal_energy,'tangent_energy':tangent_energy,'friction_increment':friction_loss,
                                  'released_history_energy':sum(value['energy'] for key,value in self.history.items() if key not in history),'penetration_mm':max_depth,'max_coulomb_ratio':max_ratio,'image_count':len(shifts)}

    @property
    def L(self):return self._L
