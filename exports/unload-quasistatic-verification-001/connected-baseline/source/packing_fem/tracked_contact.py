"""Experimental self-contact coupling: containment, closest features, history.

Research opt-in only. Node-based quadrature still does NOT resolve general
edge-edge contact. Ambiguous history transitions stop rather than reset a spring.
"""
from itertools import product
import numpy as np
from scipy.spatial import cKDTree
from .periodic_contact import Contact
from .contact_geometry import closest_triangle
from .contact_probe import friction_return
from .contact_tracking import adjacent_patch,material_rotation,swept_edge_event
from .explicit_periodic import affine_basis


class TrackedContact(Contact):
    def update_topology(self,t):
        super().update_topology(t);self.tets=np.asarray(t)

    def accept_geometry(self,x,H):
        self.accepted_geometry=(x.copy(),H.copy())

    def handle_edge_event(self,i,j,shift,before,after,event):
        raise ValueError('Unresolved edge-edge contact: '+str(dict(edge_a=int(i),edge_b=int(j),shift=shift,**event)))

    def guard_edges(self,x,H,shifts):
        if not hasattr(self,'accepted_geometry'):return
        old,oldH=self.accepted_geometry
        edges=np.unique(np.sort(np.concatenate([self.faces[:,[0,1]],self.faces[:,[1,2]],self.faces[:,[2,0]]]),axis=1),axis=0)
        if not len(edges):return
        olde=old[edges];newe=x[edges];centers=(olde.sum(axis=1)+newe.sum(axis=1))/4
        radius=np.max(np.linalg.norm(np.concatenate([olde,newe],axis=1)-centers[:,None],axis=2),axis=1)
        shifted=np.concatenate([centers+.5*(H+oldH)@s for s in shifts]);tree=cKDTree(shifted)
        max_shift_motion=max(np.linalg.norm((H-oldH)@s)/2 for s in shifts)
        lists=tree.query_ball_point(centers,radius+radius.max()+max_shift_motion+self.tolerance)
        for i,candidates in enumerate(lists):
            for index in candidates:
                image,j=divmod(index,len(edges));shift=shifts[image]
                if (i,j,*shift)>(j,i,*(-np.asarray(shift))):continue
                if i==j and not any(shift):continue
                local=False
                for node in edges[i]:
                    ids=edges[j];d=np.rint((self.X[ids]+np.asarray(shift)*self.L-self.X[node])/self.tolerance).astype(np.int64)
                    if any((int(cl),*map(int,q)) in self.exclude[int(self.classes[node])] for cl,q in zip(self.classes[ids],d)):local=True;break
                if local:continue
                before=np.r_[olde[i],olde[j]+oldH@shift];after=np.r_[newe[i],newe[j]+H@shift]
                event=swept_edge_event(before,after,self.L*1e-9)
                if event is not None:self.handle_edge_event(i,j,shift,before,after,event)

    def evaluate(self,x,velocity,H,Hdot,dt):
        force=np.zeros_like(x);extra=np.zeros(6);history={}
        report=dict(contacts=0,normal_energy=0.,tangent_energy=0.,friction_increment=0.,
            algorithmic_increment=0.,released_history_energy=0.,penetration_mm=0.,max_coulomb_ratio=0.,image_count=0,
            transferred_pairs=0)
        if not len(self.faces):return force,extra,history,report
        invH=np.linalg.inv(H);frac=x@invH.T;pad=self.skin*np.linalg.norm(invH,axis=1)
        lo=np.ceil(frac[self.nodes].min(axis=0)-frac.max(axis=0)-pad).astype(int)
        hi=np.floor(frac[self.nodes].max(axis=0)-frac.min(axis=0)+pad).astype(int)
        count=1
        for v in hi-lo+1:count*=int(v)
        if count>729:raise ValueError('Periodic image budget exceeded; no truncation')
        shifts=list(product(*[range(a,b+1) for a,b in zip(lo,hi)]));report['image_count']=count
        self.guard_edges(x,H,shifts)
        tet=x[self.tets];D=np.stack([tet[:,i]-tet[:,0] for i in (1,2,3)],axis=-1)
        if np.any(np.linalg.det(D)<=0):raise ValueError('Nonpositive contact tetrahedron')
        invD=np.linalg.inv(D);tlo=tet.min(axis=1);thi=tet.max(axis=1)
        tri=x[self.faces];center=tri.mean(axis=1)
        cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normals=cross/np.linalg.norm(cross,axis=1)[:,None]
        node_normal=np.zeros_like(x)
        for k in range(3):np.add.at(node_normal,self.faces[:,k],cross)
        node_normal/=np.maximum(np.linalg.norm(node_normal,axis=1)[:,None],1e-30)
        radius=float(np.max(np.linalg.norm(tri-center[:,None],axis=2)))
        tree=cKDTree(np.concatenate([center+H@s for s in shifts]));nf=len(self.faces)
        by_node={key[0]:(key,state) for key,state in self.history.items()}
        for node,indices in zip(self.nodes,tree.query_ball_point(x[self.nodes],radius+self.skin)):
            inside=False
            for shift in shifts:
                p=x[node]-H@shift
                candidates=np.flatnonzero(np.all((p>=tlo-1e-12)&(p<=thi+1e-12),axis=1))
                if not len(candidates):continue
                for owner in candidates:
                    ids=self.tets[owner]
                    delta=np.rint((self.X[ids]+np.asarray(shift)*self.L-self.X[node])/self.tolerance).astype(np.int64)
                    if any((int(cl),*map(int,d)) in self.exclude[int(self.classes[node])] for cl,d in zip(self.classes[ids],delta)):continue
                    uv=invD[owner]@(p-tet[owner,0]);w=np.r_[1-uv.sum(),uv]
                    if w.min()>=-1e-12:inside=True;break
                if inside:break
            if not inside:continue
            choices=[]
            for index in indices:
                image,face=divmod(index,nf);shift=shifts[image]
                if self.adjacent(node,face,shift):continue
                if node_normal[node]@normals[face]>-.05:continue
                q,w=closest_triangle(x[node],tri[face]+H@shift)
                choices.append((float(np.sum((q-x[node])**2)),face,shift,q,w))
            if not choices:raise ValueError('Foreign solid containment without free contact surface')
            d2,face,shift,q,w=min(choices,key=lambda item:(item[0],item[1],item[2]));depth=np.sqrt(d2)
            if depth>self.skin:raise ValueError('Self-contact beyond search band')
            if depth<1e-14:continue
            n=(q-x[node])/depth;A=.5*self.node_area0[node];kn=self.kn*A;kt=.5*kn;fn=kn*depth
            key=(int(node),face,*shift);old_item=by_node.get(int(node));spring=np.zeros(3)
            if old_item is not None:
                old_key,old=old_item
                if not adjacent_patch(self.X,self.faces,self.L,old_key[1],old_key[2:],face,shift):
                    raise ValueError('Ambiguous contact patch transition; history transfer refused')
                anchor=old['weights']@(x[self.faces[old_key[1]]]+H@np.asarray(old_key[2:]))
                triangle=x[self.faces[old_key[1]]]+H@np.asarray(old_key[2:])
                R=material_rotation(old['triangle'],triangle,old['normal'],n)
                du=x[node]-anchor-R@(old['slave']-old['closest'])
                fr=friction_return(old['spring'],old['normal'],n,du,fn,kt,self.mu,rotation=R)
                spring=fr['spring'];report['friction_increment']+=fr['friction_loss']
                report['algorithmic_increment']+=fr['trapezoidal_correction'];report['transferred_pairs']+=int(old_key!=key)
            ft=-kt*spring;f=fn*n+ft;force[node]+=f
            master=-w[:,None]*f;np.add.at(force,self.faces[face],master)
            extra+=np.einsum('aij,ai->j',affine_basis(np.repeat((np.asarray(shift)*self.L)[None],3,axis=0)),master)
            energy=.5*kt*float(spring@spring)
            history[key]=dict(spring=spring,normal=n,weights=w,triangle=tri[face]+H@shift,
                              slave=x[node].copy(),closest=q,energy=energy)
            report['contacts']+=1;report['normal_energy']+=.5*kn*d2;report['tangent_energy']+=energy
            report['penetration_mm']=max(report['penetration_mm'],float(depth))
            report['max_coulomb_ratio']=max(report['max_coulomb_ratio'],float(np.linalg.norm(ft))/fn)
        retained_nodes={key[0] for key in history}
        report['released_history_energy']=sum(old['energy'] for key,old in self.history.items() if key[0] not in retained_nodes)
        return force,extra,history,report
