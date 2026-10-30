"""Material surface quadrature embedded in the existing guarded friction law.

Virtual points have no mass/DOFs. Forces and stiffness majorants are pulled
back by the transpose of their exact material interpolation. Rule changes
require replay from a known inactive initial state, never history interpolation.
"""
from copy import deepcopy
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from .area_contact import AreaContact,SurfaceResolutionError
from .explicit_periodic import Periodic
from . import surface_patch


class EmbeddedContact:
    def __init__(self,X,t,periodic,kn,mu,skin,quadrature_rules=None,quadrature_tolerance=.001,error_budget='component',**kwargs):
        self.X=np.array(X,copy=True);self.periodic=periodic;self.rules=deepcopy(quadrature_rules or {});self.quadrature_tolerance=quadrature_tolerance
        if error_budget not in ('component','normalized'):raise ValueError('Unknown embedded error budget')
        self.error_budget=error_budget
        self.retain_partition=bool(kwargs.pop('retain_partition',False))
        self.rule_strategy=kwargs.pop('rule_strategy','single_pose')
        self.lookahead=tuple(kwargs.pop('lookahead_steps',()))
        self.absolute_energy_budget=kwargs.pop('absolute_energy_budget',0.)
        self.absolute_force_budget=kwargs.pop('absolute_force_budget',0.)
        if min(self.absolute_energy_budget,self.absolute_force_budget)<0 or not np.isfinite([self.absolute_energy_budget,self.absolute_force_budget]).all():raise ValueError('Invalid global surface integration budgets')
        if (self.absolute_energy_budget or self.absolute_force_budget) and error_budget!='normalized':raise ValueError('Absolute integration budgets require normalized error accounting')
        self.atlas_max_entries=kwargs.pop('atlas_max_entries',12000000)
        self.atlas_max_points=kwargs.pop('atlas_max_points',80000)
        if any(not isinstance(k,int) or k<1 for k in (self.atlas_max_entries,self.atlas_max_points)):raise ValueError('Positive atlas workspace budgets required')
        if len(self.lookahead)>4 or any(not isinstance(k,int) or not 0<k<=128 for k in self.lookahead):raise ValueError('Invalid cubature-only lookahead steps')
        if self.rule_strategy not in ('single_pose','pose_atlas'):raise ValueError('Unknown material cubature strategy')
        if self.rule_strategy=='pose_atlas' and self.retain_partition:raise ValueError('Choose atlas or experimental partition retention')
        kwargs.pop('quadrature_mode',None)
        # Discover the same physical/cell-cut faces as the existing backend.
        base=AreaContact(X,t,periodic,kn,mu,skin,quadrature_mode='material',**kwargs)
        faces=base.faces;N=len(X);rows=list(range(N));cols=list(range(N));values=[1.]*N
        added=[];virtual_faces=[];areas=base.integration_area0.copy();self.face_points={}
        for key,rule in sorted(self.rules.items(),key=lambda item:int(item[0])):
            face=int(key);B=np.asarray(rule['bary'],float);weights=np.asarray(rule['weights'],float)
            ids=faces[face];tri=X[ids];A=np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))/2
            if B.ndim!=2 or B.shape[1]!=3 or weights.shape!=(len(B),) or np.min(B)<-1e-12 or np.max(abs(B.sum(1)-1))>1e-12 or np.any(weights<=0) or abs(weights.sum()-A)>max(1e-12,A*1e-10):raise ValueError('Invalid full-reference-face quadrature rule')
            areas[ids]-=A/3;nodes=[]
            for bary,w in zip(B,weights):
                node=N+len(added);added.append((bary@tri,float(w)));virtual_faces.append(face);nodes.append(node)
                rows.extend([node]*3);cols.extend(ids);values.extend(bary)
            self.face_points[face]=(np.asarray(nodes),B,weights)
            if 'partition' in rule:surface_patch.validate_partition(rule['partition'])
        if np.min(areas)<-1e-10:raise ValueError('Negative residual nodal integration area')
        self.P=csr_matrix((values,(rows,cols)),shape=(N+len(added),N));augX=self.P@X
        self.inner=AreaContact(augX,t,Periodic(augX,periodic.L),kn,mu,skin,quadrature_mode='material',**kwargs)
        guard=self.inner.guard;guard.node_area0[:N]=np.maximum(areas,0.)
        if added:guard.node_area0[N:]=np.array([p[1] for p in added])
        self.inner.quadrature_weights=guard.node_area0.copy();guard.nodes=np.r_[base.nodes,np.arange(N,len(augX))]
        original_adjacent=guard.adjacent
        def adjacent(node,face,shift):
            if node<N:return original_adjacent(node,face,shift)
            source_face=faces[virtual_faces[node-N]]
            return any(original_adjacent(int(k),face,shift) for k in source_face)
        guard.adjacent=adjacent
        for node,face in enumerate(virtual_faces,N):
            self.inner.peers[node]=set(map(int,faces[face]))
            if self.inner.material_neighborhood is not None:
                self.inner.material_neighborhood.components[node]=self.inner.material_neighborhood.components[faces[face,0]]
        self._virtual_faces=virtual_faces
        self.nodes=base.nodes;self.faces=faces;self.node_area0=base.node_area0.copy();self.quadrature_mode='embedded'
        self._base_coverage=self.inner.edge_coverage
        self.inner.edge_coverage=self
        self._candidate_build=-1;self._face_allocation={}

    def __getattr__(self,name):return getattr(self.inner,name)
    @property
    def history(self):return self.inner.history
    @history.setter
    def history(self,value):self.inner.history=value
    @property
    def capture_audit(self):return self.inner.capture_audit
    @capture_audit.setter
    def capture_audit(self,value):self.inner.capture_audit=value
    @property
    def integration_area0(self):return np.asarray(self.P.T@self.inner.integration_area0).ravel()
    def edge_stiffness_diagonal(self):return np.asarray(self.P.T@self.inner.edge_stiffness_diagonal()).ravel()
    def accept_geometry(self,x,H):self.inner.accept_geometry(self.P@x,H)
    def replay_force_state(self,x,velocity,H):
        self._velocity=velocity;self._Hdot=H*0;self._dt=0.
        self._face_allocation={};self._certificates={};self._requests={};self._branch_index=None;self._base_coverage.cache={};self._x=x;self._H=H
        response=self.inner.replay_force_state(self.P@x,self.P@velocity,H)
        return self._collapse(response)
    def _collapse(self,response):return np.asarray(self.P.T@response[0]),response[1],response[2],response[3]
    def evaluate(self,x,velocity,H,Hdot,dt):
        self._velocity=velocity;self._Hdot=Hdot;self._dt=dt
        self._face_allocation={};self._certificates={};self._requests={};self._branch_index=None;self._base_coverage.cache={};self._x=x;self._H=H
        try:response=self.inner.evaluate(self.P@x,self.P@velocity,H,Hdot,dt)
        except (ValueError,RuntimeError):
            if self._requests:self._raise_refinement()
            raise
        active=set(key[1] for key in self.inner.trial_patch_branches)
        faces=set()
        for node in active:
            if node<len(self.X):faces.update(i for i,face in enumerate(self.faces) if node in face)
            else:faces.add(self._virtual_faces[node-len(self.X)])
        if faces:self._candidates(faces)
        for face in sorted(faces):
            keys=sorted(set(self._face_candidates.get(face,[])))
            groups=surface_patch.components(np.array([self.faces[j] for j,_ in keys]),[shift for _,shift in keys],self.X,self.L) if keys else []
            self._certify(face,keys,groups,{})
        if self._requests:self._raise_refinement()
        if self.absolute_energy_budget or self.absolute_force_budget:
            for audit_pass in range(2):
                per_face={}
                for (face,_),record in self._certificates.items():
                    old=per_face.setdefault(face,[0.,0.]);old[0]=max(old[0],record['absolute_energy_excess_Nmm']);old[1]=max(old[1],record['absolute_force_excess_L1_N'])
                energy=sum(v[0] for v in per_face.values());force=sum(v[1] for v in per_face.values())
                passed=energy<=self.absolute_energy_budget*.5*(1+1e-12)+1e-30 and force<=self.absolute_force_budget*.5*(1+1e-12)+1e-30
                if passed:break
                if audit_pass:raise SurfaceResolutionError('Assembled absolute integration budget exceeded')
                # Redistribute ONLY if the measured assembled excess uses up
                # the global budget. Inactive faces consume no applied budget.
                affected=[face for face,value in per_face.items() if max(value)>0]
                for face in affected:self._face_allocation[face]=(self.absolute_energy_budget/len(affected),self.absolute_force_budget/len(affected))
                for face,keys in list(self._certificates):
                    if face not in affected:continue
                    del self._certificates[(face,keys)]
                    groups=surface_patch.components(np.array([self.faces[j] for j,_ in keys]),[shift for _,shift in keys],self.X,self.L) if keys else []
                    self._certify(face,list(keys),groups,{})
                if self._requests:self._raise_refinement()
            self.last_integration_audit=dict(absolute_energy_excess_Nmm=energy,absolute_force_excess_L1_N=force,energy_budget_Nmm=self.absolute_energy_budget,force_budget_N=self.absolute_force_budget,certified_faces=len(per_face))
        return self._collapse(response)
    def _raise_refinement(self):
        error=SurfaceResolutionError('Embedded material quadrature requires inactive-checkpoint replay')
        error.feature=dict(kind='material_quadrature',requires_quadrature_refinement=True,rules=self._requests)
        raise error
    def update_topology(self,t):self.inner.update_topology(t)

    def _candidates(self,source_faces):
        guard=self.inner.guard
        if self._candidate_build!=guard.neighbor_builds or self.__dict__.get('_face_pose') is not guard.neighbor_geometry:
            # Build in exactly the guard's Verlet pose, even when a source
            # face first becomes active between neighbor rebuilds.
            points,H=guard.neighbor_geometry
            x=points[:len(self.X)];tri=x[self.faces];center=tri.mean(1)
            radius=np.linalg.norm(tri-center[:,None],axis=2).max(1)
            shifts=list(guard.trial_shifts)
            self._face_tree=cKDTree(np.concatenate([center+H@shift for shift in shifts]))
            self._face_search=(tri,center,radius,H,shifts)
            self._face_pose=guard.neighbor_geometry
            self._face_candidates={};self._face_candidate_count=0
            self._candidate_build=guard.neighbor_builds
        pending=sorted(set(map(int,source_faces))-self._face_candidates.keys())
        if not pending:return
        # Same Verlet skin and motion rebuild schedule as the guarded search.
        tri,center,radius,H,shifts=self._face_search;nf=len(self.faces)
        lists=self._face_tree.query_ball_point(center[pending],radius[pending]+radius.max()+guard.skin)
        count=sum(map(len,lists))
        if self._face_candidate_count+count>guard.max_candidates:raise ValueError('Embedded triangle candidate budget exceeded')
        self._face_candidate_count+=count
        for i,neighbors in zip(pending,lists):
            candidates=[]
            for index in neighbors:
                image,j=divmod(index,nf);shift=tuple(shifts[image])
                if all(guard.adjacent(int(k),j,shift) for k in self.faces[i]):continue
                if surface_patch.separation_bound(tri[i],tri[j]+H@shift)>=guard.skin:continue
                candidates.append((j,shift))
            self._face_candidates[i]=candidates

    def resolved(self,ids,shift,points,distance,point_candidates,x,H,shifts):
        # Called by the existing collision guard for each nonlocal active EE.
        edge=set(map(int,ids[:2]));source_faces=[i for i,f in enumerate(self.faces) if edge.issubset(set(map(int,f)))]
        self._candidates(source_faces)
        opposing=set(map(int,ids[2:]));target_faces=[i for i,f in enumerate(self.faces) if opposing.issubset(set(map(int,f)))]
        checked=False
        for face in source_faces:
            candidates=self._face_candidates.get(face,[])
            if not candidates:continue
            keys=sorted(set(candidates));cf=np.array([self.faces[j] for j,_ in keys]);cs=[s for _,s in keys]
            if not any(j in target_faces and tuple(s)==tuple(shift) for j,s in keys):continue
            groups=surface_patch.components(cf,cs,self.X,self.L)
            self._certify(face,keys,groups,point_candidates);checked=True
        if checked:return 'material_surface_integral'
        return self._base_coverage.resolved(ids,shift,points,distance,point_candidates,x,H,shifts)

    def _certify(self,face,keys,groups,point_candidates):
        key=(face,tuple(keys))
        if key in self._certificates:return
        source=self._x[self.faces[face]]
        if self._branch_index is None:
            self._branch_index={}
            for history_key,branches in self.inner.trial_patch_branches.items():self._branch_index.setdefault(history_key[1],[]).append(branches)
        if face in self.face_points:nodes,B,w=self.face_points[face]
        else:
            nodes=self.faces[face];B=np.eye(3)
            tri=self.X[self.faces[face]];w=np.full(3,np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))/6)
        admissible=sorted(set((v[1][2],tuple(v[3])) for node in nodes for branches in self._branch_index.get(int(node),[]) for v in branches))
        all_keys=sorted(set(keys)|set(admissible));key_lookup={key:i for i,key in enumerate(all_keys)}
        targets=np.array([self._x[self.faces[j]]+self._H@shift for j,shift in all_keys])
        identities=[(int(self.periodic.classes[node]),*np.rint((self.X[node]+self.L*np.asarray(shift))/(self.L*1e-10)).astype(np.int64)) for j,shift in all_keys for node in self.faces[j]]
        unique=sorted(set(identities));mapping=np.array([unique.index(item) for item in identities],int).reshape(-1,3)
        groups=[[key_lookup[keys[k]] for k in group] for group in groups]
        tri=self.X[self.faces[face]];A=np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))/2
        size=10+3*len(unique)
        # Reference uncertainty is uniformly bounded over ALL source faces.
        # Applied error is measured and shared globally at evaluate() exit.
        reference_energy_budget=self.absolute_energy_budget/len(self.faces)
        reference_force_budget=self.absolute_force_budget/len(self.faces)
        energy_budget,force_budget=self._face_allocation.get(face,(self.absolute_energy_budget,self.absolute_force_budget))
        if groups:
            try:reference=surface_patch.integrate(source,targets,A,initial_partition=(self.rules.get(str(face),{}).get('partition') if self.retain_partition else None),compress_rule=(not self.retain_partition or 'partition' not in self.rules.get(str(face),{})),kn=.5*self.kn,thickness=self.thickness,tolerance=self.quadrature_tolerance*.25,vertex_map=mapping,groups=groups,pointwise=True,error_budget=self.error_budget,absolute_energy_budget=reference_energy_budget/8,absolute_force_budget=reference_force_budget/8)
            except surface_patch.IntegrationRefinement as exc:raise SurfaceResolutionError(str(exc)) from exc
        else:reference=dict(value=np.zeros(size),bary=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]]),weights=np.full(3,A/3))
        current=np.zeros(size)
        from .barrier_contact import barrier
        for node,bary,weight in zip(nodes,B,w):
            for branches in self._branch_index.get(int(node),[]):
                E,Fn,_=barrier(branches[0][0],.5*self.kn,self.thickness)
                current[0]+=weight*E
                for candidate in branches:
                    force=weight*Fn/len(branches)*candidate[-1]/candidate[0]
                    current[1:10]+= (bary[:,None]*force).ravel()
                    index=key_lookup[(candidate[1][2],tuple(candidate[3]))]
                    for j,target_node in enumerate(mapping[index]):current[10+3*target_node:13+3*target_node]+=candidate[6][j+1]*force
        exact=reference['value'];force_scale=max(float(np.linalg.norm(exact[1:10].reshape(3,3).sum(0))),float(np.linalg.norm(exact[10:].reshape(-1,3),axis=1).sum()),1e-16,force_budget/(2*self.quadrature_tolerance*(len(exact)-1)))
        energy_error=abs(current[0]-exact[0])/max(abs(exact[0]),1e-18,energy_budget/(2*self.quadrature_tolerance))
        force_error=float(np.max(abs(current[1:]-exact[1:])))/force_scale
        relative_force_scale=max(float(np.linalg.norm(exact[1:10].reshape(3,3).sum(0))),float(np.linalg.norm(exact[10:].reshape(-1,3),axis=1).sum()),1e-16)
        energy_excess=max(0.,float(abs(current[0]-exact[0])-self.quadrature_tolerance*max(abs(exact[0]),1e-18)))
        force_excess=float(np.maximum(abs(current[1:]-exact[1:])-self.quadrature_tolerance*relative_force_scale,0.).sum())
        needs_refinement=(max(energy_error,force_error)>self.quadrature_tolerance) if not (energy_budget or force_budget) else (energy_excess>energy_budget/2 or force_excess>force_budget/2)
        if needs_refinement:
            if self.rule_strategy=='pose_atlas' and groups:
                from .contact_atlas import build
                old=self.rules.get(str(face),{});poses=list(old.get('atlas',[]))
                pose=dict(source=source.tolist(),targets=targets.tolist(),mapping=mapping.tolist(),groups=groups)
                if pose not in poses:poses.append(pose)
                controls=[]
                # Integration-only predicted geometries; never accepted bulk
                # states. Every actual step retains its normal force audit.
                for steps in self.lookahead:
                    if self._dt<=0:continue
                    delta=steps*self._dt
                    predicted_source=source+delta*self._velocity[self.faces[face]]
                    predicted_targets=np.array([self._x[self.faces[j]]+self._H@shift+delta*(self._velocity[self.faces[j]]+self._Hdot@shift) for j,shift in all_keys])
                    if any(surface_patch.separation_bound(predicted_source,target)<=0 for target in predicted_targets):continue
                    controls.append(dict(source=predicted_source.tolist(),targets=predicted_targets.tolist(),mapping=mapping.tolist(),groups=groups,lookahead_steps=steps))
                try:rule=build(poses+controls,A,.5*self.kn,self.thickness,self.quadrature_tolerance,old.get('bary'),max_points=self.atlas_max_points,max_entries=self.atlas_max_entries,energy_budget=energy_budget/2,force_budget=force_budget/2)
                except surface_patch.IntegrationRefinement as exc:
                    error=SurfaceResolutionError(str(exc))
                    error.feature=dict(kind='contact_atlas_fit_failure',face=face,poses=poses,controls=controls,reference_area_mm2=A,kn=.5*self.kn,thickness=self.thickness,tolerance=self.quadrature_tolerance,previous_bary=old.get('bary'),energy_budget=energy_budget/2,force_budget=force_budget/2,max_entries=self.atlas_max_entries,max_points=self.atlas_max_points)
                    raise error from exc
                self._requests[str(face)]=dict(bary=rule['bary'].tolist(),weights=rule['weights'].tolist(),atlas=poses,atlas_controls=controls,
                     atlas_enrichment_rounds=rule['enrichment_rounds'],atlas_fit_mode=rule['fit_mode'],atlas_fit_acceptance=rule['fit_acceptance'],atlas_audits=rule['audits'],candidate_points=rule['candidate_points'],moment_error=rule['moment_error'],
                     normal_energy_relative_error=float(energy_error),normal_force_relative_error=force_error)
            else:
                self._requests[str(face)]=dict(bary=reference['bary'].tolist(),weights=reference['weights'].tolist(),**(dict(partition=reference.get('partition',np.eye(3)[None]).tolist()) if self.retain_partition else {}),normal_energy_relative_error=float(energy_error),normal_force_relative_error=force_error)
        reference_force_scale=max(float(np.linalg.norm(exact[1:10].reshape(3,3).sum(0))),float(np.linalg.norm(exact[10:].reshape(-1,3),axis=1).sum()),1e-16)
        self._certificates[key]=dict(energy_error=float(energy_error),force_error=force_error,
            absolute_energy_excess_Nmm=energy_excess,absolute_force_excess_L1_N=force_excess)


def reinitialize_inactive(old,new,x,velocity,H,contact_free_prefix):
    """A topology change may reset quadrature only before ANY contact history."""
    if not contact_free_prefix or old.history or old.minimum_separation<old.thickness:
        raise ValueError('Embedded remesh requires a certified contact-free prefix; active-history reset prohibited')
    response=new.evaluate(x,velocity,H,H*0,0.)
    if response[2] or response[-1]['normal_energy']+response[-1]['tangent_energy']>1e-16:
        raise ValueError('Embedded remesh creates active initial quadrature')
    return response


def initialize_inactive_geometry(contact,x,H,contact_free_prefix):
    """Initialize a verified restart pose, not a jump from reference geometry.

    The caller has checked the saved material state and the entire noncontact
    prefix. Future increments still use the normal linear-motion certificate.
    """
    if not contact_free_prefix or contact.history:raise ValueError('Inactive geometry initialization requires a contact-free checkpoint')
    inner=getattr(contact,'inner',contact);guard=getattr(inner,'guard',inner)
    points=contact.P@x if hasattr(contact,'P') else x
    guard.accepted_geometry=(points.copy(),H.copy())
    guard.accepted_shifts=tuple(getattr(guard,'trial_shifts',[(0,0,0)]))
    if hasattr(guard,'neighbor_geometry'):del guard.neighbor_geometry
