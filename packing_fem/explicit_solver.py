"""Research-only explicit compatible-tetra FEM, periodic contact and remeshing.

Velocity Verlet with a reduced consistent constraint mass. No global stiffness
iterations. Material points use finite-strain Hencky J2; optional distance
barriers cover point-face and edge-edge contact. Remeshing is boundary-preserving
internal 2-to-3 reconnection and opt-in audited surface edge bisection.
"""
import json
import hashlib
from pathlib import Path
import numpy as np
from .explicit_material import geometry,advance,internal_force,constants,response
from .explicit_periodic import Periodic,deformation
from .macro_loading import uniaxial_load,diagnostic_cauchy
from .periodic_contact import Contact
from .adaptive_tet import propose,quality,closed_surface_volume


IMPLEMENTATION={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}


def lumped_mass(t,V0,rho,nn):
    result=np.zeros(nn);np.add.at(result,t.ravel(),np.repeat(rho*V0/4,4));return result


def profile(time,c):
    T=c['ramp_s'];h=c.get('hold_s',0.)
    if time>=T:return -c['compression_fraction'],0.,0.
    s=max(0,time/T);amplitude=c['compression_fraction']
    return -amplitude*(10*s**3-15*s**4+6*s**5),-amplitude*(30*s**2-60*s**3+30*s**4)/T,-amplitude*(60*s-180*s**2+120*s**3)/T**2


def stable_dt(x,t,V0,mass,c,velocity,contact,stress=None,elastic_b=None):
    geometry_fn=geometry
    if c.get('element_backend','numpy')=='native':
        from .native_kernel import load
        geometry_fn=load(c).geometry
    contact_bound=float('inf');bulk=float('inf')
    if c.get('timestep_method') not in ('triangle_mass_bound','assembled_mass_bound'):
        v,_,_=geometry_fn(x,t);p=x[t];faces=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
        triangles=p[:,faces];areas=np.linalg.norm(np.cross(triangles[:,:,1]-triangles[:,:,0],triangles[:,:,2]-triangles[:,:,0]),axis=2)/2
        height=3*v/areas.max(axis=1);G,K=constants(c);rho=c['density_tonne_mm3']*V0/v
        wave=np.sqrt((K+4*G/3)/rho)*np.maximum(1.,V0/v)
        bulk=float(np.min(height/wave))
    if c.get('timestep_method')=='triangle_mass_bound':
        from .explicit_timestep import triangle_frequency_bound
        if c.get('element_backend','numpy')=='native':
            from .native_kernel import load
            bound=load(c).bound(x,t,mass,c,stress,elastic_b)
        else:bound=triangle_frequency_bound(x,t,mass,c,stress,elastic_b)
        bulk=2/np.sqrt(bound)
    if c.get('timestep_method')=='assembled_mass_bound':
        from .explicit_timestep import assembled_frequency_bound
        bulk=2/np.sqrt(assembled_frequency_bound(x,t,mass,c,stress,elastic_b))
    if len(contact.faces):
        f=x[contact.faces];A=np.linalg.norm(np.cross(f[:,1]-f[:,0],f[:,2]-f[:,0]),axis=1)/2
        nodeA=np.zeros(len(x))
        for k in range(3):np.add.at(nodeA,contact.faces[:,k],A/3)
        nodeA=np.maximum(nodeA,getattr(contact,'integration_area0',contact.node_area0))
        nodes=contact.nodes
        diagonal=2*c['contact_penalty_mpa_mm']*nodeA[nodes]
        if hasattr(contact,'edge_stiffness_diagonal'):diagonal+=contact.edge_stiffness_diagonal()[nodes]
        if c.get('active_contact_timestep',False):
            if c.get('contact_backend') not in ('area_guarded','embedded_area'):raise ValueError('Active contact timestep requires area_guarded')
            # Outside the compactly supported barrier the contact Hessian is
            # zero. Keep the accepted active-support bound plus motion guards.
            diagonal=contact.edge_stiffness_diagonal()[nodes]
        active=diagonal>0
        if np.any(active):contact_bound=float(np.sqrt(np.min(mass[nodes[active]]/diagonal[active])))
    motion=c['contact_skin_mm']*.1/max(float(np.max(np.linalg.norm(velocity,axis=1))),1e-30)
    return min(c['cfl']*bulk,c['cfl']*contact_bound,motion,c.get('maximum_dt_s',float('inf')))


def validate(c):
    if c.get('surface_remesh_strategy','midpoint') not in ('midpoint','closest_quality'):raise ValueError('Unknown surface remesh strategy')
    if c.get('closest_tie_rule','stop') not in ('stop','clarke'):raise ValueError('Unknown closest tie rule')
    if c.get('element_backend','numpy') not in ('numpy','native'):raise ValueError('Unknown element backend')
    for k in ['length_mm','young_mpa','density_tonne_mm3','ramp_s','contact_penalty_mpa_mm','contact_skin_mm','cfl']:
        if not np.isfinite(c[k]) or c[k]<=0:raise ValueError(f'Positive finite {k} required')
    if not 0<c['cfl']<=.3 or not 0<=c['compression_fraction']<.9:raise ValueError('Prototype CFL/compression bounds exceeded')
    for key in ('hold_s','damping_per_s','friction','compression_fraction','minimum_quality','remesh_quality','remesh_energy_tolerance','energy_tolerance','kinetic_tolerance','max_penetration_mm'):
        if not np.isfinite(c[key]):raise ValueError(f'Finite {key} required')
    if not 0<c['minimum_quality']<1 or not 0<c['remesh_quality']<=1 or c['remesh_energy_tolerance']<=0:raise ValueError('Invalid remesh accuracy/quality settings')
    if c['energy_tolerance']<=0 or c['kinetic_tolerance']<=0 or c['max_penetration_mm']<=0:raise ValueError('Invalid audit tolerances')
    if c.get('hold_s',0)<0 or c.get('damping_per_s',0)<0 or c['friction']<0:raise ValueError('Negative numerical/contact setting')
    if c.get('initial_peeq',0)<0:raise ValueError('Negative initial plastic history')
    for key in ('max_steps','output_every'):
        if not isinstance(c[key],int) or c[key]<1:raise ValueError(f'Positive integer {key} required')
    for name in ('multiple_patches','certified_local_filter','local_material_filter','edge_stationarity_filter','active_contact_timestep','surface_remesh_on_edge_guard'):
        if name in c and not isinstance(c[name],bool):raise ValueError(f'Boolean {name} required')
    for name in ('surface_remesh_every','surface_remesh_max_events'):
        if name in c and (not isinstance(c[name],int) or c[name]<0):raise ValueError(f'Nonnegative integer {name} required')
    if any(not isinstance(k,int) or k<1 for k in c.get('surface_remesh_steps',[])):raise ValueError('Positive surface remesh steps required')
    if not isinstance(c.get('remesh_every',0),int) or c.get('remesh_every',0)<0:raise ValueError('Invalid remesh interval')
    if c.get('certified_local_filter',False) and (c.get('initial_remesh',False) or c.get('remesh_every',0)):
        raise ValueError('Canonical certificate cannot follow internal reconnection')
    if c.get('unload_s',0)<0 or c.get('recovery_hold_s',0)<0:raise ValueError('Negative unloading time')
    if c.get('recovery_hold_s',0)>0 and c.get('unload_s',0)<=0:raise ValueError('Recovery hold needs an unloading segment')
    if c.get('contact_backend','legacy') not in ('legacy','tracked_guarded','tracked_edges','distance_barrier','area_guarded','embedded_area'):raise ValueError('Unknown contact backend')
    if not np.isfinite(c.get('edge_penalty_scale',1.)) or c.get('edge_penalty_scale',1.)<=0:raise ValueError('Positive edge penalty scale required')
    if c.get('loading_mode','displacement') not in ('displacement','nominal_pressure','cauchy_pressure','porosity_target','porosity_cycle'):raise ValueError('Unknown loading mode')
    if c.get('loading_mode')=='porosity_cycle':
        from .porosity_cycle import PorosityCycle
        PorosityCycle(c)
        if c.get('pressure_stages'):raise ValueError('Porosity cycle has no pressure stages or pressure cap')
    if c.get('loading_mode') in ('nominal_pressure','cauchy_pressure','porosity_target') and (not np.isfinite(c.get('pressure_mpa',0)) or c.get('pressure_mpa',0)<=0):raise ValueError('Positive finite pressure required')
    if c.get('timestep_method','element_wave') not in ('element_wave','assembled_mass_bound','triangle_mass_bound'):raise ValueError('Unknown timestep method')
    if c.get('loading_mode')=='porosity_target' and (not 0<c.get('target_final_porosity',-1)<1 or c.get('hold_s',0)<=0 or c.get('unload_s',0)>0):raise ValueError('Porosity control needs a physical target, positive equilibration hold and no unloading')
    if c.get('pressure_stages'):
        stages=c['pressure_stages']
        if c.get('loading_mode')!='cauchy_pressure' or c.get('unload_s',0):raise ValueError('Pressure stages require Cauchy control without unloading')
        previous=0.
        for stage in stages:
            if not all(np.isfinite(stage.get(k,float('nan'))) for k in ('pressure_mpa','ramp_s','hold_s','max_hold_s')):raise ValueError('Finite pressure-stage parameters required')
            if not previous<stage['pressure_mpa']<=c['pressure_mpa'] or not 0<stage['ramp_s'] or not 0<stage['hold_s']<=stage['max_hold_s']:raise ValueError('Invalid increasing pressure stages')
            previous=stage['pressure_mpa']
        if abs(previous-c['pressure_mpa'])>1e-12:raise ValueError('Final pressure stage must equal pressure cap')
    constants(c)


def solve(directory,X,t,c):
    geometry_fn,advance_fn,internal_fn,quality_fn=geometry,advance,internal_force,quality
    c=dict(c);restart=None;operator_update=[];hold_budget_extended=False
    implementation=IMPLEMENTATION
    if c.get('resume_checkpoint'):
        from .checkpoint import load as load_checkpoint
        restart,metadata=load_checkpoint(c['resume_checkpoint'])
        old_implementation=metadata.get('implementation_sha256',{})
        differences=[name for name in set(old_implementation)|set(implementation) if old_implementation.get(name)!=implementation.get(name)]
        if differences:
            compatible={'area_contact.py','embedded_contact.py','surface_patch.py'}
            if c.get('allow_solver_update_at_inactive_checkpoint'):compatible.add('explicit_solver.py')
            if not c.get('allow_contact_operator_update') or not restart.get('contact_free_prefix',False) or not set(differences)<=compatible:raise ValueError('Checkpoint implementation mismatch')
            operator_update=sorted(differences)
        ignored={'resume_checkpoint','checkpoint_every','checkpoint_stop_after','max_steps','output_every','inactive_quadrature_replay','allow_contact_operator_update','allow_solver_update_at_inactive_checkpoint','extend_stage_hold_budget','checkpoint_free_every','stop_request_file'}
        if c.get('extend_stage_hold_budget'):
            previous=metadata['config'].get('pressure_stages',[]);current=c.get('pressure_stages',[])
            if not previous or len(previous)!=len(current):raise ValueError('Hold extension requires the same pressure stages')
            for old,new in zip(previous,current):
                if {k:v for k,v in old.items() if k!='max_hold_s'}!={k:v for k,v in new.items() if k!='max_hold_s'} or new['max_hold_s']<old['max_hold_s']:raise ValueError('Only increasing hold budgets may be continued')
                hold_budget_extended=hold_budget_extended or new['max_hold_s']>old['max_hold_s']
        def clean(config):
            result={k:v for k,v in config.items() if k not in ignored}
            if result.get('contact_backend') in ('area_guarded','embedded_area'):
                for key,value in dict(local_material_filter=True,edge_stationarity_filter=False,multiple_patches=False,certified_local_filter=False).items():result.setdefault(key,value)
            result.pop('embedded_quadrature_rules',None)
            if c.get('extend_stage_hold_budget') and 'pressure_stages' in result:result['pressure_stages']=[{k:v for k,v in stage.items() if k!='max_hold_s'} for stage in result['pressure_stages']]
            return result
        changed_rules=c.get('embedded_quadrature_rules',{})!=restart.get('runtime_rules',metadata['config'].get('embedded_quadrature_rules',{}))
        if changed_rules:
            if not c.get('inactive_quadrature_replay') or not restart.get('contact_free_prefix',False):raise ValueError('Quadrature changes require a certified contact-free checkpoint prefix')
            left=clean(c);right=clean(metadata['config']);left.pop('embedded_quadrature_rules',None);right.pop('embedded_quadrature_rules',None)
        else:left=clean(c);right=clean(metadata['config'])
        if left!=right:raise ValueError('Checkpoint physical/numerical configuration mismatch')
        X=restart['X'];t=restart['t']
    if c.get('contact_backend') in ('area_guarded','embedded_area'):c.setdefault('local_material_filter',True);c.setdefault('edge_stationarity_filter',False);c.setdefault('multiple_patches',False);c.setdefault('certified_local_filter',False)
    validate(c);directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    if c.get('element_backend','numpy')=='native':
        from .native_kernel import load
        kernel=load(c);geometry_fn,advance_fn,internal_fn,quality_fn=kernel.geometry,kernel.advance,kernel.internal_force,kernel.quality
        (directory/'native-kernel.json').write_text(json.dumps(kernel.identity(),indent=2)+'\n')
    np.savez_compressed(directory/'initial-mesh.npz',points=X,tets=t)
    (directory/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    V0,initial_gradient,D=geometry_fn(X,t);inverse_D=initial_gradient[:,1:,:];initial_V0=float(V0.sum());periodic=Periodic(X,c['length_mm'],mass_frame=c.get('mass_frame','pinned'))
    pressure_control=c.get('loading_mode') in ('nominal_pressure','cauchy_pressure','porosity_target')
    goal_control=c.get('loading_mode')=='porosity_target'
    cauchy_control=c.get('loading_mode')=='cauchy_pressure'
    if goal_control and not c['target_final_porosity']<1-initial_V0/c['length_mm']**3:raise ValueError('Porosity target must be below the actual initial porosity')
    if pressure_control:periodic.prescribed_active=False;periodic.free=np.arange(periodic.ndof)
    mass=lumped_mass(t,V0,c['density_tonne_mm3'],len(X));periodic.mass(mass)
    contact_type=Contact
    if c.get('contact_backend','legacy')=='tracked_guarded':
        from .tracked_contact import TrackedContact
        contact_type=TrackedContact
    contact_kwargs={}
    if c.get('contact_backend')=='tracked_edges':
        from .edge_tracked_contact import EdgeTrackedContact
        contact_type=EdgeTrackedContact;contact_kwargs['edge_penalty_scale']=c.get('edge_penalty_scale',1.)
    if c.get('contact_backend') in ('distance_barrier','area_guarded','embedded_area'):
        from .barrier_contact import BarrierContact
        from .area_contact import AreaContact
        contact_type=AreaContact if c['contact_backend'] in ('area_guarded','embedded_area') else BarrierContact;contact_kwargs={'thickness':c.get('barrier_thickness_mm',.02),'max_candidates':c.get('contact_candidate_budget',500000)}
    if c.get('contact_backend') in ('area_guarded','embedded_area'):contact_kwargs.update(local_material_filter=c.get('local_material_filter',True),edge_stationarity_filter=c.get('edge_stationarity_filter',False),multiple_patches=c.get('multiple_patches',False),certified_local_filter=c.get('certified_local_filter',False),closest_tie_rule=c.get('closest_tie_rule','stop'),edge_quadrature_relative_tolerance=c.get('edge_quadrature_relative_tolerance',0.),quadrature_mode=c.get('contact_quadrature_mode','nodal'))
    if c.get('contact_backend')=='embedded_area':
        from .embedded_contact import EmbeddedContact
        contact_type=EmbeddedContact
        contact_kwargs.update(quadrature_rules=(restart.get('runtime_rules',c.get('embedded_quadrature_rules',{})) if restart is not None and not changed_rules else c.get('embedded_quadrature_rules',{})),quadrature_tolerance=c.get('surface_quadrature_tolerance',.001),error_budget=c.get('surface_error_budget','component'),retain_partition=c.get('retain_surface_partition',False))
    contact=contact_type(X,t,periodic,c['contact_penalty_mpa_mm'],c['friction'],c['contact_skin_mm'],**contact_kwargs)
    if c.get('save_contact_audit',False):
        if c.get('contact_backend') not in ('area_guarded','embedded_area'):raise ValueError('Contact audit capture requires area_guarded')
        contact.capture_audit=True
    b=np.repeat(np.eye(3)[None],len(t),axis=0);ep=np.full(len(t),float(c.get('initial_peeq',0.)));q=np.zeros(periodic.ndof);v=q.copy();x=X.copy()
    if 'initial_velocity_mm_s' in c:
        initial_velocity=np.asarray(c['initial_velocity_mm_s'],float)
        if initial_velocity.shape!=X.shape or not np.all(np.isfinite(initial_velocity)):raise ValueError('Invalid manufactured initial velocity')
        v=periodic.remap_velocity(periodic.force(mass[:,None]*initial_velocity),0.)
    f_int,volume,energy,J,D,sigma=internal_fn(x,t,b,ep,c)
    fc,extra,hist,ci=contact.evaluate(x,periodic.velocities(v),np.eye(3)*periodic.L,np.zeros((3,3)),0.)
    contact.history=hist
    if hasattr(contact,'accept_geometry'):contact.accept_geometry(x,np.eye(3)*periodic.L)
    f=periodic.force(f_int+fc,extra);a=periodic.acceleration(f,v,0.,c['damping_per_s'])
    release_time=c['ramp_s']+c.get('hold_s',0.);unload=c.get('unload_s',0.)
    duration=release_time+unload+c.get('recovery_hold_s',0.);release_load=None;external=np.zeros(periodic.ndof);command_pressure=0.
    goal_phase='ramp';goal_load_time=0.;goal_hold_start=0.;goal_pressure=0.;goal_reached=False
    if goal_control:duration=c.get('max_goal_time_s',10*(c['ramp_s']+c['hold_s']))
    stage_program=c.get('pressure_stages',[]);stage_index=0;stage_start=0.;stage_records=[]
    if stage_program:duration=sum(stage['ramp_s']+stage['max_hold_s'] for stage in stage_program)
    retry_limit=float('inf');contact_retries=0;retry_log=[];algorithmic_variation=0.
    from .barrier_contact import ContactResolutionError
    time=0.;W=0.;damp=0.;friction=0.;released=0.;algorithmic=0.;steps=0;remesh_events=[];history=[];failure=None
    cycle=None
    if c.get('loading_mode')=='porosity_cycle':
        from .porosity_cycle import PorosityCycle
        cycle=PorosityCycle(c,restart.get('porosity_cycle_state') if restart is not None else None)
        if 1-volume.sum()/periodic.L**3<=c['target_final_porosity']:raise ValueError('Initial porosity must exceed cycle target')
        duration=c['maximum_cycle_time_s']
    def try_remesh(ap):
        nonlocal t,b,ep,V0,mass,v,f_int,volume,energy,J,D,sigma,f,a,inverse_D
        proposer=propose
        if c.get('volume_reconnections',False):
            from .reconnection import propose_reconnection
            proposer=propose_reconnection
        proposal=proposer(x,t,b,ep,V0,c,c['remesh_quality'],c['remesh_energy_tolerance'])
        if proposal is not None:
            t2,b2,ep2,V02,event=proposal;oldM=periodic.M;oldmomentum=oldM@v;oldke=float(.5*v@oldmomentum)
            oldnv=periodic.velocities(v)
            mass2=lumped_mass(t2,V02,c['density_tonne_mm3'],len(X));periodic.mass(mass2);v2=periodic.remap_velocity(oldmomentum,v[periodic.prescribed]);newke=float(.5*v2@(periodic.M@v2))
            scale=max(oldke,float(V0@energy),c['young_mpa']*initial_V0*1e-12)
            error=abs(newke-oldke)/scale
            newnv=periodic.velocities(v2)
            Pold=(mass[:,None]*oldnv).sum(axis=0);Pnew=(mass2[:,None]*newnv).sum(axis=0)
            Lold=np.cross(x,mass[:,None]*oldnv).sum(axis=0);Lnew=np.cross(x,mass2[:,None]*newnv).sum(axis=0)
            pscale=max(float(np.linalg.norm(Pold)),np.sqrt(2*mass.sum()*scale))
            perr=float(np.linalg.norm(Pnew-Pold))/pscale;lerr=float(np.linalg.norm(Lnew-Lold))/(pscale*periodic.L)
            event.update(linear_momentum_scaled_error=perr,angular_momentum_scaled_error=lerr)
            newfi,newvol,newenergy,newJ,newD,newsigma=internal_fn(x,t2,b2,ep2,c)
            force_error=float(np.linalg.norm(newfi-f_int))/max(float(np.linalg.norm(f_int)),c['young_mpa']*initial_V0/periodic.L*1e-12)
            material_error=float(np.max(abs(newvol-V02*newJ)/V02))
            event.update(internal_force_scaled_error=force_error,material_volume_relative_error=material_error,
                         contact_material_samples_preserved=True,contact_history_count=len(contact.history),
                         material_energy_increment_Nmm=float(V02@newenergy-V0@energy))
            admissible=max(error,perr,lerr)<=c['remesh_energy_tolerance']
            if c.get('volume_reconnections',False):admissible=admissible and force_error<=c['remesh_energy_tolerance'] and material_error<1e-8
            if admissible:
                # Free surface facets do not change in a 2-to-3 flip,
                # so retain contact history rather than resetting it.
                event.update(time_s=time,kinetic_energy_scaled_error=error,accepted=True)
                t,b,ep,V0,mass,v=t2,b2,ep2,V02,mass2,v2
                contact.update_topology(t)
                f_int,volume,energy,J,D,sigma=newfi,newvol,newenergy,newJ,newD,newsigma
                inverse_D=geometry_fn(x,t)[1][:,1:,:]
                f=periodic.force(f_int+fc,extra);a=periodic.acceleration(f+external,v,ap if periodic.prescribed_active else 0.,c['damping_per_s'])
            else:
                event.update(time_s=time,kinetic_energy_scaled_error=error,accepted=False,reason='volume remesh momentum/energy/force audit')
                periodic.mass(mass)
            remesh_events.append(event)
            return bool(event['accepted'])
        return False

    def try_surface_remesh(ap,requested_edge=None,fraction=.5,candidate_context=None):
        nonlocal X,x,t,b,ep,V0,mass,periodic,contact,q,v,f_int,volume,energy,J,D,sigma,f,a,fc,extra,ci,external,inverse_D
        from .surface_remesh import bisect,transfer_contact,audit
        from .adaptive_tet import boundary
        if c.get('contact_backend') not in ('area_guarded','embedded_area'):raise ValueError('Surface refinement requires an area backend')
        if sum(e.get('accepted',False) and e.get('operation')=='periodic_edge_bisection' for e in remesh_events)>=c.get('surface_remesh_max_events',1):return
        faces=boundary(t);edges=np.unique(np.sort(np.concatenate([faces[:,[i,j]] for i,j in ((0,1),(1,2),(2,0))]),axis=1),axis=0)
        lengths=np.linalg.norm(x[edges[:,1]]-x[edges[:,0]],axis=1)
        ordered=np.argsort(-lengths,kind='stable')
        # Caller may prescribe one edge for reproducible verification. Otherwise
        # refine the longest current boundary edge (a numerical choice).
        choices=[np.asarray(requested_edge,int)] if requested_edge is not None else ([np.asarray(c['surface_remesh_edge'],int)] if 'surface_remesh_edge' in c else edges[ordered])
        event=dict(operation='periodic_edge_bisection',time_s=time,accepted=False)
        if candidate_context is not None:event['candidate_selection']=candidate_context
        try:
            if c.get('contact_backend')=='embedded_area' and not contact_free_prefix:raise ValueError('Active embedded remeshing requires inactive-checkpoint replay')
            oldnv=periodic.velocities(v)
            proposal=None
            for edge in choices:
                try:proposal=bisect(X,x,t,b,ep,V0,oldnv,edge,periodic.L,fraction=fraction);break
                except ValueError as exc:event['proposal_rejection']=str(exc)
            if proposal is None:raise ValueError('No conforming periodic surface edge available')
            p=proposal;event.update(p['report'])
            np.savez_compressed(directory/f'surface-remesh-attempt-{len(remesh_events)+1:04d}.npz',reference_points=p['X'],points=p['x'],tets=p['t'],elastic_b=p['b'],peeq=p['ep'],reference_volume=p['V0'],parents=p['parents'],supports=p['supports'],weights=p['weights'])
            p2=Periodic(p['X'],periodic.L,mass_frame=c.get('mass_frame','pinned'))
            p2.prescribed_active=periodic.prescribed_active
            p2.free=np.delete(np.arange(p2.ndof),p2.prescribed) if p2.prescribed_active else np.arange(p2.ndof)
            m2=lumped_mass(p['t'],p['V0'],c['density_tonne_mm3'],len(p['X']));p2.mass(m2)
            q2=p2.remap_positions(p['x'],q[-6:])
            if np.max(abs(p2.positions(q2)-p['x']))>periodic.L*1e-10:raise ValueError('Periodic displacement transfer failed')
            if np.max(abs(q2[-6:]-q[-6:]))>1e-10:raise ValueError('Remesh changes macroscopic cell')
            v2=p2.remap_velocity(p2.force(m2[:,None]*p['velocity']),v[periodic.prescribed])
            p['velocity']=p2.velocities(v2)
            remesh_contact_kwargs=dict(contact_kwargs)
            if contact.extension is not None:
                if fraction!=.5:raise ValueError('Non-midpoint geometry extension is not implemented')
                from .refined_extension import refine_extension
                remesh_contact_kwargs['geometry_extension']=refine_extension(contact.extension,p)
            if c.get('contact_backend')=='embedded_area':remesh_contact_kwargs['quadrature_rules']={}
            newcontact=contact_type(p['X'],p['t'],p2,c['contact_penalty_mpa_mm'],c['friction'],c['contact_skin_mm'],**remesh_contact_kwargs)
            H=deformation(q[-6:])*periodic.L
            if c.get('contact_backend')=='embedded_area':
                from .embedded_contact import reinitialize_inactive
                evaluated=reinitialize_inactive(contact,newcontact,p['x'],p2.velocities(v2),H,contact_free_prefix)
                event['inactive_quadrature_reinitialized']=True
            else:evaluated=transfer_contact(contact,newcontact,p,H)
            if newcontact.quadrature_mode=='material':
                event['material_quadrature']=dict(area_before=float(contact.integration_area0.sum()),area_after=float(newcontact.integration_area0.sum()),points_before=int(np.count_nonzero(contact.integration_area0)),points_after=int(np.count_nonzero(newcontact.integration_area0)),new_geometry_nodes=int(len(p['X'])-len(X)))
            audit_report=audit((X,x,t,b,ep,V0,oldnv),p,c,ci,evaluated[-1],c['remesh_energy_tolerance'],old_response=(fc,extra),new_response=evaluated)
            event.update(audit_report)
            if quality_fn(p['x'],p['t']).min()<c['minimum_quality']:event.update(accepted=False,reason='Refined mesh below minimum quality')
            if not event['accepted']:return
            external2=np.zeros(p2.ndof);external2[-6:]=external[-6:]
            fc2,extra2,_,ci2=evaluated
            newcontact.capture_audit=c.get('save_contact_audit',False)
            if newcontact.capture_audit:
                if newcontact.quadrature_mode=='material':fc2,extra2,hist2,ci2=newcontact.replay_force_state(p['x'],p2.velocities(v2),H)
                else:fc2,extra2,hist2,ci2=newcontact.evaluate(p['x'],p2.velocities(v2),H,H*0,0.)
                newcontact.history=hist2
            newcontact.accept_geometry(p['x'],H)
            fi2,vol2,en2,J2,D2,sigma2=internal_fn(p['x'],p['t'],p['b'],p['ep'],c)
            f2=p2.force(fi2+fc2,extra2);a2=p2.acceleration(f2+external2,v2,ap if p2.prescribed_active else 0.,c['damping_per_s'])
            inverse_D2=geometry_fn(p['x'],p['t'])[1][:,1:,:]
            # Commit only after every potentially failing calculation completes.
            X,x,t,b,ep,V0,mass,periodic,contact,q,v=p['X'],p['x'],p['t'],p['b'],p['ep'],p['V0'],m2,p2,newcontact,q2,v2
            inverse_D=inverse_D2
            fc,extra,ci,external,f_int,volume,energy,J,D,sigma,f,a=fc2,extra2,ci2,external2,fi2,vol2,en2,J2,D2,sigma2,f2,a2
        except ValueError as exc:event.update(accepted=False,reason=str(exc))
        finally:remesh_events.append(event)
        return event['accepted']

    if c.get('initial_remesh',False):try_remesh(0.)
    initial_energy=float(V0@energy)+ci['normal_energy']+ci['tangent_energy']+float(.5*v@(periodic.M@v));dt_min=float('inf');dt_max=0.
    contact_free_prefix=not bool(hist) and ci.get('minimum_separation_mm',0)>=c.get('barrier_thickness_mm',.02)
    initial_available_energy=float(V0@(energy-c['yield_mpa']*ep-.5*c['hardening_mpa']*ep**2))+ci['normal_energy']+ci['tangent_energy']+float(.5*v@(periodic.M@v))
    def snapshot():
        F=deformation(q[-6:]);vc=float(np.linalg.det(F)*periodic.L**3)
        if vc<=0:raise ValueError('Inverted macroscopic cell')
        kinetic=float(.5*v@(periodic.M@v));internal=float(V0@energy)
        total=kinetic+internal+ci['normal_energy']+ci['tangent_energy']+damp+friction+released+algorithmic-initial_energy
        reaction=periodic.M@a-f+c['damping_per_s']*(periodic.M@v)
        nominal=float(-(external[periodic.prescribed] if pressure_control else reaction[periodic.prescribed])/periodic.L**3)
        dynamic=reaction-external;dynamic[-6:]/=periodic.L
        return [time,-q[periodic.prescribed],1-volume.sum()/vc,nominal,kinetic,internal,W,damp,friction,released,
                ci['normal_energy'],ci['tangent_energy'],total-W,float(ep.max()),float(J.min()),float(quality_fn(x,t).min()),
                ci['penetration_mm'],ci['max_coulomb_ratio'],ci['contacts'],len(t),float(np.max(np.abs(dynamic[periodic.free]))),algorithmic,ci.get('edge_contacts',0),ci.get('minimum_separation_mm',float('nan')),command_pressure if pressure_control else float('nan')]
    history.append(snapshot());audit_rows=[history[0][0:7]+[history[0][12],ci['normal_energy'],ci['tangent_energy'],ci['contacts']]];coulomb_peak=ci['max_coulomb_ratio']
    progress=directory/'progress.jsonl';progress.write_text('')
    def log_progress():
        row=history[-1]
        with progress.open('a') as stream:stream.write(json.dumps({'step':steps,'time_s':time,'axial_compression':row[1],'porosity':row[2],'contacts':int(row[18]),'edge_contacts':int(row[22]),'elements':len(t),'min_quality':row[15],'nominal_pressure_MPa':row[3],'kinetic_Nmm':row[4],'internal_Nmm':row[5],'energy_residual_Nmm':row[12]})+'\n')
    log_progress()
    from .area_contact import SurfaceResolutionError
    resolution_attempts=[];failure_feature=None;checkpoint_paths=[]
    checkpoint_names=('X','t','V0','initial_V0','mass','b','ep','q','v','x','f_int','volume','energy','J','D','sigma','fc','extra','ci','f','a','external','command_pressure','time','W','damp','friction','released','algorithmic','algorithmic_variation','steps','history','audit_rows','coulomb_peak','initial_energy','initial_available_energy','dt_min','dt_max','duration','release_load','retry_limit','contact_retries','retry_log','remesh_events','resolution_attempts','goal_phase','goal_load_time','goal_hold_start','goal_pressure','goal_reached','stage_index','stage_start','stage_records','contact_free_prefix','inverse_D')
    def checkpoint(scope):
        from .checkpoint import save,contact_state
        state={name:scope[name] for name in checkpoint_names};state['contact']=contact_state(contact);state['prescribed_active']=periodic.prescribed_active
        if cycle is not None:state['porosity_cycle_state']=cycle.state
        if c.get('contact_backend')=='embedded_area':state['runtime_rules']=contact.rules
        target=directory/f'checkpoint-{steps:08d}'
        if target.exists() and not (target/'manifest.json').exists():
            # Retain an interrupted serialization; never mistake an existing
            # directory for a complete checkpoint or overwrite its evidence.
            serial=1
            while (directory/f'checkpoint-{steps:08d}-recovery-{serial:03d}').exists():serial+=1
            target=directory/f'checkpoint-{steps:08d}-recovery-{serial:03d}'
        if not target.exists():save(target,state,dict(config={k:v for k,v in c.items() if k!='resume_checkpoint'},implementation_sha256=implementation,checkpoint_fields_version=2))
        checkpoint_paths.append(str(target))
        return target
    if restart is not None:
        from .checkpoint import restore_contact
        if 'inverse_D' not in restart:restart['inverse_D']=geometry_fn(restart['x'],restart['t'])[1][:,1:,:]
        X,t,V0,initial_V0,mass,b,ep,q,v,x,f_int,volume,energy,J,D,sigma,fc,extra,ci,f,a,external,command_pressure,time,W,damp,friction,released,algorithmic,algorithmic_variation,steps,history,audit_rows,coulomb_peak,initial_energy,initial_available_energy,dt_min,dt_max,duration,release_load,retry_limit,contact_retries,retry_log,remesh_events,resolution_attempts,goal_phase,goal_load_time,goal_hold_start,goal_pressure,goal_reached,stage_index,stage_start,stage_records,contact_free_prefix,inverse_D=(restart[name] for name in checkpoint_names)
        periodic.prescribed_active=restart['prescribed_active'];periodic.free=np.delete(np.arange(periodic.ndof),periodic.prescribed) if periodic.prescribed_active else np.arange(periodic.ndof);periodic.mass(mass)
        if changed_rules or operator_update:
            check_force,check_volume,check_energy,check_J,check_D,check_sigma=internal_fn(x,t,b,ep,c)
            if metadata.get('checkpoint_fields_version',1)<2:
                if not np.allclose(f,periodic.force(check_force+fc,extra),rtol=1e-10,atol=1e-12):raise ValueError('Legacy checkpoint generalized-force consistency failed')
                f_int=check_force
            for saved,recomputed in ((f_int,check_force),(volume,check_volume),(energy,check_energy),(J,check_J),(D,check_D),(sigma,check_sigma),(x,periodic.positions(q))):
                if not np.allclose(saved,recomputed,rtol=1e-10,atol=1e-12):raise ValueError('Checkpoint physical-state consistency failed')
            if 'explicit_solver.py' in operator_update:
                if periodic.prescribed_active or not pressure_control:raise ValueError('Solver update requires free pressure control')
                if len(audit_rows)>1:
                    old_dt=audit_rows[-1][0]-audit_rows[-2][0]
                    expected_a=periodic.acceleration(f+external,v,0.,c['damping_per_s'])/(1-.5*old_dt*c['damping_per_s'])
                    if not np.allclose(a,expected_a,rtol=1e-8,atol=1e-8):raise ValueError('Checkpoint acceleration consistency failed')
            H=deformation(q[-6:])*periodic.L
            from .embedded_contact import initialize_inactive_geometry
            initialize_inactive_geometry(contact,x,H,contact_free_prefix)
            new_fc,new_extra,new_hist,new_ci=contact.evaluate(x,periodic.velocities(v),H,H*0,0.)
            if new_hist or new_ci['normal_energy']+new_ci['tangent_energy']>1e-16:raise ValueError('New quadrature is active at inactive replay checkpoint')
            if np.max(abs(new_fc))>1e-14 or np.max(abs(new_extra))>1e-14:raise ValueError('Inactive quadrature change creates force')
            fc,extra,ci=new_fc,new_extra,new_ci;contact.history=new_hist;contact.accept_geometry(x,H)
        else:restore_contact(contact,restart['contact'])
        if hold_budget_extended and stage_index<len(stage_program):duration=sum(stage['ramp_s']+stage['max_hold_s'] for stage in stage_program)
    try:
        while time<duration-1e-14:
            if c.get('stop_request_file') and Path(c['stop_request_file']).exists():raise RuntimeError('Requested orderly stop; accepted state saved')
            if cycle is None and unload>0 and periodic.prescribed_active and time>=release_time-1e-14:
                release_load=float((periodic.M@a-f+c['damping_per_s']*(periodic.M@v))[periodic.prescribed])
                periodic.prescribed_active=False;periodic.free=np.arange(periodic.ndof);periodic.mass(mass)
            if steps>=c['max_steps']:raise RuntimeError('Step budget exhausted; no extrapolation')
            dt=min(stable_dt(x,t,V0,mass,c,periodic.velocities(v),contact,sigma,b),np.sqrt(.2*c['contact_skin_mm']/max(float(np.max(np.linalg.norm(periodic.velocities(a),axis=1))),1e-30)),duration-time,retry_limit)
            if cycle is not None and cycle.interval_end()-time>1e-14:dt=min(dt,cycle.interval_end()-time)
            if cycle is None and periodic.prescribed_active and unload>0 and release_time-time>1e-14:dt=min(dt,release_time-time)
            if dt<c.get('minimum_dt_s',1e-14):raise RuntimeError('Stable timestep below budget')
            dt_min=min(dt_min,dt);dt_max=max(dt_max,dt);next_time=time+dt
            qp,vp,ap=cycle.displacement(next_time) if cycle is not None else profile(next_time,c)
            qnew=q+dt*v+.5*dt**2*a
            if periodic.prescribed_active:qnew[periodic.prescribed]=qp
            xnew=periodic.positions(qnew)
            if np.linalg.det(deformation(qnew[-6:]))<=0:raise ValueError('Inverted macroscopic cell')
            if np.max(np.linalg.norm(xnew-x,axis=1))>.25*c['contact_skin_mm']:raise RuntimeError('Contact motion-resolution limit exceeded')
            volnew,gradient_new,Dnew=geometry_fn(xnew,t)
            bnew,epnew=advance_fn(b,ep,Dnew@inverse_D,c)
            if epnew.max()>c.get('max_peeq',5.):raise RuntimeError('Trial material strain range exceeded')
            fnew_int,volnew,enew,Jnew,Dnew,snew=internal_fn(xnew,t,bnew,epnew,c)
            if np.max(np.abs(volnew-V0*Jnew)/V0)>1e-8:raise RuntimeError('Material/geometric volume consistency lost')
            if volnew.sum()>np.linalg.det(deformation(qnew[-6:]))*periodic.L**3*(1+1e-7):raise RuntimeError('Nonphysical negative pore volume; no overlap correction permitted')
            vhalf=v+.5*dt*a
            if periodic.prescribed_active:vhalf[periodic.prescribed]=vp
            H=deformation(qnew[-6:])*periodic.L;Hdot=(deformation(vhalf[-6:])-np.eye(3))*periodic.L
            try:
                fc,extra,newhist,cnew=contact.evaluate(xnew,periodic.velocities(vhalf),H,Hdot,dt)
            except SurfaceResolutionError as exc:
                feature=getattr(exc,'feature',None)
                attempt=dict(accepted_time_s=time,trial_time_s=next_time,trial_dt_s=dt,reason=str(exc),feature=feature,repaired=False)
                resolution_attempts.append(attempt)
                if feature is not None and feature.get('requires_quadrature_refinement',False):
                    attempt['repair_stop']='Material quadrature refinement is required; geometry-only splitting cannot add material integration history'
                    raise
                if c.get('surface_remesh_on_edge_guard',False) and feature is not None and feature['kind']=='ee':
                    # Refine the accepted state, discard the failed trial, and
                    # recompute its step with the new mass/kinematics/history.
                    if c.get('surface_remesh_strategy','midpoint')=='closest_quality':
                        from .surface_remesh import closest_edge_candidates
                        candidates=closest_edge_candidates(X,x,t,b,ep,V0,periodic.velocities(v),feature,periodic.L)
                        attempt['candidate_selection']=candidates;attempt['candidate_event_indices']=[]
                        repaired=False
                        for candidate in candidates:
                            if not candidate['geometrically_valid'] or candidate['minimum_quality']<c['minimum_quality']:continue
                            if sum(e.get('accepted',False) and e.get('operation')=='periodic_edge_bisection' for e in remesh_events)>=c.get('surface_remesh_max_events',1):
                                attempt['repair_stop']='surface remesh cap reached';break
                            index=len(remesh_events)
                            repaired=try_surface_remesh(profile(time,c)[2],candidate['edge'],candidate['fraction'],dict(candidate=candidate,ranking=candidates))
                            attempt['candidate_event_indices'].append(index)
                            if repaired:break
                    else:repaired=try_surface_remesh(profile(time,c)[2],feature['ids'][:2])
                    attempt['repaired']=bool(repaired)
                    if repaired:
                        contact_retries+=1
                        if contact_retries>c.get('max_contact_retries',10000):raise RuntimeError('Contact retry budget exhausted')
                        continue
                raise
            except ContactResolutionError as exc:
                if c.get('contact_backend') not in ('distance_barrier','area_guarded','embedded_area'):raise
                contact_retries+=1;retry_limit=dt*.5
                retry_log.append({'accepted_time_s':time,'trial_dt_s':dt,'reason':str(exc)})
                if contact_retries>c.get('max_contact_retries',10000):raise RuntimeError('Contact retry budget exhausted')
                continue
            retry_limit=float('inf')
            if cnew['penetration_mm']>c['max_penetration_mm']:raise RuntimeError('Contact penetration limit exceeded')
            fnew=periodic.force(fnew_int+fc,extra)
            previous_external=external.copy()
            external[:]=0
            if pressure_control:
                phase=min(1.,((goal_load_time+dt) if goal_control else next_time)/c['ramp_s']);smooth=10*phase**3-15*phase**4+6*phase**5
                if goal_control and goal_phase=='hold':smooth=goal_pressure/c['pressure_mpa']
                if unload>0 and next_time>release_time:
                    phase=min(1.,(next_time-release_time)/unload);smooth=1-(10*phase**3-15*phase**4+6*phase**5)
                command_pressure=c['pressure_mpa']*smooth
                if stage_program:
                    stage=stage_program[stage_index];previous=stage_program[stage_index-1]['pressure_mpa'] if stage_index else 0.
                    phase=min(1.,max(0.,(next_time-stage_start)/stage['ramp_s']));smooth=10*phase**3-15*phase**4+6*phase**5
                    command_pressure=previous+(stage['pressure_mpa']-previous)*smooth
                if cauchy_control:external[-6:]=uniaxial_load(deformation(qnew[-6:]),command_pressure,periodic.L**3)
                else:external[periodic.prescribed]=-command_pressure*periodic.L**3
            elif cycle is not None and not periodic.prescribed_active:
                external[periodic.prescribed]=cycle.external_force(next_time)
            elif not periodic.prescribed_active:
                phase=min(1.,max(0.,(next_time-release_time)/unload));smooth=10*phase**3-15*phase**4+6*phase**5
                external[periodic.prescribed]=release_load*(1-smooth)
            anew=periodic.acceleration(fnew+external,vhalf,ap if periodic.prescribed_active else 0.,c['damping_per_s'])
            vnew=vhalf+.5*dt*anew
            if periodic.prescribed_active:vnew[periodic.prescribed]=vp
            r0=(periodic.M@a-f+c['damping_per_s']*(periodic.M@v))[periodic.prescribed]
            r1=(periodic.M@anew-fnew+c['damping_per_s']*(periodic.M@vnew))[periodic.prescribed]
            W+=float(.5*(previous_external+external)@(qnew-q)) if pressure_control or (cycle is not None and not periodic.prescribed_active) else .5*(r0+r1)*(qnew[periodic.prescribed]-q[periodic.prescribed])
            vm=.5*(v+vnew);damp+=dt*c['damping_per_s']*float(vm@(periodic.M@vm))
            friction+=cnew['friction_increment'];released+=cnew['released_history_energy'];algorithmic+=cnew.get('algorithmic_increment',0.);algorithmic_variation+=abs(cnew.get('algorithmic_increment',0.))
            q,v,a,x,b,ep,volume,energy,J,D,sigma,f,ci,f_int=qnew,vnew,anew,xnew,bnew,epnew,volnew,enew,Jnew,Dnew,snew,fnew,cnew,fnew_int
            inverse_D=gradient_new[:,1:,:]
            contact_free_prefix=contact_free_prefix and not bool(newhist) and cnew.get('minimum_separation_mm',0)>=c.get('barrier_thickness_mm',.02)
            contact.history=newhist
            if hasattr(contact,'accept_geometry'):contact.accept_geometry(x,H)
            time=next_time;steps+=1
            if (c.get('remesh_every',0) and steps%c['remesh_every']==0) or (c.get('volume_reconnections',False) and quality_fn(x,t).min()<c['minimum_quality']):
                for repair in range(c.get('maximum_reconnections_per_step',1)):
                    if not try_remesh(ap):break
            if steps in c.get('surface_remesh_steps',[]) or (c.get('surface_remesh_every',0) and steps%c['surface_remesh_every']==0 and quality_fn(x,t).min()<c.get('surface_remesh_quality',.35)):
                try_surface_remesh(ap)
            kinetic_now=float(.5*v@(periodic.M@v));internal_now=float(V0@energy)
            residual_now=kinetic_now+internal_now+ci['normal_energy']+ci['tangent_energy']+damp+friction+released+algorithmic-initial_energy-W
            audit_rows.append([time,-q[periodic.prescribed],1-volume.sum()/(np.linalg.det(deformation(q[-6:]))*periodic.L**3),command_pressure if pressure_control else -external[periodic.prescribed]/periodic.L**3,kinetic_now,internal_now,W,residual_now,ci['normal_energy'],ci['tangent_energy'],ci['contacts']]);coulomb_peak=max(coulomb_peak,ci['max_coulomb_ratio'])
            if quality_fn(x,t).min()<c['minimum_quality']:raise RuntimeError('Mesh quality exhausted; local remeshing cannot repair this boundary distortion')
            if cycle is not None:
                reaction=periodic.M@a-f+c['damping_per_s']*(periodic.M@v)
                residual=(f+external).copy();residual[-6:]/=periodic.L
                force_scale=max(abs(float(reaction[periodic.prescribed])),cycle.state['peak_force'],c['young_mpa']*periodic.L**3*1e-9)/periodic.L
                force_ratio=float(np.linalg.norm(residual[periodic.free]))/force_scale
                scale=max(abs(W),abs(internal_now-history[0][5]),initial_available_energy,c['young_mpa']*initial_V0*1e-12)
                available=max(abs(internal_now-history[0][5])+ci['normal_energy']+ci['tangent_energy'],c['young_mpa']*initial_V0*1e-12)
                event=cycle.observe(time,float(audit_rows[-1][2]),float(-q[periodic.prescribed]),float(reaction[periodic.prescribed]),force_ratio,kinetic_now/available,abs(residual_now)/scale)
                if event is not None:
                    cycle.state['records'][-1].update(cauchy_stress_MPa=diagnostic_cauchy(deformation(q[-6:]),-f[-6:],periodic.L**3),
                         contacts=int(ci['contacts']),minimum_quality=float(quality_fn(x,t).min()),
                         accepted_steps=steps)
                if event=='release':
                    cycle.release(time,float(reaction[periodic.prescribed]));periodic.prescribed_active=False;periodic.free=np.arange(periodic.ndof);periodic.mass(mass)
                    external[periodic.prescribed]=cycle.state['unload_force'];a=periodic.acceleration(f+external,v,0.,c['damping_per_s'])
                if event=='complete':duration=time
                if event is not None and c.get('save_checkpoint',False):checkpoint(locals())
            if goal_control:
                if goal_phase=='ramp':goal_load_time=min(c['ramp_s'],goal_load_time+dt)
                phi=1-volume.sum()/(np.linalg.det(deformation(q[-6:]))*periodic.L**3)
                if goal_phase=='ramp' and (phi<=c['target_final_porosity'] or goal_load_time>=c['ramp_s']):
                    goal_phase='hold';goal_hold_start=time;goal_pressure=-external[periodic.prescribed]/periodic.L**3
                elif goal_phase=='hold' and time-goal_hold_start>=c['hold_s']:
                    residual=(f+external).copy();residual[-6:]/=periodic.L
                    force_ratio=np.linalg.norm(residual[periodic.free])/max(goal_pressure*periodic.L**2,c['young_mpa']*periodic.L**2*1e-9)
                    kinetic_ratio=kinetic_now/max(abs(internal_now-history[0][5]),c['young_mpa']*initial_V0*1e-12)
                    if force_ratio<c.get('force_tolerance',.02) and kinetic_ratio<c['kinetic_tolerance']:
                        if phi<=c['target_final_porosity']:goal_reached=True;duration=time
                        elif goal_load_time<c['ramp_s']:goal_phase='ramp'
                        else:raise RuntimeError('Porosity target not reached at equilibrated pressure cap')

            if stage_program:
                stage=stage_program[stage_index];elapsed=time-stage_start
                if elapsed>=stage['ramp_s']+stage['hold_s']:
                    residual=(f+external).copy();residual[-6:]/=periodic.L
                    force_ratio=float(np.linalg.norm(residual[periodic.free]))/max(stage['pressure_mpa']*periodic.L**2,1e-30)
                    available=max(abs(internal_now-history[0][5])+ci['normal_energy']+ci['tangent_energy'],c['young_mpa']*initial_V0*1e-12)
                    kinetic_ratio=kinetic_now/available
                    scale=max(abs(W),abs(internal_now-history[0][5]),initial_available_energy,c['young_mpa']*initial_V0*1e-12)
                    stage_checks=dict(force=bool(force_ratio<c.get('force_tolerance',.02)),kinetic=bool(kinetic_ratio<c['kinetic_tolerance']),energy=bool(abs(residual_now)/scale<c['energy_tolerance']))
                    if all(stage_checks.values()) or elapsed>=stage['ramp_s']+stage['max_hold_s']:
                        stage_records.append(dict(pressure_MPa=command_pressure,time_s=time,porosity=float(audit_rows[-1][2]),force_ratio=force_ratio,kinetic_ratio=kinetic_ratio,checks=stage_checks,passed=all(stage_checks.values())))
                        if not all(stage_checks.values()):raise RuntimeError('Pressure-stage equilibration budget exhausted')
                        stage_index+=1;stage_start=time
                        if stage_index==len(stage_program):duration=time
                        if c.get('save_checkpoint',False):checkpoint(locals())
            if c.get('checkpoint_every',0) and steps%c['checkpoint_every']==0:checkpoint(locals())
            if c.get('checkpoint_free_every',0) and contact_free_prefix and steps%c['checkpoint_free_every']==0:checkpoint(locals())
            if c.get('checkpoint_stop_after') is not None and steps>=c['checkpoint_stop_after']:
                checkpoint(locals());raise RuntimeError('Requested checkpoint stop; continuation state saved')
            if steps%c['output_every']==0:
                history.append(snapshot());log_progress()
        if not history or history[-1][0]!=time:history.append(snapshot())
    except (ValueError,RuntimeError,np.linalg.LinAlgError,KeyboardInterrupt) as exc:
        failure=repr(exc);failure_feature=getattr(exc,'feature',None)
        if history[-1][0]!=time:history.append(snapshot())
    if c.get('save_checkpoint',False):checkpoint(locals())
    if goal_control and not goal_reached and failure is None:failure='Porosity control time budget exhausted'
    if cycle is not None and not cycle.state['complete'] and failure is None:failure='Porosity cycle time budget exhausted'
    h=np.array(history);audits=np.array(audit_rows);internal_change=np.abs(audits[:,5]-h[0,5])
    work_scale=max(float(np.max(np.abs(audits[:,6]))),float(np.max(internal_change)),initial_available_energy,c['young_mpa']*initial_V0*1e-12)
    loaded=(audits[:,3]>.5*np.max(audits[:,3])) if goal_control else ((audits[:,0]>.5*c['ramp_s']) if pressure_control else (audits[:,1]>.5*c['compression_fraction']))
    if cycle is not None:loaded=audits[:,1]>.5*max(float(np.max(audits[:,1])),1e-12)
    ke_ratio=float(np.max(audits[loaded,4]/np.maximum(internal_change[loaded],work_scale*1e-8))) if np.any(loaded) else float('inf')
    from .explicit_audit import whole_path_audit
    whole_path=whole_path_audit(audits,work_scale)
    scaled=(f+external).copy();scaled[-6:]/=periodic.L
    final_reaction=periodic.M@a-f+c['damping_per_s']*(periodic.M@v)
    force_scale=max(abs(final_reaction[periodic.prescribed])/periodic.L,c['young_mpa']*periodic.L**2*max(c['compression_fraction'],1e-12)*1e-6)
    if cycle is not None:force_scale=max(force_scale,cycle.state['peak_force']/periodic.L)
    static_ratio=float(np.linalg.norm(scaled[periodic.free]))/force_scale
    boundary_error=abs(closed_surface_volume(x,t)-volume.sum())/volume.sum()
    material_volume_error=float(np.max(np.abs(volume-V0*J)/V0))
    checks={'closed_boundary_volume':boundary_error<1e-10,'material_volume_consistency':material_volume_error<1e-8,'static_equilibrium':static_ratio<c.get('force_tolerance',.02),'path_completed':failure is None,'positive_J':bool(np.all(h[:,14]>0)),
            'physical_porosity':bool(np.all((h[:,2]>=-1e-7)&(h[:,2]<1))),
            'energy_balance':bool(np.max(np.abs(audits[:,7]))/work_scale<c['energy_tolerance']),
            'low_kinetic_energy':ke_ratio<c['kinetic_tolerance'],
            'coulomb_bound':bool(coulomb_peak<=c['friction']+1e-10),
            'reference_volume_conserved':bool(abs(V0.sum()-initial_V0)/initial_V0<1e-10),
            'penetration_bound':bool(np.max(h[:,16])<=c['max_penetration_mm'])}
    checks['contact_numerical_energy_small']=(algorithmic_variation+released)/work_scale<c['energy_tolerance']
    if c.get('volume_reconnections',False):checks['volume_remesh_energy_small']=sum(abs(e.get('material_energy_increment_Nmm',0.)) for e in remesh_events if e.get('accepted'))/work_scale<c['energy_tolerance']
    if pressure_control and not goal_control:checks['requested_pressure_reached']=bool(failure is None and np.max(audits[:,3])>=c['pressure_mpa']*(1-1e-10))
    if goal_control:checks['goal_equilibrium_reached']=goal_reached
    if 'target_final_porosity' in c and cycle is None:checks['target_porosity_at_final']=bool(failure is None and h[-1,2]<=c['target_final_porosity'])
    if unload>0:checks['unload_completed']=failure is None and not periodic.prescribed_active and time>=duration-1e-14
    if 'target_final_porosity' in c:checks['target_porosity_observed']=bool(np.any((audits[:,2]>=0)&(audits[:,2]<=c['target_final_porosity'])))
    if cycle is not None:
        checks['loaded_porosity_equilibrated']=cycle.state['loaded_target']
        checks['cycle_completed']=cycle.state['complete']
        checks['unload_completed']=cycle.state['complete']
    checks={k:bool(value) for k,value in checks.items()}
    result={'hold_budget_extended':bool(hold_budget_extended),'checkpoint_contact_operator_update':operator_update,'pressure_stages':stage_records,'checkpoint_paths':checkpoint_paths,'whole_path_audit':whole_path,'accepted_step_audit_file':'accepted-step-audit.csv.gz','pressure_measure':'cauchy' if cauchy_control else 'nominal_reference', 'peak_command_pressure_MPa':float(np.max(audits[:,3])) if pressure_control else None, 'final_work_equivalent_cauchy_stress_MPa':diagnostic_cauchy(deformation(q[-6:]),-f[-6:],periodic.L**3), 'final_external_work_equivalent_cauchy_stress_MPa':diagnostic_cauchy(deformation(q[-6:]),external[-6:],periodic.L**3), 'area_contact_scope':('embedded_material_quadrature_with_adaptive_normal_audit_and_friction' if c.get('contact_backend')=='embedded_area' else 'single_closest_patch_with_edge_collision_guard' if c.get('contact_backend')=='area_guarded' else None),'mass_frame':c.get('mass_frame','pinned'),'porosity_goal_equilibrated':goal_reached if goal_control else None,'peak_nominal_pressure_MPa':float(np.max(h[:,3])) if pressure_control else None,'contact_quadrature_variation_Nmm':algorithmic_variation,'contact_released_history_Nmm':released,'contact_neighbor_builds':getattr(contact,'neighbor_builds',None),'audit_sampling':'every_accepted_step','loading_mode':c.get('loading_mode','displacement'),'contact_retries':contact_retries,'contact_backend':c.get('contact_backend','legacy'),'edge_edge_response_implemented':c.get('contact_backend') in ('tracked_edges','distance_barrier'),'edge_response_scope':('collision_guard_only_no_edge_spring' if c.get('contact_backend')=='area_guarded' else 'unsigned_distance_barrier_all_closest_features' if c.get('contact_backend')=='distance_barrier' else ('interior_only_handoff_unvalidated' if c.get('contact_backend')=='tracked_edges' else 'none')),'status':'prototype_checks_passed_not_mesh_or_contact_validated' if all(checks.values()) else 'diagnostics_only',
            'checks':checks,'failure':failure,'failure_feature':failure_feature,'surface_resolution_attempts':resolution_attempts,'steps':steps,'dt_min_s':dt_min,'dt_max_s':dt_max,'kinetic_internal_ratio_loaded':ke_ratio if np.isfinite(ke_ratio) else None,
            'closed_boundary_volume_relative_error':boundary_error,'material_volume_relative_error':material_volume_error,'final_static_residual_ratio':float(static_ratio),'energy_scale_Nmm':work_scale,'initial_available_energy_Nmm':initial_available_energy,'energy_absolute_residual_Nmm':float(np.max(np.abs(audits[:,7]))),'energy_relative_error':float(np.max(np.abs(audits[:,7]))/work_scale),'remesh_events':remesh_events,
            'initial_porosity':float(h[0,2]),'final_porosity':float(h[-1,2]),'final_axial_compression':float(h[-1,1]),'residual_compression_strain':float(h[-1,1]) if unload>0 and checks.get('unload_completed') and checks['static_equilibrium'] else None,
            'limitations':['linear-tetra plastic locking not converged','contact quadrature, barrier thickness, mollification and general friction convergence not validated','surface edge bisection is experimental; no coarsening or general distortion repair','numerical density and damping are not measured material data','uncalibrated Kirchhoff-yield Hencky J2 material','not a verified 40 to 10 percent compaction result']}
    np.savetxt(directory/'history.csv',h,delimiter=',',comments='',header='time_s,axial_compression,porosity,nominal_compression_MPa,kinetic_Nmm,internal_Nmm,external_work_Nmm,damping_Nmm,friction_Nmm,released_contact_history_Nmm,normal_contact_Nmm,tangent_contact_Nmm,energy_residual_Nmm,peak_peeq,min_elastic_J,min_quality,penetration_mm,coulomb_ratio,contacts,elements,free_dynamic_residual_N,contact_algorithmic_Nmm,edge_contacts,minimum_separation_mm,command_pressure_MPa')
    np.savetxt(directory/'accepted-step-audit.csv.gz',audits,delimiter=',',comments='',header='time_s,axial_compression,porosity,command_pressure_MPa,kinetic_Nmm,internal_Nmm,external_work_Nmm,energy_residual_Nmm,normal_contact_Nmm,tangent_contact_Nmm,contacts')
    np.savez_compressed(directory/'last-state.npz',points=x,reference_points=X,tets=t,elastic_b=b,peeq=ep,reference_volume=V0,generalized_position=q,generalized_velocity=v)
    if c.get('contact_backend') in ('area_guarded','embedded_area'):
        np.savez_compressed(directory/'contact-quadrature.npz',reference_points=X,area_weights=contact.integration_area0,mode=np.array(contact.quadrature_mode))
    if c.get('contact_backend')=='embedded_area':
        (directory/'embedded-quadrature-rules.json').write_text(json.dumps(contact.rules)+'\n')
    if c.get('save_contact_audit',False):
        from .contact_audit import save_capture
        save_capture(directory/'contact-audit.json',contact,f,external,time,steps)
    if resolution_attempts:(directory/'surface-resolution-attempts.json').write_text(json.dumps(resolution_attempts,indent=2)+'\n')
    if retry_log:(directory/'contact-retries.json').write_text(json.dumps(retry_log,indent=2)+'\n')
    from .contact_audit import plain
    result=plain(result)
    if cycle is not None:result['porosity_cycle']=plain(cycle.state)
    if cycle is not None:(directory/'porosity-cycle.json').write_text(json.dumps(result['porosity_cycle'],indent=2)+'\n')
    (directory/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    import meshio
    meshio.write(directory/'last-mesh.vtu',meshio.Mesh(x,[('tetra',t)],cell_data={'peeq':[ep],'cauchy_stress_MPa':[sigma.reshape(-1,9)]}))
    return result
