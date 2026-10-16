"""Research-only explicit compatible-tetra FEM, periodic contact and remeshing.

Velocity Verlet with a reduced consistent constraint mass. No global stiffness
iterations. Material points use finite-strain Hencky J2; optional distance
barriers cover point-face and edge-edge contact. Remeshing is boundary-preserving
internal 2-to-3 reconnection only.
"""
import json
from pathlib import Path
import numpy as np
from .explicit_material import geometry,advance,internal_force,constants,response
from .explicit_periodic import Periodic,deformation
from .periodic_contact import Contact
from .adaptive_tet import propose,quality,closed_surface_volume


def lumped_mass(t,V0,rho,nn):
    result=np.zeros(nn);np.add.at(result,t.ravel(),np.repeat(rho*V0/4,4));return result


def profile(time,c):
    T=c['ramp_s'];h=c.get('hold_s',0.)
    if time>=T:return -c['compression_fraction'],0.,0.
    s=max(0,time/T);amplitude=c['compression_fraction']
    return -amplitude*(10*s**3-15*s**4+6*s**5),-amplitude*(30*s**2-60*s**3+30*s**4)/T,-amplitude*(60*s-180*s**2+120*s**3)/T**2


def stable_dt(x,t,V0,mass,c,velocity,contact,stress=None,elastic_b=None):
    v,_,_=geometry(x,t);p=x[t];faces=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
    triangles=p[:,faces];areas=np.linalg.norm(np.cross(triangles[:,:,1]-triangles[:,:,0],triangles[:,:,2]-triangles[:,:,0]),axis=2)/2
    height=3*v/areas.max(axis=1);G,K=constants(c);rho=c['density_tonne_mm3']*V0/v
    wave=np.sqrt((K+4*G/3)/rho)*np.maximum(1.,V0/v)
    bulk=float(np.min(height/wave));contact_bound=float('inf')
    if c.get('timestep_method')=='assembled_mass_bound':
        from .explicit_timestep import assembled_frequency_bound
        bulk=2/np.sqrt(assembled_frequency_bound(x,t,mass,c,stress,elastic_b))
    if len(contact.faces):
        f=x[contact.faces];A=np.linalg.norm(np.cross(f[:,1]-f[:,0],f[:,2]-f[:,0]),axis=1)/2
        nodeA=np.zeros(len(x))
        for k in range(3):np.add.at(nodeA,contact.faces[:,k],A/3)
        nodeA=np.maximum(nodeA,contact.node_area0)
        nodes=contact.nodes
        diagonal=2*c['contact_penalty_mpa_mm']*nodeA[nodes]
        if hasattr(contact,'edge_stiffness_diagonal'):diagonal+=contact.edge_stiffness_diagonal()[nodes]
        contact_bound=float(np.sqrt(np.min(mass[nodes]/diagonal)))
    motion=c['contact_skin_mm']*.1/max(float(np.max(np.linalg.norm(velocity,axis=1))),1e-30)
    return min(c['cfl']*bulk,c['cfl']*contact_bound,motion,c.get('maximum_dt_s',float('inf')))


def validate(c):
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
    if not isinstance(c.get('remesh_every',0),int) or c.get('remesh_every',0)<0:raise ValueError('Invalid remesh interval')
    if c.get('unload_s',0)<0 or c.get('recovery_hold_s',0)<0:raise ValueError('Negative unloading time')
    if c.get('recovery_hold_s',0)>0 and c.get('unload_s',0)<=0:raise ValueError('Recovery hold needs an unloading segment')
    if c.get('contact_backend','legacy') not in ('legacy','tracked_guarded','tracked_edges','distance_barrier'):raise ValueError('Unknown contact backend')
    if not np.isfinite(c.get('edge_penalty_scale',1.)) or c.get('edge_penalty_scale',1.)<=0:raise ValueError('Positive edge penalty scale required')
    if c.get('loading_mode','displacement') not in ('displacement','nominal_pressure','porosity_target'):raise ValueError('Unknown loading mode')
    if c.get('loading_mode') in ('nominal_pressure','porosity_target') and (not np.isfinite(c.get('pressure_mpa',0)) or c.get('pressure_mpa',0)<=0):raise ValueError('Positive finite pressure required')
    if c.get('timestep_method','element_wave') not in ('element_wave','assembled_mass_bound'):raise ValueError('Unknown timestep method')
    if c.get('loading_mode')=='porosity_target' and (not 0<c.get('target_final_porosity',-1)<1 or c.get('hold_s',0)<=0 or c.get('unload_s',0)>0):raise ValueError('Porosity control needs a physical target, positive equilibration hold and no unloading')
    constants(c)


def solve(directory,X,t,c):
    validate(c);directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(directory/'initial-mesh.npz',points=X,tets=t)
    (directory/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    V0,_,D=geometry(X,t);initial_V0=float(V0.sum());periodic=Periodic(X,c['length_mm'],mass_frame=c.get('mass_frame','pinned'))
    pressure_control=c.get('loading_mode') in ('nominal_pressure','porosity_target')
    goal_control=c.get('loading_mode')=='porosity_target'
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
    if c.get('contact_backend')=='distance_barrier':
        from .barrier_contact import BarrierContact
        contact_type=BarrierContact;contact_kwargs={'thickness':c.get('barrier_thickness_mm',.02),'max_candidates':c.get('contact_candidate_budget',500000)}
    contact=contact_type(X,t,periodic,c['contact_penalty_mpa_mm'],c['friction'],c['contact_skin_mm'],**contact_kwargs)
    b=np.repeat(np.eye(3)[None],len(t),axis=0);ep=np.full(len(t),float(c.get('initial_peeq',0.)));q=np.zeros(periodic.ndof);v=q.copy();x=X.copy()
    if 'initial_velocity_mm_s' in c:
        initial_velocity=np.asarray(c['initial_velocity_mm_s'],float)
        if initial_velocity.shape!=X.shape or not np.all(np.isfinite(initial_velocity)):raise ValueError('Invalid manufactured initial velocity')
        v=periodic.remap_velocity(periodic.force(mass[:,None]*initial_velocity),0.)
    f_int,volume,energy,J,D,sigma=internal_force(x,t,b,ep,c)
    fc,extra,hist,ci=contact.evaluate(x,periodic.velocities(v),np.eye(3)*periodic.L,np.zeros((3,3)),0.)
    contact.history=hist
    if hasattr(contact,'accept_geometry'):contact.accept_geometry(x,np.eye(3)*periodic.L)
    f=periodic.force(f_int+fc,extra);a=periodic.acceleration(f,v,0.,c['damping_per_s'])
    release_time=c['ramp_s']+c.get('hold_s',0.);unload=c.get('unload_s',0.)
    duration=release_time+unload+c.get('recovery_hold_s',0.);release_load=None;external=np.zeros(periodic.ndof)
    goal_phase='ramp';goal_load_time=0.;goal_hold_start=0.;goal_pressure=0.;goal_reached=False
    if goal_control:duration=c.get('max_goal_time_s',10*(c['ramp_s']+c['hold_s']))
    retry_limit=float('inf');contact_retries=0;retry_log=[];algorithmic_variation=0.
    from .barrier_contact import ContactResolutionError
    time=0.;W=0.;damp=0.;friction=0.;released=0.;algorithmic=0.;steps=0;remesh_events=[];history=[];failure=None
    def try_remesh(ap):
        nonlocal t,b,ep,V0,mass,v,f_int,volume,energy,J,D,sigma,f,a
        proposal=propose(x,t,b,ep,V0,c,c['remesh_quality'],c['remesh_energy_tolerance'])
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
            if max(error,perr,lerr)<=c['remesh_energy_tolerance']:
                # Free surface facets do not change in a 2-to-3 flip,
                # so retain contact history rather than resetting it.
                event.update(time_s=time,kinetic_energy_scaled_error=error,accepted=True)
                t,b,ep,V0,mass,v=t2,b2,ep2,V02,mass2,v2
                contact.update_topology(t)
                f_int,volume,energy,J,D,sigma=internal_force(x,t,b,ep,c)
                f=periodic.force(f_int+fc,extra);a=periodic.acceleration(f+external,v,ap if periodic.prescribed_active else 0.,c['damping_per_s'])
            else:
                event.update(time_s=time,kinetic_energy_scaled_error=error,accepted=False,reason='momentum projection energy error')
                periodic.mass(mass)
            remesh_events.append(event)

    if c.get('initial_remesh',False):try_remesh(0.)
    initial_energy=float(V0@energy)+ci['normal_energy']+ci['tangent_energy']+float(.5*v@(periodic.M@v));dt_min=float('inf');dt_max=0.
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
                ci['normal_energy'],ci['tangent_energy'],total-W,float(ep.max()),float(J.min()),float(quality(x,t).min()),
                ci['penetration_mm'],ci['max_coulomb_ratio'],ci['contacts'],len(t),float(np.max(np.abs(dynamic[periodic.free]))),algorithmic,ci.get('edge_contacts',0),ci.get('minimum_separation_mm',float('nan'))]
    history.append(snapshot());audit_rows=[history[0][0:7]+[history[0][12]]];coulomb_peak=ci['max_coulomb_ratio']
    progress=directory/'progress.jsonl';progress.write_text('')
    def log_progress():
        row=history[-1]
        with progress.open('a') as stream:stream.write(json.dumps({'step':steps,'time_s':time,'axial_compression':row[1],'porosity':row[2],'contacts':int(row[18]),'edge_contacts':int(row[22]),'elements':len(t),'min_quality':row[15],'nominal_pressure_MPa':row[3],'kinetic_Nmm':row[4],'internal_Nmm':row[5],'energy_residual_Nmm':row[12]})+'\n')
    log_progress()
    try:
        while time<duration-1e-14:
            if unload>0 and periodic.prescribed_active and time>=release_time-1e-14:
                release_load=float((periodic.M@a-f+c['damping_per_s']*(periodic.M@v))[periodic.prescribed])
                periodic.prescribed_active=False;periodic.free=np.arange(periodic.ndof);periodic.mass(mass)
            if steps>=c['max_steps']:raise RuntimeError('Step budget exhausted; no extrapolation')
            dt=min(stable_dt(x,t,V0,mass,c,periodic.velocities(v),contact,sigma,b),np.sqrt(.2*c['contact_skin_mm']/max(float(np.max(np.linalg.norm(periodic.velocities(a),axis=1))),1e-30)),duration-time,retry_limit)
            if periodic.prescribed_active and unload>0 and release_time-time>1e-14:dt=min(dt,release_time-time)
            if dt<c.get('minimum_dt_s',1e-14):raise RuntimeError('Stable timestep below budget')
            dt_min=min(dt_min,dt);dt_max=max(dt_max,dt);next_time=time+dt
            qp,vp,ap=profile(next_time,c);qnew=q+dt*v+.5*dt**2*a
            if periodic.prescribed_active:qnew[periodic.prescribed]=qp
            xnew=periodic.positions(qnew)
            if np.linalg.det(deformation(qnew[-6:]))<=0:raise ValueError('Inverted macroscopic cell')
            if np.max(np.linalg.norm(xnew-x,axis=1))>.25*c['contact_skin_mm']:raise RuntimeError('Contact motion-resolution limit exceeded')
            volnew,_,Dnew=geometry(xnew,t)
            bnew,epnew=advance(b,ep,Dnew@np.linalg.inv(D),c)
            if epnew.max()>c.get('max_peeq',5.):raise RuntimeError('Trial material strain range exceeded')
            fnew_int,volnew,enew,Jnew,Dnew,snew=internal_force(xnew,t,bnew,epnew,c)
            if np.max(np.abs(volnew-V0*Jnew)/V0)>1e-8:raise RuntimeError('Material/geometric volume consistency lost')
            if volnew.sum()>np.linalg.det(deformation(qnew[-6:]))*periodic.L**3*(1+1e-7):raise RuntimeError('Nonphysical negative pore volume; no overlap correction permitted')
            vhalf=v+.5*dt*a
            if periodic.prescribed_active:vhalf[periodic.prescribed]=vp
            H=deformation(qnew[-6:])*periodic.L;Hdot=(deformation(vhalf[-6:])-np.eye(3))*periodic.L
            try:
                fc,extra,newhist,cnew=contact.evaluate(xnew,periodic.velocities(vhalf),H,Hdot,dt)
            except ContactResolutionError as exc:
                if c.get('contact_backend')!='distance_barrier':raise
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
                external[periodic.prescribed]=-c['pressure_mpa']*periodic.L**3*smooth
            elif not periodic.prescribed_active:
                phase=min(1.,max(0.,(next_time-release_time)/unload));smooth=10*phase**3-15*phase**4+6*phase**5
                external[periodic.prescribed]=release_load*(1-smooth)
            anew=periodic.acceleration(fnew+external,vhalf,ap if periodic.prescribed_active else 0.,c['damping_per_s'])
            vnew=vhalf+.5*dt*anew
            if periodic.prescribed_active:vnew[periodic.prescribed]=vp
            r0=(periodic.M@a-f+c['damping_per_s']*(periodic.M@v))[periodic.prescribed]
            r1=(periodic.M@anew-fnew+c['damping_per_s']*(periodic.M@vnew))[periodic.prescribed]
            W+=float(.5*(previous_external+external)@(qnew-q)) if pressure_control else .5*(r0+r1)*(qnew[periodic.prescribed]-q[periodic.prescribed])
            vm=.5*(v+vnew);damp+=dt*c['damping_per_s']*float(vm@(periodic.M@vm))
            friction+=cnew['friction_increment'];released+=cnew['released_history_energy'];algorithmic+=cnew.get('algorithmic_increment',0.);algorithmic_variation+=abs(cnew.get('algorithmic_increment',0.))
            q,v,a,x,b,ep,volume,energy,J,D,sigma,f,ci=qnew,vnew,anew,xnew,bnew,epnew,volnew,enew,Jnew,Dnew,snew,fnew,cnew
            contact.history=newhist
            if hasattr(contact,'accept_geometry'):contact.accept_geometry(x,H)
            time=next_time;steps+=1
            if c.get('remesh_every',0) and steps%c['remesh_every']==0:try_remesh(ap)
            kinetic_now=float(.5*v@(periodic.M@v));internal_now=float(V0@energy)
            residual_now=kinetic_now+internal_now+ci['normal_energy']+ci['tangent_energy']+damp+friction+released+algorithmic-initial_energy-W
            audit_rows.append([time,-q[periodic.prescribed],0.,-external[periodic.prescribed]/periodic.L**3,kinetic_now,internal_now,W,residual_now]);coulomb_peak=max(coulomb_peak,ci['max_coulomb_ratio'])
            if quality(x,t).min()<c['minimum_quality']:raise RuntimeError('Mesh quality exhausted; local remeshing cannot repair this boundary distortion')
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

            if steps%c['output_every']==0:
                history.append(snapshot());log_progress()
        if not history or history[-1][0]!=time:history.append(snapshot())
    except (ValueError,RuntimeError,np.linalg.LinAlgError,KeyboardInterrupt) as exc:
        failure=repr(exc)
        if history[-1][0]!=time:history.append(snapshot())
    if goal_control and not goal_reached and failure is None:failure='Porosity control time budget exhausted'
    h=np.array(history);audits=np.array(audit_rows);internal_change=np.abs(audits[:,5]-h[0,5])
    work_scale=max(float(np.max(np.abs(audits[:,6]))),float(np.max(internal_change)),initial_available_energy,c['young_mpa']*initial_V0*1e-12)
    loaded=(audits[:,3]>.5*np.max(audits[:,3])) if goal_control else ((audits[:,0]>.5*c['ramp_s']) if pressure_control else (audits[:,1]>.5*c['compression_fraction']))
    ke_ratio=float(np.max(audits[loaded,4]/np.maximum(internal_change[loaded],work_scale*1e-8))) if np.any(loaded) else float('inf')
    scaled=(f+external).copy();scaled[-6:]/=periodic.L
    final_reaction=periodic.M@a-f+c['damping_per_s']*(periodic.M@v)
    force_scale=max(abs(final_reaction[periodic.prescribed])/periodic.L,c['young_mpa']*periodic.L**2*max(c['compression_fraction'],1e-12)*1e-6)
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
    if pressure_control and not goal_control:checks['requested_pressure_reached']=bool(failure is None and np.max(audits[:,3])>=c['pressure_mpa']*(1-1e-10))
    if goal_control:checks['goal_equilibrium_reached']=goal_reached
    if 'target_final_porosity' in c:checks['target_porosity_at_final']=bool(failure is None and h[-1,2]<=c['target_final_porosity'])
    if unload>0:checks['unload_completed']=failure is None and not periodic.prescribed_active and time>=duration-1e-14
    if 'target_final_porosity' in c:checks['target_porosity_observed']=bool(np.any((h[:,2]>=0)&(h[:,2]<=c['target_final_porosity'])))
    checks={k:bool(value) for k,value in checks.items()}
    result={'mass_frame':c.get('mass_frame','pinned'),'porosity_goal_equilibrated':goal_reached if goal_control else None,'peak_nominal_pressure_MPa':float(np.max(audits[:,3])) if pressure_control else None,'contact_quadrature_variation_Nmm':algorithmic_variation,'contact_released_history_Nmm':released,'contact_neighbor_builds':getattr(contact,'neighbor_builds',None),'audit_sampling':'every_accepted_step','loading_mode':c.get('loading_mode','displacement'),'contact_retries':contact_retries,'contact_backend':c.get('contact_backend','legacy'),'edge_edge_response_implemented':c.get('contact_backend') in ('tracked_edges','distance_barrier'),'edge_response_scope':('unsigned_distance_barrier_all_closest_features' if c.get('contact_backend')=='distance_barrier' else ('interior_only_handoff_unvalidated' if c.get('contact_backend')=='tracked_edges' else 'none')),'status':'prototype_checks_passed_not_mesh_or_contact_validated' if all(checks.values()) else 'diagnostics_only',
            'checks':checks,'failure':failure,'steps':steps,'dt_min_s':dt_min,'dt_max_s':dt_max,'kinetic_internal_ratio_loaded':ke_ratio if np.isfinite(ke_ratio) else None,
            'closed_boundary_volume_relative_error':boundary_error,'material_volume_relative_error':material_volume_error,'final_static_residual_ratio':float(static_ratio),'energy_scale_Nmm':work_scale,'initial_available_energy_Nmm':initial_available_energy,'energy_absolute_residual_Nmm':float(np.max(np.abs(audits[:,7]))),'energy_relative_error':float(np.max(np.abs(audits[:,7]))/work_scale),'remesh_events':remesh_events,
            'initial_porosity':float(h[0,2]),'final_porosity':float(h[-1,2]),'final_axial_compression':float(h[-1,1]),'residual_compression_strain':float(h[-1,1]) if unload>0 and checks.get('unload_completed') and checks['static_equilibrium'] else None,
            'limitations':['linear-tetra plastic locking not converged','contact quadrature, barrier thickness, mollification and general friction convergence not validated','internal 2-to-3 reconnection only; no surface remeshing','numerical density and damping are not measured material data','uncalibrated Kirchhoff-yield Hencky J2 material','not a verified 40 to 10 percent compaction result']}
    np.savetxt(directory/'history.csv',h,delimiter=',',comments='',header='time_s,axial_compression,porosity,nominal_compression_MPa,kinetic_Nmm,internal_Nmm,external_work_Nmm,damping_Nmm,friction_Nmm,released_contact_history_Nmm,normal_contact_Nmm,tangent_contact_Nmm,energy_residual_Nmm,peak_peeq,min_elastic_J,min_quality,penetration_mm,coulomb_ratio,contacts,elements,free_dynamic_residual_N,contact_algorithmic_Nmm,edge_contacts,minimum_separation_mm')
    np.savez_compressed(directory/'last-state.npz',points=x,reference_points=X,tets=t,elastic_b=b,peeq=ep,reference_volume=V0,generalized_position=q,generalized_velocity=v)
    if retry_log:(directory/'contact-retries.json').write_text(json.dumps(retry_log,indent=2)+'\n')
    (directory/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    import meshio
    meshio.write(directory/'last-mesh.vtu',meshio.Mesh(x,[('tetra',t)],cell_data={'peeq':[ep],'cauchy_stress_MPa':[sigma.reshape(-1,9)]}))
    return result
