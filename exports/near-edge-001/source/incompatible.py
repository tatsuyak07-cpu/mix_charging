"""Experimental periodic C3D8I: expose native modes and verify all 11 gradients.

Source reference: CalculiX shape8hu.f and genmodes.f at
fd588701edd6a92fde06e99b1bdf01f3b0f80f2c. Native 2.20 E output is the
independent runtime check; source correspondence alone is not acceptance.
"""
import json
from pathlib import Path
import subprocess
import numpy as np
from . import plastic,sintered
from .cli import solver_environment
from .continuum import GAUSS,SIGNS,jacobians,deformation_gradients,stress_tensors
from .results import read_dat,find


def mode_gradients(points,elements):
    center=np.einsum('eni,nj->eij',points[elements],SIGNS/8)
    reference=jacobians(points,elements)
    ratio=np.linalg.det(center)[:,None]/np.linalg.det(reference)
    return -2*GAUSS[None,:,:,None]*np.linalg.inv(center)[:,None,:,:]*ratio[:,:,None,None]


def read_restart_groups(path):
    from scipy.io import FortranFile,FortranEOFError
    current=[]
    with FortranFile(path,'r') as f:
        while True:
            try:data=f.read_record(np.uint8).tobytes()
            except FortranEOFError:break
            if len(data)==80 and data.startswith(b'Version'):
                if current:yield current
                current=[]
            current.append(data)
    if current:yield current


def restart_state(records,points,elements,dat_u,dat_macro):
    """Fail closed on version/layout, topology, coordinates and native DAT mismatch."""
    if not records[0].startswith(b'Version 2.20'):raise ValueError('Unsupported restart version')
    integer=lambda index:int(np.frombuffer(records[index],dtype='<i4')[0])
    nk=integer(9);ne=integer(10);nkon=integer(12)
    mi=np.frombuffer(records[13],dtype='<i4');mt=int(mi[1])+1
    if ne!=len(elements) or mt!=4:raise ValueError('Restart dimensions mismatch')
    widths=(8,11)
    matches=[]
    for width in widths:
        if nkon!=width*ne:continue
        for data in records:
            if len(data)!=4*nkon:continue
            kon=np.frombuffer(data,dtype='<i4').reshape(ne,width)
            if np.array_equal(kon[:,:8],elements+1):matches.append(kon)
    if len(matches)!=1:raise ValueError('Missing/ambiguous restart connectivity')
    kon=matches[0];modes=kon[:,8:]-1
    if modes.size and (np.any(modes<len(points)+6) or np.any(modes>=nk) or len(np.unique(modes))!=modes.size):
        raise ValueError('Invalid/shared internal-mode node IDs')
    coords=[]
    for data in records:
        if len(data)!=8*3*nk:continue
        co=np.frombuffer(data,dtype='<f8').reshape(nk,3)
        if np.array_equal(co[:len(points)],points):coords.append(co)
    if len(coords)!=1:raise ValueError('Missing/ambiguous restart coordinates')
    if modes.size:
        if not np.allclose(coords[0][modes],points[elements].mean(axis=1)[:,None,:],atol=1e-12,rtol=0):
            raise ValueError('Internal-mode center mapping mismatch')
    # In native restartwrite.f VOLD follows MATNAME and IELMAT. VEOLD may
    # have the same dimensions and become indistinguishable at zero-load
    # elastic states; values alone are not a safe record selector.
    selected=[]
    for i,data in enumerate(records[:-2]):
        if len(data)!=80 or data[:3]!=b'MAT' or data[3:].strip():continue
        if len(records[i+1])!=4*ne or not np.all(np.frombuffer(records[i+1],dtype='<i4')==1):continue
        if len(records[i+2])==8*mt*nk:selected.append(records[i+2])
    if len(selected)!=1:raise ValueError('Missing/ambiguous MATNAME/IELMAT/VOLD restart layout')
    field=np.frombuffer(selected[0],dtype='<f8').reshape(nk,mt)[:,1:4]
    wanted=np.vstack([dat_u,dat_macro])
    if not np.allclose(field[:len(wanted)],wanted,rtol=2e-6,atol=2e-8):
        raise ValueError('Restart VOLD disagrees with native DAT displacement')
    return field[:len(points)],field[len(points):len(points)+6],field[modes] if modes.size else np.zeros((ne,3,3))


def write_model(directory,c,points,elements,indices,estimate,allow_large=False):
    directory=Path(directory)
    kind=c.get('element_type','C3D8I')
    if kind not in ('C3D8','C3D8I'):raise ValueError('Experimental backend requires C3D8 or C3D8I')
    base={**c,'element_type':'C3D8','model_type':'finite_strain_elastoplastic_trial'}
    plastic.write_model(directory,base,points,elements,indices,estimate,allow_large)
    path=directory/'plastic.inp';text=path.read_text().split('*STEP,NLGEOM,INC=1000')[0]
    text=text.replace('TYPE=C3D8,',f'TYPE={kind},')
    m=json.loads((directory/'plastic-metadata.json').read_text());zz=m['macro_nodes'][2]
    anchor=text.split('*BOUNDARY\n')[1].splitlines()[0]
    fixed=[anchor]+[f'{node},2,3,0' for node in m['macro_nodes']]
    steps=c.get('compression_steps',20)
    if not isinstance(steps,int) or steps<1:raise ValueError('Positive integer compression_steps required')
    if c.get('control','displacement') not in ('force','displacement'):raise ValueError('Unknown control')
    for k in range(1,steps+2):
        lines=['*STEP,NLGEOM,INC=1000','*STATIC,SOLVER=SPOOLES','1,1,1e-6,1']
        if k<=steps:
            if c.get('control','displacement')=='force':
                lines+=['*CLOAD',f"{zz},1,{-c['target_mpa']*c['length_mm']**2/m['macro_scale']*k/steps}"]
            else:
                lines+=['*BOUNDARY',f"{zz},1,1,{-c['compression_fraction']*c['length_mm']*m['macro_scale']*k/steps}"]
        else:lines+=['*BOUNDARY,OP=NEW']+fixed+['*CLOAD',f'{zz},1,0']
        lines+=['*NODE PRINT,NSET=PHYSICAL,FREQUENCY=999999','U','*NODE PRINT,NSET=MACRO,FREQUENCY=999999','U',
                '*EL PRINT,ELSET=SOLID,FREQUENCY=999999','S,PEEQ,E',
                '*EL PRINT,ELSET=SOLID,FREQUENCY=999999,TOTALS=ONLY','ELSE,EVOL',
                '*RESTART,WRITE,FREQUENCY=1','*END STEP']
        text+='\n'.join(lines)+'\n'
    path.write_text(text)
    m=json.loads((directory/'plastic-metadata.json').read_text())
    m.update(config=c,element_type=kind,backend='experimental_periodic_internal_modes',
             first_mode_node=len(points)+7,mode_nodes_per_element=3 if kind=='C3D8I' else 0,
             mode_order_source='genmodes.f: consecutive triples, ascending element IDs',
             mode_gradient_source_commit='fd588701edd6a92fde06e99b1bdf01f3b0f80f2c',compression_steps=steps,
             state_source='native 2.20 unformatted restart, per-step endpoints')
    (directory/'mode-metadata.json').write_text(json.dumps(m,indent=2)+'\n')
    (directory/'plastic-metadata.json').rename(directory/'c3d8-template-metadata-not-executed.json')
    (directory/'sintered-metadata.json').rename(directory/'elastic-template-metadata-not-executed.json')


def _ip(s,prefix,ne):
    rows=find(s,prefix,'SOLID');rows=rows[np.lexsort((rows[:,1],rows[:,0]))]
    if len(rows)!=ne*8 or not np.array_equal(rows[:,0],np.repeat(np.arange(1,ne+1),8)) or not np.array_equal(rows[:,1],np.tile(np.arange(1,9),ne)):
        raise ValueError('Incomplete or unordered integration-point output')
    return rows[:,2:]


def analyze(directory,completed,code):
    directory=Path(directory);m=json.loads((directory/'mode-metadata.json').read_text());c=m['config'];L=c['length_mm']
    with np.load(directory/'sintered-geometry.npz') as g:p,e=g['points'],g['elements']
    ne=len(e);w0=sintered.hex_quadrature(p,e);gm=mode_gradients(p,e)
    states=iter(read_restart_groups(directory/'plastic.rout'))
    steps=m['compression_steps']
    previous_f=np.broadcast_to(np.eye(3),w0.shape+(3,3)).copy();previous_p=np.zeros_like(previous_f)
    previous_F=np.eye(3);previous_P=np.zeros((3,3));wmicro=wmacro=0.
    history=[];energy=[];last=None;max_E_error=0.;max_no_modes_error=0.;max_mode_u=0.;max_periodic=0.
    n=int(round(L/c['mesh_mm']));lookup=np.full((n+1,)*3,-1,dtype=int)
    lookup[tuple(np.rint(p/c['mesh_mm']).astype(int).T)]=np.arange(len(p))
    ep_previous=np.zeros((ne,8));plastic_monotone=True;force_error=0.;macro_virial=[]
    for time,s in read_dat(directory/'plastic.dat'):
        if abs(time-round(time))>1e-7:continue
        du=find(s,'displacements','PHYSICAL');du=du[np.argsort(du[:,0])][:,1:4]
        dm=find(s,'displacements','MACRO');dm=dm[np.argsort(dm[:,0])][:,1:4]
        group=next(states)
        extra=3*ne if m['element_type']=='C3D8I' else 0
        if int(np.frombuffer(group[9],dtype='<i4')[0])!=len(p)+6+extra or int(np.frombuffer(group[12],dtype='<i4')[0])!=8*ne+extra:
            raise ValueError('Native element topology disagrees with requested element type')
        u,macro_u,um=restart_state(group,p,e,du,dm)
        time=time/steps if time<=steps else 1+(time-steps)
        q=macro_u[:,0]/m['macro_scale']/L
        F=np.eye(3)+np.array([[q[0],q[3],q[4]],[q[3],q[1],q[5]],[q[4],q[5],q[2]]])
        fnodal=deformation_gradients(p,p+u,e)
        max_mode_u=max(max_mode_u,float(np.max(np.abs(um))))
        flocal=fnodal+np.einsum('eai,eqaj->eqij',um,gm)
        native_E=stress_tensors(_ip(s,'strains',ne).reshape(ne,8,6))
        green=.5*(flocal.swapaxes(-1,-2)@flocal-np.eye(3))
        green_corner=.5*(fnodal.swapaxes(-1,-2)@fnodal-np.eye(3))
        max_E_error=max(max_E_error,float(np.max(np.abs(green-native_E))))
        max_no_modes_error=max(max_no_modes_error,float(np.max(np.abs(green_corner-native_E))))
        jlocal=np.linalg.det(flocal);jmacro=float(np.linalg.det(F));vcell=jmacro*L**3
        vmaterial=float(np.sum(jlocal*w0));vgeometry=float(sintered.hex_quadrature(p+u,e).sum())
        sigma=stress_tensors(_ip(s,'stresses',ne).reshape(ne,8,6))
        piola=jlocal[...,None,None]*sigma@np.linalg.inv(flocal).swapaxes(-1,-2)
        P=(piola*w0[...,None,None]).sum(axis=(0,1))/L**3
        sigbar=(sigma*(jlocal*w0)[...,None,None]).sum(axis=(0,1))/vcell
        macro_virial.append([float(np.max(np.abs(P@F.T-jmacro*sigbar))),float(np.max(np.abs(P)))])
        wmicro+=float(np.sum(.5*(previous_p+piola)*(flocal-previous_f)*w0[...,None,None]))
        wmacro+=float(L**3*np.sum(.5*(previous_P+P)*(F-previous_F)))
        internal=float(find(s,'total internal energy','SOLID').ravel()[-1])
        evol=float(find(s,'total volume','SOLID').ravel()[-1])
        ep=_ip(s,'equivalent plastic strain',ne).reshape(ne,8)
        plastic_monotone &= bool(np.all(ep>=ep_previous-1e-7))
        for axis in range(3):
            a=np.take(lookup,0,axis=axis);b=np.take(lookup,n,axis=axis);valid=(a>=0)&(b>=0)
            max_periodic=max(max_periodic,float(np.max(np.abs(u[b[valid]]-u[a[valid]]-L*(F-np.eye(3))[:,axis]))))
        if c.get('control')=='force' and time<=1+1e-8:
            force_error=max(force_error,abs(P[2,2]+c['target_mpa']*time)/c['target_mpa'])
        history.append([time,-q[2],-P[2,2],-sigbar[2,2],1-vmaterial/vcell,1-vgeometry/vcell,
                        float(ep.max()),float(jlocal.min()),jmacro,vmaterial,vgeometry,evol])
        energy.append([time,wmacro,wmicro,internal]);last=(u,P,sigbar)
        previous_f=flocal;previous_p=piola;previous_F=F;previous_P=P;ep_previous=ep
    if not history:raise ValueError('No accepted mode frames')
    try:next(states);raise ValueError('Extra unmatched restart state')
    except StopIteration:pass
    h=np.array(history);en=np.array(energy);scale=max(float(np.max(np.abs(en[:,3]))),1e-14)
    comp=en[en[:,0]<=1+1e-8][-1];stress_scale=max(x[1] for x in macro_virial)
    virial=max(x[0] for x in macro_virial)/max(stress_scale,1e-14)
    phi_difference=float(np.max(np.abs(h[:,4]-h[:,5])))
    checks={'solver_completed':completed,'unload_complete':abs(h[-1,0]-2)<1e-6,
            'native_green_strain_match':max_E_error<1e-5,'positive_material_J':bool(np.min(h[:,7])>0),
            'positive_macro_J':bool(np.min(h[:,8])>0),'periodic_displacements':max_periodic<1e-5,
            'physical_geometry_porosity':bool(np.all((h[:,5]>-1e-7)&(h[:,5]<1))),
            'physical_material_porosity':bool(np.all((h[:,4]>-1e-7)&(h[:,4]<1))),
            'reference_EVOL_matches_mesh':bool(np.max(np.abs(h[:,11]-w0.sum()))<1e-6*w0.sum()),
            'volume_definition_difference':phi_difference<1e-4,
            'macro_stress_virial':virial<.01,
            'macro_micro_work':float(np.max(np.abs(en[:,1]-en[:,2])))/scale<.01,
            'compression_internal_work':abs(comp[2]-comp[3])/max(abs(comp[3]),1e-14)<.02,
            'cycle_internal_work':abs(en[-1,2]-en[-1,3])/scale<.02,
            'plastic_accumulation':plastic_monotone,'hardening_table_range':bool(np.max(h[:,6])<=5),'unloaded_macro_stress':float(np.max(np.abs(last[1])))<1e-5}
    if c.get('control')=='force':checks['prescribed_nominal_stress']=force_error<.01
    result={'status':'experimental_periodic_checks_passed_not_mesh_or_contact_validated' if all(checks.values()) else 'diagnostics_only',
            'solver_exit_code':code,'element_type':m['element_type'],'checks':{k:bool(v) for k,v in checks.items()},
            'max_native_green_strain_error':max_E_error,'max_omitting_modes_green_error':max_no_modes_error,
            'max_mode_displacement_mm':max_mode_u,'periodic_max_residual_mm':max_periodic,
            'stress_virial_relative_error':virial,'force_control_relative_error':force_error,
            'macro_micro_work_relative_error':float(np.max(np.abs(en[:,1]-en[:,2])))/scale,
            'compression_work_relative_error':float(abs(comp[2]-comp[3])/max(abs(comp[3]),1e-14)),
            'cycle_work_relative_error':float(abs(en[-1,2]-en[-1,3])/scale),
            'max_porosity_definition_difference':phi_difference,
            'peak_nominal_compression_MPa':float(h[:,2].max()),'peak_cauchy_compression_MPa':float(h[:,3].max()),
            'residual_compression_strain':float(h[-1,1]) if checks['unload_complete'] and checks['unloaded_macro_stress'] else None,
            'last_axial_compression_strain':float(h[-1,1]),'final_geometry_porosity':float(h[-1,5]),
            'final_material_porosity':float(h[-1,4]),'peak_peeq':float(h[:,6].max()),
            'solver_EVOL_definition':'reference volume (not deformed)',
            'limitations':['uncalibrated material','no self-contact or damage','mesh convergence unverified','not a 100 mm / 0.1 MPa calculation']}
    np.savetxt(directory/'mode-history.csv',h,delimiter=',',header='step_time,axial_compression_strain,nominal_compression_MPa,cauchy_compression_MPa,material_porosity,geometry_porosity,peak_peeq,min_material_J,macro_J,material_solid_mm3,geometry_solid_mm3,solver_reference_EVOL_mm3',comments='')
    np.savetxt(directory/'mode-energy.csv',en,delimiter=',',header='step_time,macro_work_Nmm,micro_work_Nmm,solver_internal_Nmm',comments='')
    (directory/'mode-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    import meshio
    meshio.write(directory/'mode-last-frame.vtu',meshio.Mesh(p+last[0],[('hexahedron',e)],point_data={'displacement_mm':last[0]}))
    return result


def run(directory,timeout=1200):
    directory=Path(directory).resolve();m=json.loads((directory/'mode-metadata.json').read_text())
    cap=int(m['config'].get('solver_memory_gib',24)*1024**2)
    command=['/bin/bash','-c','ulimit -v "$1"; exec "$2" -i plastic','mode-limit',str(cap),'/workspace/environment/calculix/usr/bin/ccx']
    with (directory/'mode.log').open('x') as f:
        try:code=subprocess.run(command,cwd=directory,env=solver_environment(),stdout=f,stderr=subprocess.STDOUT,timeout=timeout).returncode
        except subprocess.TimeoutExpired:code=124
    log=(directory/'mode.log').read_text(errors='replace')
    return analyze(directory,code==0 and 'Job finished' in log and '*ERROR' not in log,code)
