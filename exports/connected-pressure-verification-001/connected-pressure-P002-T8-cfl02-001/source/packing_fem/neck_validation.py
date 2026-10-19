"""Fixed-geometry neck refinement and C3D8/C3D8I component benchmarks.

Quarter of two truncated 10 mm spheres at z=6,18 mm with a 2 mm neck.
No periodic packing, self-contact, temperature, damage, or material calibration.
Boundary RF is used only on physical nodes without MPCs. Incompatible-mode
internal gradients are NOT reconstructed using eight corner nodes.
"""
import json
from pathlib import Path
import subprocess
import numpy as np
from .cli import solver_environment
from .results import read_dat,find
from .sintered import hex_quadrature

CORNERS=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                  [0,0,1],[1,0,1],[1,1,1],[0,1,1]])


def mesh_from_grid(axes,mask):
    cells=np.argwhere(mask)
    vertices=(cells[:,None,:]+CORNERS).reshape(-1,3)
    index,inverse=np.unique(vertices,axis=0,return_inverse=True)
    points=np.column_stack([axes[k][index[:,k]] for k in range(3)])
    return points,inverse.reshape(-1,8)


def neck_mesh(refinement):
    if refinement not in (1,2,4):raise ValueError('Supported refinement factors are 1,2,4')
    base=[np.arange(0,5.01,.5),np.arange(0,5.01,.5),np.arange(3,21.01,.5)]
    centers=[.5*(v[:-1]+v[1:]) for v in base]
    x,y,z=np.meshgrid(*centers,indexing='ij')
    sphere=(x*x+y*y+(z-6)**2<=25)|(x*x+y*y+(z-18)**2<=25)
    mask0=sphere|((x*x+y*y<=1)&(z>=6)&(z<=18))
    axes=[]
    for dim,b in enumerate(base):
        values=[]
        for lo,hi in zip(b[:-1],b[1:]):
            refined=hi<=1.5 if dim<2 else lo>=10 and hi<=14
            subdivisions=refinement if refined else 1
            values.extend(np.linspace(lo,hi,subdivisions+1)[:-1])
        axes.append(np.array(values+[b[-1]]))
    sample=[.5*(v[:-1]+v[1:]) for v in axes]
    indices=[np.searchsorted(b,s,side='right')-1 for b,s in zip(base,sample)]
    mask=mask0[np.ix_(*indices)]
    points,elements=mesh_from_grid(axes,mask)
    volume=float(hex_quadrature(points,elements).sum())
    # Exact cut area of the original stepped neck, not pi*r^2.
    mid=np.argmin(np.abs(centers[2]-12))
    area=float(mask0[:,:,mid].sum()*.5**2)
    return points,elements,{'geometry':'fixed 0.5 mm stepped union; quarter specimen',
        'grain_diameter_mm':10.,'neck_diameter_mm':2.,'sphere_centers_z_mm':[6.,18.],
        'grip_z_mm':[3.,21.],'refinement_factor':refinement,'neck_element_edge_mm':.5/refinement,
        'initial_solid_volume_mm3':volume,'quarter_neck_area_mm2':area,
        'elements':len(elements),'physical_nodes':len(points),'geometry_rerasterized':False,
        'refinement_note':'Conforming coordinate planes extend through shoulders; no hanging nodes.'}


def top_surface_weights(points,elements):
    """Reference-area weights for a nonuniform rectangular-face grid."""
    faces=elements[np.all(np.isclose(points[elements[:,4:8],2],21),axis=1)][:,4:8]
    v=points[faces]
    area=np.linalg.norm(np.cross(v[:,1]-v[:,0],v[:,3]-v[:,0]),axis=1)
    weights=np.zeros(len(points))
    np.add.at(weights,faces.ravel(),np.repeat(area/4,4))
    if weights.sum()<=0:raise ValueError('No top surface at z=21 mm')
    return weights/weights.sum()


def _nodes(lines,name,ids):
    lines.append(f'*NSET,NSET={name}')
    for start in range(0,len(ids),16):lines.append(','.join(map(str,ids[start:start+16])))


def write_neck(directory,c,refinement,element_type):
    if element_type not in ('C3D8','C3D8I'):raise ValueError('Unsupported component element')
    if c['diameter_mm']!=10 or c['neck_diameter_mm']!=2:
        raise ValueError('Fixed component geometry requires diameter 10 mm and neck 2 mm')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    p,e,geometry=neck_mesh(refinement)
    np.savez_compressed(directory/'geometry.npz',points=p,elements=e)
    lines=['*HEADING','Fixed stepped neck; quarter symmetry; mm N MPa; uncalibrated','*NODE']
    lines += [f'{i},'+','.join(map(str,v)) for i,v in enumerate(p,1)]
    lines += [f'*ELEMENT,TYPE={element_type},ELSET=SOLID']
    lines += [f'{i},'+','.join(map(str,v+1)) for i,v in enumerate(e,1)]
    lines += ['*SOLID SECTION,ELSET=SOLID,MATERIAL=MAT','*MATERIAL,NAME=MAT','*ELASTIC',
              f"{c['young_mpa']},{c['poisson']}",'*PLASTIC,HARDENING=ISOTROPIC',
              f"{c['yield_mpa']},0",f"{c['yield_mpa']+5*c['hardening_mpa']},5",
              '*NSET,NSET=PHYSICAL,GENERATE',f'1,{len(p)},1']
    for name,dim,value in [('XSYM',0,0),('YSYM',1,0),('BOTTOM',2,3),('TOP',2,21)]:
        _nodes(lines,name,(np.flatnonzero(np.isclose(p[:,dim],value))+1).tolist())
    fixed=['XSYM,1,1,0','YSYM,2,2,0','BOTTOM,3,3,0']
    lines+=['*BOUNDARY']+fixed
    for stage in ('compress','release'):
        lines+=['*STEP,NLGEOM,INC=1000','*STATIC,SOLVER=SPOOLES',
                f"{c['initial_increment']},1,1e-7,{c['max_increment']}"]
        if stage=='compress':lines+=['*BOUNDARY',f"TOP,3,3,{-18*c['compression_fraction']}"]
        else:lines+=['*BOUNDARY,OP=NEW']+fixed+['*CLOAD','TOP,3,0']
        lines+=['*NODE PRINT,NSET=PHYSICAL,FREQUENCY=1','U','*NODE PRINT,NSET=TOP,FREQUENCY=1','RF',
                '*EL PRINT,ELSET=SOLID,FREQUENCY=1','PEEQ',
                '*EL PRINT,ELSET=SOLID,FREQUENCY=1,TOTALS=ONLY','ELSE,EVOL','*END STEP']
    (directory/'neck.inp').write_text('\n'.join(lines)+'\n')
    metadata={'config':c,'geometry':geometry,'element_type':element_type,
              'unused_reference_config_fields':['length_mm','porosity','target_mpa','seed','mesh_mm','neck_neighbors'],
              'top_ids':(np.flatnonzero(p[:,2]==21)+1).tolist(),
              'limitations':['isolated quarter component, not periodic RVE','uncalibrated J2 plasticity',
                             'no self-contact or damage','internal incompatible-mode strain not reconstructed']}
    (directory/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


def solve_neck(directory,timeout=900):
    directory=Path(directory).resolve();m=json.loads((directory/'metadata.json').read_text())
    c=m['config'];code=124
    command=['/bin/bash','-c','ulimit -v "$1"; exec "$2" -i neck','neck-limit',
             str(int(c.get('solver_memory_gib',24)*1024**2)),'/workspace/environment/calculix/usr/bin/ccx']
    with (directory/'neck.log').open('x') as stream:
        try:code=subprocess.run(command,cwd=directory,env=solver_environment(),stdout=stream,stderr=subprocess.STDOUT,timeout=timeout).returncode
        except subprocess.TimeoutExpired:pass
    log=(directory/'neck.log').read_text(errors='replace')
    completed=code==0 and 'Job finished' in log and '*ERROR' not in log
    rows=[];last=None;work=0.;previous_u=np.zeros((len(m['top_ids']),3));previous_force=previous_u.copy()
    minimum_corner_jacobian=float('inf')
    with np.load(directory/'geometry.npz') as g:p,e=g['points'],g['elements']
    top_weights=top_surface_weights(p,e)
    for t,s in read_dat(directory/'neck.dat'):
        u=find(s,'displacements','PHYSICAL');u=u[np.argsort(u[:,0])][:,1:4]
        f=find(s,'forces','TOP');f=f[np.argsort(f[:,0])]
        if not np.array_equal(f[:,0],m['top_ids']) or len(u)!=len(p):raise ValueError('Incomplete physical output')
        top_u=u[np.array(m['top_ids'])-1];force=f[:,1:4]
        work+=float(np.sum(.5*(force+previous_force)*(top_u-previous_u)))
        peeq=find(s,'equivalent plastic strain','SOLID')[:,-1]
        internal=float(find(s,'total internal energy','SOLID').ravel()[-1])
        volume=float(find(s,'total volume','SOLID').ravel()[-1])
        minimum_corner_jacobian=min(minimum_corner_jacobian,float(hex_quadrature(p+u,e).min()))
        rows.append([t,float(-np.dot(top_weights,u[:,2])),float(-force[:,2].sum()),float(peeq.max()),internal,work,volume])
        previous_u=top_u;previous_force=force;last=u
    if not rows:raise ValueError('No accepted frames')
    a=np.array(rows);scale=max(float(np.max(np.abs(a[:,4]))),1e-14)
    compress=a[:,0]<=1+1e-8;load=a[compress][-1]
    checks={'solver_completed':completed,'unload_completed':abs(a[-1,0]-2)<1e-6,
            'positive_corner_geometry_jacobian':minimum_corner_jacobian>0,
            'positive_solver_volume':bool(np.all(a[:,6]>0)),
            'plastic_strain_generated':float(a[:,3].max())>1e-5,
            'residual_displacement':float(a[-1,1])>1e-5,
            'zero_final_top_external_force':abs(float(a[-1,2]))<1e-7,
            'compression_work_balance':abs(float(load[4]-load[5]))/max(abs(float(load[4])),1e-14)<.02,
            'cycle_work_balance':abs(float(a[-1,4]-a[-1,5]))/scale<.02}
    result={'status':'component_cycle_passed_not_RVE_validated' if all(checks.values()) else 'diagnostics_only',
            'solver_exit_code':code,'checks':{k:bool(v) for k,v in checks.items()},'element_type':m['element_type'],
            'geometry':m['geometry'],'peak_quarter_force_N':float(a[:,2].max()),
            'peak_neck_nominal_stress_MPa':float(a[:,2].max()/m['geometry']['quarter_neck_area_mm2']),
            'residual_grip_compression_mm':float(a[-1,1]),'grip_mean_definition':'reference-area-weighted top surface','peak_peeq':float(a[:,3].max()),
            'compression_work_relative_error':float(abs(load[4]-load[5])/max(abs(load[4]),1e-14)),
            'cycle_work_relative_error':float(abs(a[-1,4]-a[-1,5])/scale),
            'incompatible_mode_local_work_audit':'not_performed',
            'note':'RF-based boundary work vs solver ELSE; no fictitious eight-node local strain audit for C3D8I.'}
    np.savetxt(directory/'history.csv',a,delimiter=',',header='step_time,grip_compression_mm,quarter_force_N,peak_peeq,solver_internal_Nmm,boundary_work_Nmm,solver_volume_mm3',comments='')
    (directory/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    if last is not None:
        import meshio
        meshio.write(directory/'last-frame.vtu',meshio.Mesh(p+last,[('hexahedron',e)],point_data={'displacement_mm':last}))
    return result


def beam_benchmark(directory,element_type):
    """Small-displacement cantilever: same 20 cubes, compare with beam theory."""
    if element_type not in ('C3D8','C3D8I'):raise ValueError('Unsupported beam element')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    p,e=mesh_from_grid([np.arange(21.),np.array([0.,1.]),np.array([0.,1.])],np.ones((20,1,1),bool))
    left=(np.flatnonzero(p[:,0]==0)+1).tolist();tip=(np.flatnonzero(p[:,0]==20)+1).tolist()
    load=1e-5
    lines=['*HEADING','Independent slender cantilever benchmark; mm N MPa','*NODE']
    lines += [f'{i},'+','.join(map(str,v)) for i,v in enumerate(p,1)]
    lines += [f'*ELEMENT,TYPE={element_type},ELSET=SOLID']
    lines += [f'{i},'+','.join(map(str,v+1)) for i,v in enumerate(e,1)]
    lines += ['*SOLID SECTION,ELSET=SOLID,MATERIAL=MAT','*MATERIAL,NAME=MAT','*ELASTIC','10,.33']
    _nodes(lines,'FIXED',left);_nodes(lines,'TIP',tip)
    lines += ['*BOUNDARY','FIXED,1,3,0','*STEP','*STATIC,SOLVER=SPOOLES','*CLOAD']
    lines += [f'{node},2,{-load/len(tip)}' for node in tip]
    lines += ['*NODE PRINT,NSET=TIP','U','*END STEP']
    (directory/'beam.inp').write_text('\n'.join(lines)+'\n')
    with (directory/'beam.log').open('x') as f:
        run=subprocess.run(['/workspace/environment/calculix/usr/bin/ccx','-i','beam'],cwd=directory.resolve(),
                           env=solver_environment(),stdout=f,stderr=subprocess.STDOUT,timeout=30)
    log=(directory/'beam.log').read_text(errors='replace')
    if run.returncode!=0 or 'Job finished' not in log:raise ValueError('Beam solver failed')
    _,s=list(read_dat(directory/'beam.dat'))[-1]
    tip_disp=float(-find(s,'displacements','TIP')[:,2].mean())
    euler=load*20**3/(3*10/12)
    shear=load*20/((5/6)*(10/(2*(1+.33))))
    expected=euler+shear
    result={'element_type':element_type,'solver_completed':True,'tip_displacement_mm':tip_disp,
            'euler_bernoulli_mm':euler,'timoshenko_reference_mm':expected,
            'relative_error_to_timoshenko':abs(tip_disp-expected)/expected,
            'within_5_percent_beam_reference':abs(tip_disp-expected)/expected<.05,
            'note':'Independent elastic slender-beam benchmark, not a yield or RVE calibration.'}
    (directory/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
