"""Finite-deformation periodic elastoplastic trial, before self-contact validation.

No temperature, viscosity, soil-clay calibration, damage or contact is implied.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from . import sintered
from .cli import solver_environment
from .results import read_dat, find


def validate(c):
    sintered.validate(c)
    if c.get('element_type','C3D8')!='C3D8':
        raise ValueError('Periodic backend supports C3D8 only; C3D8I is a separate component benchmark')
    if c.get('model_type')!='finite_strain_elastoplastic_trial':raise ValueError('Explicit nonlinear trial model_type required')
    for key in ('yield_mpa','hardening_mpa','compression_fraction','initial_increment','max_increment'):
        if key not in c or not np.isfinite(c[key]) or c[key]<=0:raise ValueError(f'Positive {key} required')
    if not c['initial_increment']<=c['max_increment']<=1:raise ValueError('Invalid increment bounds')
    if not 0<c['compression_fraction']<.5:raise ValueError('Trial compression must be below 50%')
    if c.get('contact_model')!='not_implemented':raise ValueError('Self-contact is not implemented/validated')
    if c.get('material_calibration')!='uncalibrated_trial':raise ValueError('Only uncalibrated trial material supported')


def write_model(directory,c,points,elements,integers,estimate,allow_large=False):
    """Reuse exact periodic u(X+L)-u(X)=(F-I)L, with symmetric macro F.

    Coefficients do not depend on current F. The solid solver uses NLGEOM.
    Symmetric F fixes macroscopic rigid rotation, not local element rotations.
    """
    validate(c);directory=Path(directory)
    sintered.write_model(directory,c,points,elements,integers,estimate,allow_large)
    source=directory/'sintered.inp'
    lines=source.read_text().splitlines();lines=lines[:lines.index('*STEP')]
    lines[1]='Finite-deformation elastoplastic trial; mm N MPa; uncalibrated; no self-contact'
    pos=lines.index('*ELASTIC')+2
    table=['*PLASTIC,HARDENING=ISOTROPIC',f"{c['yield_mpa']},0",f"{c['yield_mpa']+5*c['hardening_mpa']},5"]
    lines[pos:pos]=table
    meta=json.loads((directory/'sintered-metadata.json').read_text())
    zz=meta['macro_nodes'][2];scale=meta['macro_scale'];anchor=lines[lines.index('*BOUNDARY')+1]
    fixed=[anchor]+[f'{node},2,3,0' for node in meta['macro_nodes']]
    outputs=['*NODE PRINT,NSET=PHYSICAL,FREQUENCY=1','U','*NODE PRINT,NSET=MACRO,FREQUENCY=1','U',
             '*EL PRINT,ELSET=SOLID,FREQUENCY=1','S,PEEQ',
             '*EL PRINT,ELSET=SOLID,FREQUENCY=1,TOTALS=ONLY','ELSE']
    for step in ('compress','release'):
        lines += ['*STEP,NLGEOM,INC=1000','*STATIC,SOLVER=SPOOLES',
                  f"{c['initial_increment']},1,1e-7,{c['max_increment']}"]
        if step=='compress':
            lines += ['*BOUNDARY',f"{zz},1,1,{-c['compression_fraction']*c['length_mm']*scale}"]
        else:
            lines += ['*BOUNDARY,OP=NEW']+fixed+['*CLOAD',f'{zz},1,0']
        lines += outputs+['*END STEP']
    with (directory/'plastic.inp').open('x') as stream:stream.write('\n'.join(lines)+'\n')
    # The helper-generated elastic input is explicitly retained as a template, never executed.
    source.rename(directory/'elastic-template-not-executed.inp')
    meta.update(model_type='finite_strain_elastoplastic_trial',
                contact_model='not_implemented',material_calibration='uncalibrated_trial',
                load_protocol='prescribed compression then release to zero generalized axial load',
                target_stress_is_not_enforced=True)
    (directory/'plastic-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')


def analyze(directory,completed,code):
    directory=Path(directory);m=json.loads((directory/'plastic-metadata.json').read_text());c=m['config']
    with np.load(directory/'sintered-geometry.npz') as g:p,e=g['points'],g['elements']
    history=[];last=None;initial=sintered.hex_quadrature(p,e).sum()
    for time,s in read_dat(directory/'plastic.dat'):
        u=find(s,'displacements','PHYSICAL');u=u[np.argsort(u[:,0])][:,1:4]
        macro=find(s,'displacements','MACRO');q=macro[np.argsort(macro[:,0])][:,1]/m['macro_scale']/c['length_mm']
        F=np.eye(3)+np.array([[q[0],q[3],q[4]],[q[3],q[1],q[5]],[q[4],q[5],q[2]]])
        w=sintered.hex_quadrature(p+u,e)
        stress=find(s,'stresses','SOLID');stress=stress[np.lexsort((stress[:,1],stress[:,0]))]
        plastic=find(s,'equivalent plastic strain','SOLID');plastic=plastic[np.lexsort((plastic[:,1],plastic[:,0]))]
        if len(u)!=len(p) or len(stress)!=8*len(e) or len(plastic)!=len(stress):raise ValueError('Incomplete output')
        expected=np.repeat(np.arange(1,len(e)+1),8)
        if not np.array_equal(stress[:,0],expected) or not np.array_equal(plastic[:,0],expected):
            raise ValueError('Missing/duplicate element output')
        if not np.array_equal(stress[:,1],np.tile(np.arange(1,9),len(e))) or not np.array_equal(stress[:,:2],plastic[:,:2]):
            raise ValueError('Missing/duplicate integration-point output')
        J=float(np.linalg.det(F));box=J*c['length_mm']**3
        mean=(stress[:,2:8]*w.ravel()[:,None]).sum(axis=0)/box
        phi=1-w.sum()/box
        residual=0.;n=int(round(c['length_mm']/c['mesh_mm']))
        lookup=np.full((n+1,)*3,-1,dtype=int);lookup[tuple(np.rint(p/c['mesh_mm']).astype(int).T)]=np.arange(len(p))
        for axis in range(3):
            low=np.take(lookup,0,axis=axis);high=np.take(lookup,n,axis=axis);valid=(low>=0)&(high>=0)
            residual=max(residual,float(np.max(np.abs(u[high[valid]]-u[low[valid]]-c['length_mm']*(F-np.eye(3))[:,axis]))))
        history.append([time,-q[2],-mean[2],phi,float(plastic[:,-1].max()),float(w.min()),J,residual])
        last=(u,w,stress,plastic,mean,F)
    columns='step_time,axial_compression_strain,compression_cauchy_MPa,porosity,peak_peeq,min_jacobian,cell_det_F,periodic_residual_mm'
    np.savetxt(directory/'plastic-history.csv',history,delimiter=',',header=columns,comments='')
    # DAT displacements/macro variables have ~7 significant digits; independent
    # rounding gives O(1e-8) apparent pores in an exactly solid affine cube.
    # Allow 1e-7 absolute porosity for this output precision, without clipping.
    checks={'solver_completed':completed,'end_of_unload_reached':bool(history and abs(history[-1][0]-2)<1e-6),
            'positive_jacobians':bool(history and min(row[5] for row in history)>0),
            'positive_cell':bool(history and min(row[6] for row in history)>0),
            'physical_porosity':bool(history and all(-1e-7<=row[3]<1 for row in history)),
            'periodic_displacements':bool(history and max(row[7] for row in history)<1e-5)}
    result={'status':'diagnostics_only','solver_exit_code':code,'checks':checks,
            'material_calibration':'uncalibrated_trial','self_contact':'not_implemented',
            'mesh_convergence':'unverified','porosity_output_tolerance':1e-7,'stress_target_mpa':c['target_mpa'],'target_stress_enforced':False,
            'initial_porosity':float(1-initial/c['length_mm']**3)}
    if history:
        result.update(peak_compression_mpa=float(max(row[2] for row in history)),
                      residual_compression_strain=history[-1][1],final_mean_cauchy_stress_mpa=last[4].tolist(),
                      final_porosity=history[-1][3],peak_equivalent_plastic_strain=float(max(row[4] for row in history)))
        checks.update(hardening_table_range=result['peak_equivalent_plastic_strain']<=5,
                      plastic_strain_generated=result['peak_equivalent_plastic_strain']>1e-5,
                      residual_deformation=result['residual_compression_strain']>1e-5,
                      unloaded_stress=bool(np.max(np.abs(last[4]))<1e-5))
        if all(checks.values()):result['status']='trial_material_cycle_passed_contact_and_calibration_unverified'
        import meshio
        meshio.write(directory/'plastic-last-frame.vtu',meshio.Mesh(p+last[0],[('hexahedron',e)],
                     point_data={'displacement_mm':last[0]},
                     cell_data={'peeq':[last[3][:,-1].reshape(-1,8).mean(axis=1)]}))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        h=np.array(history);fig,axes=plt.subplots(1,2,figsize=(10,4))
        axes[0].plot(h[:,1],h[:,2],'.-');axes[0].set(xlabel='Axial compression strain',ylabel='Mean Cauchy compression [MPa]')
        axes[1].plot(h[:,0],100*h[:,3],'.-');axes[1].set(xlabel='Step time (not physical time)',ylabel='Porosity [%]')
        fig.suptitle('Uncalibrated elastoplastic trial | self-contact not implemented')
        fig.tight_layout();fig.savefig(directory/'plastic-history.png',dpi=160);plt.close(fig)
    result['checks']={key:bool(value) for key,value in checks.items()}
    (directory/'plastic-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def run(directory,solver,timeout):
    directory=Path(directory).resolve();m=json.loads((directory/'plastic-metadata.json').read_text())
    executable=shutil.which(solver)
    if executable is None:raise ValueError('CalculiX unavailable')
    command=['/bin/bash','-c','ulimit -v "$1"; exec "$2" -i plastic','fem-limit',
             str(int(m['config'].get('solver_memory_gib',24)*1024**2)),executable]
    with (directory/'plastic.log').open('x') as stream:
        try:code=subprocess.run(command,cwd=directory,env=solver_environment(),stdout=stream,stderr=subprocess.STDOUT,timeout=timeout).returncode
        except subprocess.TimeoutExpired:code=124
    log=(directory/'plastic.log').read_text(errors='replace')
    result=analyze(directory,code==0 and 'Job finished' in log and '*ERROR' not in log,code)
    print(json.dumps(result,indent=2));return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--timeout',type=float,default=600);parser.add_argument('--allow-large',action='store_true')
    a=parser.parse_args()
    if a.timeout<=0:parser.error('Positive timeout required')
    c=json.loads(a.config.read_text());validate(c)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    p,e,idx,estimate=sintered.prepare(a.output,c,a.allow_large)
    write_model(a.output,c,p,e,idx,estimate,a.allow_large)
    result=run(a.output,'/workspace/environment/calculix/usr/bin/ccx',a.timeout)
    return 0 if result['status'].startswith('trial_material_cycle_passed') else 2

if __name__=='__main__':raise SystemExit(main())
