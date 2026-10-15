"""Exclusive output folders; solver failure is a nonzero CLI outcome."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from .geometry import make_packing, sphere_mesh
from .model import validate_config, plan, write_model, dynamic_steps
from .results import analyze


def solver_environment():
    env=os.environ.copy()
    env.setdefault('OMP_NUM_THREADS','2')
    env.setdefault('MPLBACKEND','Agg')
    env.setdefault('MPLCONFIGDIR','/workspace/environment/mix_charging/mplconfig')
    env.setdefault('XDG_CACHE_HOME','/workspace/environment/mix_charging/cache')
    lib=Path('/workspace/environment/calculix/usr/lib/x86_64-linux-gnu')
    if lib.is_dir():
        env['LD_LIBRARY_PATH']=str(lib)+(os.pathsep+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    for name in ('MPLCONFIGDIR','XDG_CACHE_HOME'):
        Path(env[name]).mkdir(parents=True,exist_ok=True)
    return env


def execute(directory, solver, timeout=600, cycles=3):
    directory=Path(directory).resolve()
    meta=json.loads((directory/'metadata.json').read_text())
    c=meta['config']
    executable=shutil.which(solver)
    if executable is None:
        raise ValueError(f'CalculiX executable unavailable: {solver}')
    contents=(directory/'model.inp').read_text()
    force=c['target_mpa']*c['length_mm']**2
    previous_force=None
    expected_time=0.
    for cycle in range(cycles):
        job=f'cycle-{cycle:02d}'
        path=directory/f'{job}.inp'
        # Append only to the in-memory history; retain every executed job.
        contents += dynamic_steps(c,force,str(cycle),first=cycle==0,previous_force=previous_force).replace(
            '__MACRO_Z__',str(meta['macro_nodes'][2]))
        expected_time += c['ramp_s']+c['hold_s']
        with path.open('x') as stream:
            stream.write(contents)
        with (directory/f'{job}.log').open('x') as stream:
            try:
                run=subprocess.run([executable,'-i',job],cwd=directory,env=solver_environment(),
                                   stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
            except subprocess.TimeoutExpired:
                returncode=124
            else:
                returncode=run.returncode
        log=(directory/f'{job}.log').read_text(errors='replace')
        completed=returncode==0 and 'Job finished' in log and '*ERROR' not in log
        result=analyze(directory,job,completed,expected_time,returncode)
        print(json.dumps(result,indent=2),flush=True)
        if result['status']=='equilibrium_candidate_requires_refinement':
            return result
        if not completed or result['error'] or not result['last_accepted_increment']:
            return result
        last=result['last_accepted_increment']
        # Correct nominal load to the current lateral area. Repeat from the
        # same initial state with the complete prior loading history preserved.
        previous_force=force
        force=c['target_mpa']*last['lx_mm']*last['ly_mm']
        # Geometry, invalid element, missing contacts and unresolved facets
        # require another discretization, not more unchanged solver retries.
        blockers=('positive_elements','physical_porosity','complete_contact_candidates',
                  'adjacent_images_sufficient','sphere_geometry_resolved','contact_penetration_checked',
                  'controller_energy_checked','history_geometry_and_contacts_valid',
                  'small_penetration','small_penalty_energy','coulomb_bound')
        if any(not result['checks'].get(k,False) for k in blockers):
            return result
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description='Periodic sphere packing compression: genuine solid FEM, no cylinder')
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ('estimate','prepare','run'):
        p=sub.add_parser(command)
        p.add_argument('--config',type=Path,default=Path('configs/baseline.json'))
        p.add_argument('--output',type=Path,required=True,help='Must not already exist')
        p.add_argument('--allow-large',action='store_true')
        p.add_argument('--centers',type=Path,help='Existing N x 3 periodic centers .npy (geometry only)')
        if command=='run':
            p.add_argument('--solver',default='/workspace/environment/calculix/usr/bin/ccx')
            p.add_argument('--timeout',type=float,default=600.)
            p.add_argument('--cycles',type=int,default=3)
    p=sub.add_parser('solve')
    p.add_argument('directory',type=Path)
    p.add_argument('--solver',default='/workspace/environment/calculix/usr/bin/ccx')
    p.add_argument('--timeout',type=float,default=600.)
    p.add_argument('--cycles',type=int,default=3)
    args=parser.parse_args(argv)
    try:
        if args.command=='solve':
            if (args.directory/'mode-metadata.json').exists():
                raise ValueError('Internal-mode experiment uses packing_fem.incompatible; legacy solver refused')
            if (args.directory/'plastic-metadata.json').exists():
                raise ValueError('Nonlinear trial uses python -m packing_fem.plastic; do not send it to the linear solver')
            if args.timeout<=0 or args.cycles<1:
                raise ValueError('Positive timeout and cycle count required')
            if (args.directory/'sintered-metadata.json').exists():
                from .sintered import solve
                result=solve(args.directory,args.solver,args.timeout)
            else:
                result=execute(args.directory,args.solver,args.timeout,args.cycles)
            return 0 if result['status']=='equilibrium_candidate_requires_refinement' else 2
        c=json.loads(args.config.read_text())
        if c.get('model_type')=='finite_strain_elastoplastic_trial':
            raise ValueError('Use python -m packing_fem.plastic for the nonlinear trial; linear backend is refused')
        if c.get('model_type')=='sintered':
            from . import sintered
            sintered.validate(c)
            args.output.mkdir(parents=True,exist_ok=False)
            (args.output/'config.json').write_text(json.dumps(c,indent=2)+'\n')
            if args.centers:raise ValueError('Sintered geometry adjusts sphere count including neck volume; --centers unsupported')
            points,elements,integers,estimate=sintered.prepare(args.output,c,args.allow_large)
            if args.command=='estimate':return 0
            sintered.write_model(args.output,c,points,elements,integers,estimate,args.allow_large)
            if args.command=='run':
                result=sintered.solve(args.output,args.solver,args.timeout)
                return 0 if result['status']=='equilibrium_candidate_requires_refinement' else 2
            return 0
        validate_config(c)
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'config.json').write_text(json.dumps(c,indent=2)+'\n')
        if args.centers:
            centers=np.load(args.centers,allow_pickle=False)
            expected=int(np.floor((1-c['porosity'])*c['length_mm']**3/(np.pi*c['diameter_mm']**3/6)+.5))
            if centers.shape!=(expected,3) or not np.all(np.isfinite(centers)):
                raise ValueError('Center count/shape must match the configured porosity')
            centers=centers % c['length_mm']
            from scipy.spatial import cKDTree
            if cKDTree(centers,boxsize=c['length_mm']).query_pairs(c['diameter_mm']*(1-1e-7)):
                raise ValueError('Supplied centers overlap across periodic boundaries')
        else:
            centers=make_packing(c['length_mm'],c['diameter_mm'],c['porosity'],c['seed'])
        np.save(args.output/'centers.npy',centers)
        np.savetxt(args.output/'centers-mm.csv',centers,delimiter=',',header='x_mm,y_mm,z_mm',comments='')
        points,tets,surface=sphere_mesh(c['diameter_mm'],c['mesh_mm'],c.get('mesh_order',2))
        _,_,estimate=plan(c,centers,points,tets)
        (args.output/'estimate.json').write_text(json.dumps(estimate,indent=2)+'\n')
        print(json.dumps(estimate,indent=2),flush=True)
        if args.command=='estimate':
            return 0
        write_model(args.output,c,centers,points,tets,surface,args.allow_large)
        if args.command=='run':
            if args.timeout<=0 or args.cycles<1:
                raise ValueError('Positive timeout and cycle count required')
            result=execute(args.output,args.solver,args.timeout,args.cycles)
            return 0 if result['status']=='equilibrium_candidate_requires_refinement' else 2
        return 0
    except (ValueError,OSError,RuntimeError) as exc:
        print(str(exc),file=sys.stderr)
        return 2


if __name__=='__main__':
    sys.exit(main())
