"""Gate convergence on COMPLETE porosity cycles, never a common failed prefix."""
import argparse,itertools,json
from pathlib import Path
import numpy as np


def assess(cases,phi_tolerance=.001,response_tolerance=.02):
    readiness={}
    for name,case in cases.items():
        s=case['summary'];attempts=s.get('attempts',[]);last=attempts[-1] if attempts else {};cycle=last.get('cycle') or {}
        readiness[name]=dict(target_is_ten_percent=s.get('target_loaded_porosity')==.1,
             no_pressure_cap=s.get('no_pressure_cap') is True,loaded_target=cycle.get('loaded_target') is True,
             cycle_complete=cycle.get('complete') is True,all_solver_checks=s.get('completed') is True)
    eligible=bool(cases) and len(cases)>1 and all(all(row.values()) for row in readiness.values())
    comparisons=[]
    if eligible:
        for (na,a),(nb,b) in itertools.combinations(cases.items(),2):
            ca=a['summary']['attempts'][-1]['cycle'];cb=b['summary']['attempts'][-1]['cycle']
            def curve(case,cycle):
                rows=[r for r in cycle['records'] if r['phase']=='hold' and r['passed']]
                if not rows or any(r.get('cauchy_stress_MPa') is None for r in rows):raise ValueError('Missing held Cauchy response')
                strain=np.r_[0.,[r['axial_compression'] for r in rows]]
                phi=np.r_[case['initial_porosity'],[r['porosity'] for r in rows]]
                pressure=np.r_[0.,[-r['cauchy_stress_MPa'][2][2] for r in rows]]
                if np.any(np.diff(strain)<=0):raise ValueError('Compression holds must be strictly monotone')
                return strain,phi,pressure
            xa,pa,sa=curve(a,ca);xb,pb,sb=curve(b,cb)
            grid=np.linspace(0.,min(xa[-1],xb[-1]),201)
            p_error=float(np.max(abs(np.interp(grid,xa,pa)-np.interp(grid,xb,pb))))
            stress_error=float(np.max(abs(np.interp(grid,xa,sa)-np.interp(grid,xb,sb))))/max(float(abs(sa).max()),float(abs(sb).max()),1e-12)
            strain_error=abs(ca['loaded_compression']-cb['loaded_compression'])/max(abs(ca['loaded_compression']),abs(cb['loaded_compression']),1e-12)
            recovery_a=[r for r in ca['records'] if r['phase']=='recovery'][-1]
            recovery_b=[r for r in cb['records'] if r['phase']=='recovery'][-1]
            recovery_error=abs(recovery_a['porosity']-recovery_b['porosity'])
            checks=dict(porosity_path=p_error<=phi_tolerance,cauchy_pressure_path=stress_error<=response_tolerance,
                        loaded_compression=strain_error<=response_tolerance,recovered_porosity=recovery_error<=phi_tolerance,
                        initial_porosity=abs(a['initial_porosity']-b['initial_porosity'])<=phi_tolerance)
            comparisons.append(dict(a=na,b=nb,porosity_absolute_difference=p_error,cauchy_pressure_relative_difference=stress_error,
                 loaded_compression_relative_difference=strain_error,recovered_porosity_absolute_difference=recovery_error,checks=checks,passed=all(checks.values())))
    return dict(eligible=eligible,readiness=readiness,comparisons=comparisons,
                verified=eligible and bool(comparisons) and all(r['passed'] for r in comparisons),
                phi_absolute_tolerance=phi_tolerance,response_relative_tolerance=response_tolerance,
                note='Tolerance choices are development criteria, not measured physical accuracy. Incomplete cycles are never compared for acceptance.')


def convergence_gate(cases):
    """Require actual single-axis refinements in addition to curve agreement.

    Three distinct resolutions per axis avoid accepting two identical runs or
    one apparent refinement pair as a convergence study. Configurations are
    compared in full, apart from output/restart bookkeeping.
    """
    result=assess(cases)
    axes=dict(space=('mesh_size_mm','neck_mesh_mm','curvature_points','rve_grid_n'),
              timestep=('cfl',),loading_rate=('ramp_s','unload_s'),
              contact_integration=('surface_quadrature_tolerance',))
    bookkeeping={'resume_checkpoint','stop_request_file','output_every','save_checkpoint','checkpoint_every',
                 'checkpoint_free_every','allow_contact_operator_update','inactive_quadrature_replay',
                 'embedded_quadrature_rules','native_library','usage_scope'}
    coverage={}
    for axis,keys in axes.items():
        groups={}
        for name,case in cases.items():
            config=case.get('config')
            if config is None:continue
            physical={k:v for k,v in config.items() if k not in bookkeeping and k not in keys}
            signature=json.dumps(physical,sort_keys=True)
            levels=json.dumps({k:config[k] for k in keys if k in config},sort_keys=True)
            groups.setdefault(signature,{}).setdefault(levels,[]).append(name)
        best=max(groups.values(),key=len,default={})
        coverage[axis]=dict(distinct_levels=len(best),required_levels=3,
                            case_groups=list(best.values()),passed=len(best)>=3)
    result['response_agreement']=result['verified']
    result['refinement_axes']=coverage
    result['verified']=result['verified'] and all(r['passed'] for r in coverage.values())
    result['note']+=' Physical convergence additionally requires three controlled levels on every listed numerical axis.'
    return result


def load_case(path):
    path=Path(path);s=json.loads((path/'summary.json').read_text());geometry=json.loads((path/'geometry.json').read_text())
    config_path=path/'attempt-000'/'config.json'
    return dict(summary=s,initial_porosity=geometry['geometry']['initial_porosity'],path=str(path),config=json.loads(config_path.read_text()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',action='append',required=True,help='name=run-directory')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cases={name:load_case(path) for name,path in (entry.split('=',1) for entry in a.case)}
    result=convergence_gate(cases);(a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    if not result['verified']:raise SystemExit(2)


if __name__=='__main__':main()
