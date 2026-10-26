"""Assess stages 1-4 without treating stopped prefixes as pressure solutions."""
import argparse,csv,json
from pathlib import Path
import numpy as np


CASES=('rapid-run','slow-run','half-run','resolved-rapid-run','resolved-slow-run','resolved-half-run','resolution-half-run')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    plan=json.loads((a.input/'acceptance-plan.json').read_text());cases={};histories={}
    for name in CASES:
        directory=a.input/name/'calculation';s=json.loads((directory/'summary.json').read_text());c=json.loads((directory/'config.json').read_text())
        cases[name]=dict(summary=s,config=c)
        histories[name]=np.loadtxt(directory/'accepted-step-audit.csv.gz',delimiter=',',skiprows=1)
    comparisons={};rows=[]
    for kind,first,second in (('rate','resolved-rapid-run','resolved-slow-run'),('timestep','resolved-slow-run','resolved-half-run'),('edge_resolution','resolved-rapid-run','resolution-half-run')):
        x=histories[first];y=histories[second];maximum=min(x[:,3].max(),y[:,3].max());pressures=np.linspace(maximum*.05,maximum,101)
        compression0=np.interp(pressures,x[:,3],x[:,1]);compression1=np.interp(pressures,y[:,3],y[:,1]);porosity0=np.interp(pressures,x[:,3],x[:,2]);porosity1=np.interp(pressures,y[:,3],y[:,2])
        comparisons[kind]=dict(scope='Common stopped loading prefix only; not final equilibrium or a convergence acceptance',maximum_common_pressure_MPa=float(maximum),porosity_max_absolute_difference=float(np.max(abs(porosity0-porosity1))),compression_max_difference_over_peak=float(np.max(abs(compression0-compression1))/max(compression0.max(),compression1.max(),1e-30)))
        for values in zip(pressures,compression0,compression1,porosity0,porosity1):rows.append(dict(comparison=kind,pressure_MPa=float(values[0]),compression_first=float(values[1]),compression_second=float(values[2]),porosity_first=float(values[3]),porosity_second=float(values[4])))
    positive=json.loads((a.input/'crossed-control/summary.json').read_text())
    physical_names=('resolved-rapid-run','resolved-slow-run','resolved-half-run','resolution-half-run')
    complete={name:bool(all(cases[name]['summary']['checks'].values())) for name in physical_names}
    full_hold=all(complete.values())
    whole_kinetic={name:cases[name]['summary']['whole_path_audit']['loaded']['peak_kinetic_internal_contact_ratio'] for name in physical_names}
    kinetic_ok=all(value<=plan['whole_loaded_kinetic_internal_contact_ratio_limit'] for value in whole_kinetic.values())
    differences_ok=(comparisons['rate']['porosity_max_absolute_difference']<=plan['rate_final_porosity_absolute_tolerance'] and comparisons['rate']['compression_max_difference_over_peak']<=plan['rate_final_compression_relative_tolerance'] and comparisons['timestep']['porosity_max_absolute_difference']<=plan['timestep_final_porosity_absolute_tolerance'] and comparisons['timestep']['compression_max_difference_over_peak']<=plan['timestep_final_compression_relative_tolerance'])
    result=dict(scope=__doc__,stages=dict(stage1_pair_guard_implemented=True,stage2_manufactured_retry_passed=positive['passed'],stage2_general_new_history_transfer_complete=False,stage3_pressure_and_hold_passed=full_hold,stage4_quasistatic_verified=full_hold and kinetic_ok and differences_ok),complete_pressure_hold_checks=complete,whole_loaded_kinetic_ratios=whole_kinetic,common_prefix_comparisons=comparisons,cases=cases,physical_compaction_verified=False,
                blocker='New active surface quadrature point lacks old endpoint friction histories; the rejected proposal also changes normal contact quadrature energy.',acceptance_plan=plan)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (a.output/'common-prefix-comparison.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fields=['case','ramp_s','hold_s','cfl','edge_resolution','steps','pressure_MPa','porosity','compression','failure','path_completed','all_solver_checks']
    with (a.output/'case-outcomes.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for name,item in cases.items():
            s=item['summary'];c=item['config'];writer.writerow(dict(case=name,ramp_s=c['ramp_s'],hold_s=c['hold_s'],cfl=c['cfl'],edge_resolution=c.get('edge_quadrature_relative_tolerance',0.),steps=s['steps'],pressure_MPa=s['peak_command_pressure_MPa'],porosity=s['final_porosity'],compression=s['final_axial_compression'],failure=s['failure'],path_completed=s['checks']['path_completed'],all_solver_checks=all(s['checks'].values())))
    print(json.dumps(dict(stages=result['stages'],comparisons=comparisons),indent=2))


if __name__=='__main__':main()
