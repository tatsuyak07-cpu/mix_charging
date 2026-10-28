"""Publish compact, explicitly limited verification evidence from archived runs."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    names=['baseline','half_increment','quarter_increment','yield_half','hardening_double','mesh_075mm']
    records={};histories={};energies={};rows=[]
    for name in names:
        run=a.source/name
        if not (run/'plastic-summary.json').exists():
            records[name]={'error':'Missing solver/postprocessing summary'};continue
        config=json.loads((run/'config.json').read_text())
        result=json.loads((run/'plastic-summary.json').read_text())
        estimate=json.loads((run/'estimate.json').read_text())
        record={'config':config,'result':result,'estimate':estimate}
        target=a.output/name;target.mkdir()
        for path in run.iterdir():
            if path.suffix in ('.json','.csv','.png','.sta','.npz','.vtu') or path.name in ('plastic.inp','plastic.log'):
                shutil.copyfile(path,target/path.name)
        if (run/'energy-audit/energy-history.csv').exists():
            table=np.genfromtxt(run/'energy-audit/energy-history.csv',delimiter=',',names=True)
            energy=json.loads((run/'energy-audit/energy-summary.json').read_text())
            # A purely elastic cycle ends at zero energy: normalize by cycle peak,
            # not by the (potentially vanishing) final internal energy.
            reference=max(float(np.max(np.abs(table['solver_internal_energy_Nmm']))),1e-14)
            difference=float(abs(table['micro_work_Nmm'][-1]-table['solver_internal_energy_Nmm'][-1])/reference)
            energy['cycle_work_internal_relative_difference']=difference
            energy['work_reference_peak_internal_energy_Nmm']=reference
            energy['checks']['cycle_work_balance']=difference<.02
            energy['cycle_normalization']='peak absolute solver internal energy over cycle'
            (target/'energy-summary-peak-normalized.json').write_text(json.dumps(energy,indent=2)+'\n')
            shutil.copyfile(run/'energy-audit/energy-summary.json',target/'energy-summary-original.json')
            shutil.copyfile(run/'energy-audit/energy-history.csv',target/'energy-history.csv')
            record['energy']=energy;energies[name]=table
        records[name]=record
        histories[name]=np.genfromtxt(run/'plastic-history.csv',delimiter=',',names=True)
        rows.append([name,config['mesh_mm'],config['max_increment'],config['yield_mpa'],config['hardening_mpa'],
                     estimate['solid_elements'],result['peak_compression_mpa'],result['residual_compression_strain'],
                     result['initial_porosity'],result['final_porosity'],result['peak_equivalent_plastic_strain'],
                     all(result['checks'].values()),all(record.get('energy',{'checks':{'missing':False}})['checks'].values())])
    with (a.output/'comparison.csv').open('w') as f:
        writer=csv.writer(f,lineterminator="\n");writer.writerow(['case','mesh_mm','max_increment','yield_MPa','hardening_MPa','elements',
            'peak_compression_MPa','residual_compression_strain','initial_porosity','final_porosity','peak_peeq',
            'cycle_checks_passed','energy_checks_passed']);writer.writerows(rows)
    def changes(coarse,fine):
        x=records[coarse]['result'];y=records[fine]['result']
        stress=abs(x['peak_compression_mpa']-y['peak_compression_mpa'])/abs(y['peak_compression_mpa'])
        strain=abs(x['residual_compression_strain']-y['residual_compression_strain'])/abs(y['residual_compression_strain'])
        phi=abs(x['final_porosity']-y['final_porosity'])
        return {'coarse':coarse,'fine':fine,'peak_stress_relative_change':stress,
                'residual_strain_relative_change':strain,'final_porosity_absolute_change':phi,
                'exploratory_response_criteria_passed':stress<.02 and strain<.02 and phi<1e-4}
    response={}
    for x,y in [('baseline','half_increment'),('half_increment','quarter_increment'),('baseline','mesh_075mm')]:
        if all('result' in records.get(n,{}) for n in [x,y]):response[x+'_to_'+y]=changes(x,y)
    summary={'cases':records,'response_comparisons':response,
             'criteria':{'response_relative_change':.02,'porosity_absolute_change':1e-4,
                         'macro_micro_work_relative':.01,'work_internal_relative':.02},
             'mesh_convergence':'not_established_by_two_levels',
             'material_calibration':'not_performed_no_experimental_data',
             'self_contact':'not_implemented_or_verified','full_100mm_target_0_1_mpa':'not_run',
             'limitations':['30 mm fixed-center trial without contact; absence of intersections not verified','hardening integral is not a separated thermodynamic dissipation',
                            'boundary release can undersample unloading','response thresholds are developer-defined exploratory checks']}
    (a.output/'verification-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for name in ['baseline','half_increment','quarter_increment']:
        if name in histories:
            h=histories[name];axes[0,0].plot(h['axial_compression_strain'],h['compression_cauchy_MPa'],'.-',label=name)
    axes[0,0].set(xlabel='Axial compression strain',ylabel='Mean compression [MPa]',title='Increment dependence');axes[0,0].legend()
    for name in ['baseline','yield_half','hardening_double']:
        if name in histories:
            h=histories[name];axes[0,1].plot(h['axial_compression_strain'],h['compression_cauchy_MPa'],'.-',label=name)
    axes[0,1].set(xlabel='Axial compression strain',ylabel='Mean compression [MPa]',title='Uncalibrated material sensitivity');axes[0,1].legend()
    for name in ['baseline','mesh_075mm']:
        if name in histories:
            h=histories[name];axes[1,0].plot(h['step_time'],100*h['porosity'],'.-',label=name)
    axes[1,0].set(xlabel='Step time (not physical time)',ylabel='Porosity [%]',title='Fixed centers, different voxel widths');axes[1,0].legend()
    if 'quarter_increment' in energies:
        h=energies['quarter_increment']
        for key in ['macro_work_Nmm','micro_work_Nmm','solver_internal_energy_Nmm']:
            axes[1,1].plot(h['step_time'],h[key],label=key)
    axes[1,1].set(xlabel='Step time (not physical time)',ylabel='Energy [N mm]',title='Fine-increment work audit');axes[1,1].legend()
    fig.suptitle('Uncalibrated 30 mm elastoplastic trial | no self-contact | not a 0.1 MPa prediction')
    fig.tight_layout();fig.savefig(a.output/'verification-comparison.png',dpi=180);plt.close(fig)
    print(json.dumps(response,indent=2))

if __name__=='__main__':main()
