"""Compare preserved legacy budgets against the refined onset reference."""
import argparse,json,time
from pathlib import Path
import numpy as np
from tools import experimental_surface_budget as patch
from tools.verify_onset_budget import errors,applied,write_csv


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--references',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);rows=[];start=time.perf_counter()
    for fixture in json.loads((a.fixtures/'index.json').read_text()):
        with np.load(a.fixtures/fixture['file']) as z:source=z['source'];targets=z['targets'];mapping=z['mapping']
        with np.load(a.references/f"rule-{fixture['file']}") as z:reference=z['reference_value']
        for norm in ('component','resultant'):
            rule=patch.integrate(source,targets,fixture['reference_area_mm2'],kn=10.,tolerance=.00025,
                                 vertex_map=mapping,groups=fixture['groups'],pointwise=True,force_error_norm=norm)
            for evaluation,value in [('fine_integral',rule['value']),('applied_rule',applied(rule,source,targets,fixture['groups'],mapping))]:
                err=errors(value,reference)
                row=dict(face=fixture['face'],extra_macro_compression=fixture['extra_macro_compression'],method='legacy_'+norm,evaluation=evaluation,
                         points=len(rule['weights']),energy_Nmm=float(value[0]),reference_energy_Nmm=float(reference[0]),
                         energy_allowance_Nmm=rule['energy_allowance'],estimated_energy_error_Nmm=rule['energy_error_estimate'],
                         compression_energy_difference_Nmm=rule['compression_energy_difference'],passed=max(err['energy_error'],err['force_error'])<.001,**err)
                rows.append(row);print(json.dumps(row),flush=True)
        write_csv(a.output/'comparison.csv',rows)
    summary=dict(scope=__doc__,cases=rows,failed_cases=sum(not r['passed'] for r in rows),wall_s=time.perf_counter()-start,
                 reference='Normalized-floor tolerance 1.5625e-5: numerical refinement reference, not exact RVE integration.',
                 production_solver_modified=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
