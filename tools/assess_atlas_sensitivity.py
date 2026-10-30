"""Three-level numerical-budget sensitivity on completed contact-onset prefixes only."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from tools.assess_contact_passage import assess


def assess_levels(runs):
    cases=[];configs=[];curves=[]
    for run in runs:
        run=Path(run);s=json.loads((run/'summary.json').read_text());p=run/f"attempt-{s['attempts'][-1]['attempt']:03d}"
        c=json.loads((p/'config.json').read_text());configs.append(c)
        rows=[json.loads(x) for x in (p/'progress.jsonl').read_text().splitlines() if json.loads(x)['step']>90206];curves.append(rows)
        ledger=[json.loads(x) for x in (p/'contact-integration-budget.jsonl').read_text().splitlines()]
        fb=c['contact_absolute_force_budget_N'];eb=c['contact_absolute_energy_budget_Nmm']
        ledger_pass=([x['step'] for x in ledger]==list(range(90207,90251)) and
                     all(x['absolute_force_excess_L1_N']<=fb/2 and x['absolute_energy_excess_Nmm']<=eb/2 for x in ledger))
        cases.append(dict(run=str(run),passage=assess(run),force_budget_N=fb,energy_budget_Nmm=eb,
                          global_normal_ledger_passed=ledger_pass,max_normal_force_excess_N=max(x['absolute_force_excess_L1_N'] for x in ledger),
                          max_normal_energy_excess_Nmm=max(x['absolute_energy_excess_Nmm'] for x in ledger)))
    ignored={'contact_absolute_force_budget_N','contact_absolute_energy_budget_Nmm','embedded_quadrature_rules','stop_request_file'}
    clean=[{k:v for k,v in c.items() if k not in ignored} for c in configs]
    checks=dict(three_levels=len(runs)==3,identical_other_settings=all(c==clean[0] for c in clean),
                descending_budgets=all(cases[i]['force_budget_N']>cases[i+1]['force_budget_N'] and cases[i]['energy_budget_Nmm']>cases[i+1]['energy_budget_Nmm'] for i in range(len(cases)-1)),
                all_prefixes_passed=all(c['passage']['passed'] for c in cases),all_ledgers_passed=all(c['global_normal_ledger_passed'] for c in cases))
    comparisons=[];grid_rows=[]
    if all(checks.values()):
        compression=[np.array([r['axial_compression'] for r in rows]) for rows in curves]
        checks['monotone_compression']=all(np.all(np.diff(x)>0) for x in compression)
        if checks['monotone_compression']:
            grid=np.linspace(max(x[0] for x in compression),min(x[-1] for x in compression),201)
            thresholds={'porosity':1e-6,'nominal_pressure_MPa':1e-4,'internal_Nmm':1e-4}
            for key,limit in thresholds.items():
                y=[np.interp(grid,x,[r[key] for r in rows]) for x,rows in zip(compression,curves)]
                scale=1 if key=='porosity' else max(np.max(np.abs(y[-1])),1e-30)
                error=max(float(np.max(np.abs(v-y[-1]))/scale) for v in y[:-1])
                comparisons.append(dict(quantity=key,error=error,criterion=limit,passed=error<=limit,normalization='absolute' if key=='porosity' else 'maximum absolute finest-level value'))
                for i,g in enumerate(grid):grid_rows.append(dict(quantity=key,axial_compression=float(g),**{f'level_{j}':float(v[i]) for j,v in enumerate(y)}))
            checks['response_sensitivity']=all(c['passed'] for c in comparisons)
    return dict(passed=all(checks.values()),checks=checks,cases=cases,comparisons=comparisons,
                scope='44-step contact-onset prefix; optional absolute NORMAL integration error budgets only.',
                config_comparison_exclusions=sorted(ignored),
                limitations=['No static endpoint, loaded 10% target, unloading, timestep or spatial convergence established.',
                             'Developer-defined response sensitivity criteria; no rigorous tangential/friction integration error bound.',
                             'Measured normal excess is limited to half the global budget; reference uncertainty adds at most one eighth, in addition to the relative criterion.']),grid_rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,nargs=3,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    result,rows=assess_levels(a.runs);(a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    if rows:
        with (a.output/'response-comparison.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(result));raise SystemExit(0 if result['passed'] else 2)


if __name__=='__main__':main()
