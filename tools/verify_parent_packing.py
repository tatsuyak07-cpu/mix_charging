"""Frozen packing contact-state diagnostic; NOT an active bulk solver restart."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
from packing_fem.checkpoint import load,restore_contact
from packing_fem.embedded_contact import EmbeddedContact
from packing_fem.embedded_remesh import refine_parent_rules,transfer_parent
from packing_fem.explicit_periodic import Periodic,deformation
from packing_fem.surface_remesh import bisect


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--factor',type=float,default=2.);parser.add_argument('--budget-scale',type=float,default=1.);a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    def hashes():return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.checkpoint.iterdir() if p.is_file()}
    original=hashes();s,meta=load(a.checkpoint);c=meta['config'];X=s['X'];t=s['t'];L=c['length_mm'];p=Periodic(X,L);H=deformation(s['q'][-6:])*L
    kwargs=dict(thickness=c.get('barrier_thickness_mm',.02),max_candidates=c.get('contact_candidate_budget',1000000),quadrature_tolerance=c.get('surface_quadrature_tolerance',.001),
                error_budget='normalized',audit_mode='parent',closest_tie_rule=c.get('closest_tie_rule','clarke'),multiple_patches=c.get('multiple_patches',False),
                edge_stationarity_filter=c.get('edge_stationarity_filter',False),local_material_filter=c.get('local_material_filter',True),certified_local_filter=False,
                absolute_energy_budget=c.get('contact_absolute_energy_budget_Nmm',0.)*a.budget_scale,absolute_force_budget=c.get('contact_absolute_force_budget_N',0.)*a.budget_scale,redistribution_factor=a.factor,
                atlas_max_entries=c.get('atlas_max_matrix_entries',12000000))
    old=EmbeddedContact(X,t,p,c['contact_penalty_mpa_mm'],c['friction'],c['contact_skin_mm'],quadrature_rules=s['runtime_rules'],**kwargs)
    restore_contact(old,s['contact']);baseline=old.replay_force_state(s['x'],X*0,H)
    f0=p.force(s['fc'],s['extra']);f1=p.force(baseline[0],baseline[1]);error=float(np.linalg.norm(f1-f0)/max(np.linalg.norm(f0),1e-16))
    if error>1e-10:raise ValueError('Frozen source force state does not match saved operator: '+str(error))
    rows=[]
    # These are the two previously fully ruled active edges; no unknown nodal
    # histories are split. All ten edges remain covered by the prior analysis.
    for edge in ([1725,1740],[1887,2064]):
        row=dict(edge=edge,passed=False,redistribution_factor=a.factor)
        try:
            proposal=bisect(X,s['x'],t,s['b'],s['ep'],s['V0'],p.velocities(s['v']),edge,L)
            p2=Periodic(proposal['X'],L)
            base=EmbeddedContact(proposal['X'],proposal['t'],p2,c['contact_penalty_mpa_mm'],c['friction'],c['contact_skin_mm'],**kwargs)
            rules,mapping,parent=refine_parent_rules(old,base.faces,proposal)
            new=EmbeddedContact(proposal['X'],proposal['t'],p2,c['contact_penalty_mpa_mm'],c['friction'],c['contact_skin_mm'],quadrature_rules=rules,**kwargs)
            result,report=transfer_parent(old,new,proposal,H,mapping,parent,baseline,c['remesh_energy_tolerance'],redistribute=True)
            row.update(passed=True,transfer=report,audits={str(k):ctx['report'] for k,ctx in new.parent_contexts.items()},geometry=proposal['report'],integration_ledger=new.last_integration_audit)
        except ValueError as exc:row['reason']=str(exc)
        rows.append(row);print(json.dumps(row),flush=True)
    assert original==hashes()
    summary=dict(scope=__doc__,baseline_periodic_force_error=error,cases=rows,passed_candidates=sum(r['passed'] for r in rows),checkpoint_unchanged=True,
                 checkpoint=str(a.checkpoint),input_hashes=original,bulk_restart_performed=False,budget_scale=a.budget_scale,absolute_energy_budget_Nmm=kwargs['absolute_energy_budget'],absolute_force_budget_N=kwargs['absolute_force_budget'],full_compaction_completed=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    source=a.output/'source';source.mkdir()
    for path in Path('packing_fem').glob('*.py'):shutil.copy2(path,source/path.name)
    shutil.copy2(__file__,source/Path(__file__).name)


if __name__=='__main__':main()
