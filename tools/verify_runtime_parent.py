"""Prescribed plane look-ahead redistribution; no bulk compaction claim."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from tools.verify_parent_hybrid import case
from packing_fem.embedded_remesh import transfer_parent
from packing_fem.runtime_quadrature import repair
from packing_fem.checkpoint import save,load,contact_state,restore_contact
from packing_fem.embedded_contact import EmbeddedContact


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cases=[]
    for tol in (.001,.0005,.00025):
        for slip in (.0001,.01):
            row=dict(tolerance=tol,initial_slip_mm=slip,passed=False,events=[])
            try:
                old,c,proposal,H,before,mapping,parent=case(slip=slip,quadrature_order=8,tolerance=tol)
                response,_=transfer_parent(old,c,proposal,H,mapping,parent,before,1e-8,redistribute=True)
                x=proposal['x'].copy();c.accept_geometry(x,H)
                for dx in (.0001,-.0001):
                    y=x.copy();G=H.copy();y[proposal['X'][:,2]>.5,0]+=dx;G[0,2]+=dx
                    candidate,later,report=repair(c,proposal['t'],x,x*0,H,y,y*0,G,G*0,.0001,response)
                    if later[-1]['max_coulomb_ratio']>c.mu+1e-10:raise ValueError('Coulomb bound exceeded')
                    candidate.history=later[2];candidate.accept_geometry(y,G)
                    row['events'].append(dict(dx_mm=dx,report=report,coulomb_ratio=later[-1]['max_coulomb_ratio'],friction_increment_Nmm=later[-1]['friction_increment'],audits={str(k):v['report'] for k,v in candidate.parent_contexts.items()}))
                    c,response,x,H=candidate,later,y,G
                # Roundtrip the actual migrated rules, fixed reference and both histories.
                directory=a.output/f'checkpoint-{len(cases):02d}'
                save(directory,dict(contact=contact_state(c),rules=c.rules,x=x,H=H),dict(scope=__doc__))
                state,_=load(directory)
                resumed=EmbeddedContact(c.X,proposal['t'],c.periodic,c.kn,c.mu,c.skin,quadrature_rules=state['rules'],**c._constructor_options)
                restore_contact(resumed,state['contact']);again=resumed.replay_force_state(x,x*0,H)
                error=float(np.linalg.norm(c.periodic.force(again[0]-response[0],again[1]-response[1]))/max(np.linalg.norm(c.periodic.force(response[0],response[1])),1e-16))
                if error>1e-10:raise ValueError('Contact serialization/replay mismatch')
                if resumed.rules!=c.rules:raise ValueError('Fixed reference weights lost at checkpoint')
                row.update(passed=True,checkpoint_force_error=error)
            except ValueError as exc:row['reason']=str(exc)
            cases.append(row);print(json.dumps(row),flush=True)
    summary=dict(passed=all(r['passed'] for r in cases),cases=cases,scope=__doc__,full_compaction_completed=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    dst=a.output/'source';dst.mkdir()
    for file in Path('packing_fem').glob('*.py'):shutil.copy2(file,dst/file.name)
    shutil.copy2(__file__,dst/Path(__file__).name)
    if not summary['passed']:raise SystemExit(2)


if __name__=='__main__':main()
