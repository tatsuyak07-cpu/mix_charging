"""Enumerate unchanged-sample area-compatible cuts at saved active packing faces.

Read-only geometric/cubature eligibility analysis. No active solver restart,
friction-history interpolation, reweighting or compaction step is performed.
"""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from packing_fem.checkpoint import load
from packing_fem.periodic_contact import Contact
from packing_fem.explicit_periodic import Periodic
from packing_fem.surface_remesh import bisect
from packing_fem.embedded_remesh import refine_rules
from tools.cut_balance import balance,intersect,residual


def incident(surface,periodic,edge,L):
    X=surface.X;a,b=X[edge];ca,cb=periodic.classes[edge];classes=periodic.classes[surface.faces]
    candidates=np.flatnonzero(np.any(classes==ca,axis=1)&np.any(classes==cb,axis=1));result=[]
    for face in candidates:
        ids=surface.faces[face]
        for ia in np.flatnonzero(classes[face]==ca):
            for ib in np.flatnonzero(classes[face]==cb):
                if ia==ib:continue
                shift=(X[ids[ia]]-a)/L
                if np.allclose(shift,np.rint(shift),atol=1e-10,rtol=0) and np.allclose(X[ids[ib]]-b,L*np.rint(shift),atol=L*1e-10,rtol=0):
                    result.append((int(face),int(ia),int(ib),np.rint(shift).astype(int).tolist()))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    def hashes():return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.checkpoint.iterdir()) if p.is_file()}
    original_hashes=hashes();state,metadata=load(a.checkpoint);c=metadata['config'];X=state['X'];t=state['t'];L=c['length_mm'];periodic=Periodic(X,L)
    surface=Contact(X,t,periodic,c['contact_penalty_mpa_mm'],c['friction'],c['contact_skin_mm'])
    rules=state['runtime_rules'];rules_before=json.dumps(rules,sort_keys=True)
    old=SimpleNamespace(X=X,L=L,faces=surface.faces,rules=rules,kn=c['contact_penalty_mpa_mm'],thickness=c.get('barrier_thickness_mm',.02),quadrature_tolerance=c.get('surface_quadrature_tolerance',.001))
    active={key[1] for key in state['contact']['history']};active_faces=[];start=len(X)
    for key,rule in sorted(rules.items(),key=lambda item:int(item[0])):
        stop=start+len(rule['weights'])
        if any(start<=node<stop for node in active):active_faces.append(int(key))
        start=stop
    edges=sorted({tuple(sorted((int(ids[i]),int(ids[j])))) for ids in surface.faces[active_faces] for i,j in ((0,1),(1,2),(2,0))})
    velocity=periodic.velocities(state['v']);rows=[];face_rows=[]
    def verify(edge,fraction,label):
        result=dict(fraction=float(fraction),label=label,cubature_eligible=False,quality_passed=False)
        try:
            p=bisect(X,state['x'],t,state['b'],state['ep'],state['V0'],velocity,edge,L,fraction=fraction)
            result['geometry']=p['report'];result['quality_passed']=p['report']['minimum_quality_after']>=c['minimum_quality']
            new_surface=Contact(p['X'],p['t'],Periodic(p['X'],L),c['contact_penalty_mpa_mm'],c['friction'],c['contact_skin_mm'])
            refined,points,parent=refine_rules(old,new_surface.faces,p)
            result.update(cubature_eligible=True,retained_points=len(points),child_pose_audits=sum(len(r.get('refinement_pose_audits',[])) for r in refined.values()))
        except ValueError as exc:result['reason']=str(exc)
        result['eligible_for_later_transfer_test']=result['cubature_eligible'] and result['quality_passed']
        return result
    for edge in edges:
        reports=[];common=[[0.,1.]];missing=[]
        for face,ia,ib,shift in incident(surface,periodic,np.array(edge),L):
            tri=X[surface.faces[face]];area=float(np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))/2);tol=max(1e-12,area*1e-10)
            record=dict(face=face,edge_indices=[ia,ib],periodic_shift=shift,has_rule=str(face) in rules)
            flat=dict(edge_a=edge[0],edge_b=edge[1],face=face,has_rule=record['has_rule'],area_mm2=area,area_tolerance_mm2=tol)
            if not record['has_rule']:missing.append(face);common=[]
            else:
                rule=rules[str(face)];report=balance(rule['bary'],rule['weights'],ia,ib,area,tol)
                midpoint=residual(rule['bary'],rule['weights'],ia,ib,.5,area)
                record.update(report);record['midpoint_area_error_mm2']=midpoint
                flat.update(midpoint_area_error_mm2=midpoint,midpoint_relative_error=midpoint/area,allowed_interval_count=len(report['intervals']),root_count=len(report['roots']))
                common=intersect(common,report['intervals'])
            reports.append(record);face_rows.append(flat)
        midpoint=verify(np.asarray(edge),.5,'midpoint')
        candidates=[verify(np.asarray(edge),(lo+hi)/2,'common_interval_representative') for lo,hi in common]
        rows.append(dict(edge=list(edge),faces=reports,missing_rule_faces=sorted(set(missing)),common_area_intervals=common,midpoint=midpoint,candidate_audits=candidates))
        print(json.dumps(dict(edge=edge,missing_rules=sorted(set(missing)),common_intervals=len(common),verified_eligible=sum(v['eligible_for_later_transfer_test'] for v in candidates))),flush=True)
    assert hashes()==original_hashes and json.dumps(rules,sort_keys=True)==rules_before
    summary=dict(scope=__doc__,checkpoint=str(a.checkpoint),input_hashes=original_hashes,checkpoint_unchanged=True,rules_unchanged=True,step=state['steps'],active_histories=len(active),active_faces=active_faces,
                 unique_edges=len(edges),missing_rule_edges=sum(bool(r['missing_rule_faces']) for r in rows),area_incompatible_edges=sum(not r['missing_rule_faces'] and not r['common_area_intervals'] for r in rows),
                 edges_with_area_intervals=sum(bool(r['common_area_intervals']) for r in rows),verified_candidate_count=sum(len(r['candidate_audits']) for r in rows),
                 eligible_candidates=sum(v['eligible_for_later_transfer_test'] for r in rows for v in r['candidate_audits']),edges=rows,packing_remesh_completed=False,full_compaction_completed=False,
                 limits=['Area balance is necessary, not sufficient for history/force/material/velocity transfer.',
                         'Intervals include both possible whole-apex assignments; actual refine_rules selects its deterministic owner.',
                         'Only existing samples and current full-child-face rule requirements are considered.',
                         'One representative per common interval is audited; no proof of every historical response throughout an interval.'])
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (a.output/'face-balance.csv').open('w',newline='') as stream:
        fields=['edge_a','edge_b','face','has_rule','area_mm2','area_tolerance_mm2','midpoint_area_error_mm2','midpoint_relative_error','allowed_interval_count','root_count'];w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(face_rows)
    with (a.output/'edge-outcomes.csv').open('w',newline='') as stream:
        fields=['edge_a','edge_b','missing_rule_faces','common_interval_count','audited_candidates','eligible_candidates','midpoint_reason'];w=csv.DictWriter(stream,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow(dict(edge_a=r['edge'][0],edge_b=r['edge'][1],missing_rule_faces=' '.join(map(str,r['missing_rule_faces'])),common_interval_count=len(r['common_area_intervals']),audited_candidates=len(r['candidate_audits']),eligible_candidates=sum(v['eligible_for_later_transfer_test'] for v in r['candidate_audits']),midpoint_reason=r['midpoint'].get('reason')))
    source=a.output/'source';source.mkdir()
    for path in [Path(__file__),Path('tools/cut_balance.py'),Path('packing_fem/embedded_remesh.py'),Path('packing_fem/surface_remesh.py')]:shutil.copy2(path,source/path.name)


if __name__=='__main__':main()
