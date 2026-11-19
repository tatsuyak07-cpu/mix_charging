"""Separate repaired startup checks, inactive screens and contact-active sensitivity."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np


PHYSICAL=['length_mm','diameter_mm','neck_diameter_mm','young_mpa','poisson','yield_mpa','hardening_mpa','friction','transverse_boundary','rigid_inclusion','rigid_diameter_mm','rigid_height_mm','rigid_segments','rigid_center_mm','ipc_stiffness_mpa','ipc_slip_regularization_mm','implicit_force_tolerance_N','implicit_macro_tolerance']


def read_case(run,original):
    run=Path(run);c=json.loads((run/'config.json').read_text());a=json.loads((run/'audit.json').read_text())
    if any(c.get(k)!=original.get(k) for k in PHYSICAL):raise ValueError('Material, geometry, friction, boundary or acceptance changed')
    with (run/'history.csv').open() as f:rows=list(csv.DictReader(f))
    last=rows[-1];state=run/f"state-{int(last['step']):04d}.npz";digest=hashlib.sha256(state.read_bytes()).hexdigest()
    if not a['passed'] or a['states'][-1]['state_sha256']!=digest:raise ValueError('Missing or stale independent audit')
    if abs(float(last['nominal_pressure_MPa'])-a['states'][-1]['pressure_MPa'])>1e-10:raise ValueError('Pressure differs from audit')
    return dict(run=str(run),dhat_mm=c['ipc_dhat_mm'],accepted_increments=len(rows),
        compression=float(last['compression']),pressure_MPa=float(last['nominal_pressure_MPa']),
        porosity=float(last['porosity']),contacts=int(last['contacts']),rigid_contacts=int(last['rigid_contacts']),
        maximum_ep=float(last['maximum_peeq']),free_residual_N=float(last['free_residual_N']),
        state_sha256=digest,audit_passed=True)


def assess(root,out,active_runs=None):
    root=Path(root);out=Path(out)
    if out.exists():raise FileExistsError(out)
    original=json.loads((root/'input/original-contact-config.json').read_text())
    plan=json.loads((root/'repair-plan.json').read_text())
    extension=root/'extended-contact-plan.json'
    if extension.exists():
        extended=json.loads(extension.read_text());family=extended['extended_family_mm'];candidate=extended['candidate_activation_mm']
    else:family=plan['contact_activation_mm'];candidate=plan['candidate_activation_mm']
    initial_names=['fine-small','refined-small','fine-010','fine-005','fine-0025','coarse-005','refined-005']
    initial=[read_case(root/'initial-study-001'/name,original) for name in initial_names]
    initial.append(read_case(root/'initial-extra-001/coarse-small',original))
    startup=[]
    for case in initial:
        report=json.loads((Path(case['run'])/'initial-contact-preflight.json').read_text())
        startup.append(dict(run=case['run'],**report))
    low=[r for r in initial if r['compression']==.0001]
    initial_pass=all(r['passed'] for r in startup) and all(r['maximum_ep']==0 for r in low)
    if active_runs is None:
        active_runs=([root/'active-hybrid-001/dhat-005-hybrid',root/'active-secant-001/dhat-0025-secant',root/'active-extended-001/dhat-00125-secant'] if extension.exists() else [root/'active-study-001'/('dhat-'+n) for n in ['010','005','0025']])
    active=sorted([read_case(p,original) for p in active_runs],key=lambda c:-c['dhat_mm'])
    if len(active)!=3 or [r['dhat_mm'] for r in active]!=family:raise ValueError('Need the declared three contact activation levels')
    first=dict(np.load(Path(active[0]['run'])/'state-0000.npz'))
    for case in active[1:]:
        with np.load(Path(case['run'])/'state-0000.npz') as state:
            if set(first)!=set(state.files) or any(not np.array_equal(first[k],state[k]) for k in first):raise ValueError('Contact controls do not share the same accepted history')
    same_strain=all(abs(r['compression']-active[0]['compression'])<1e-10 for r in active)
    differences=[dict(pressure_relative=abs(b['pressure_MPa']-a['pressure_MPa'])/abs(b['pressure_MPa']),porosity_absolute=abs(b['porosity']-a['porosity'])) for a,b in zip(active,active[1:])]
    decreasing=all(differences[-1][k]<=differences[0][k]+1e-12 for k in differences[0])
    contact_pass=bool(same_strain and all(r['accepted_increments']==1 and r['contacts']>0 and r['rigid_contacts']>0 for r in active) and decreasing and differences[-1]['pressure_relative']<=plan['pressure_relative_screen'] and differences[-1]['porosity_absolute']<=plan['porosity_absolute_screen'])
    coarser=None
    if extension.exists():
        old_runs=[root/'active-study-001/dhat-010',root/'active-hybrid-001/dhat-005-hybrid',root/'active-secant-001/dhat-0025-secant']
        old_cases=[read_case(p,original) for p in old_runs]
        old_differences=[dict(pressure_relative=abs(b['pressure_MPa']-a['pressure_MPa'])/abs(b['pressure_MPa']),porosity_absolute=abs(b['porosity']-a['porosity'])) for a,b in zip(old_cases,old_cases[1:])]
        old_decrease=all(old_differences[-1][k]<=old_differences[0][k]+1e-12 for k in old_differences[0])
        coarser=dict(cases=old_cases,adjacent_differences=old_differences,errors_decrease=old_decrease,passed=bool(old_decrease and old_differences[-1]['pressure_relative']<=plan['pressure_relative_screen'] and old_differences[-1]['porosity_absolute']<=plan['porosity_absolute_screen']))
    prepared=json.loads((root/'prepared-mesh-001/metadata.json').read_text())
    coarse=next(r for r in low if r['run'].endswith('coarse-small'));fine=next(r for r in low if r['run'].endswith('refined-small'))
    inactive=[r for r in initial if Path(r['run']).name in ['fine-010','fine-005','fine-0025']]
    adoption=[]
    for name in ['refined-small','refined-005']:
        case=read_case(root/f'candidate-replay-0025/{name}',original)
        provenance=json.loads((Path(case['run'])/'provenance.json').read_text())
        if provenance['new_nonlinear_solve_performed'] or case['dhat_mm']!=candidate:raise ValueError('Adoption replay is mislabeled')
        adoption.append(case)
    result=dict(startup_preload_removed=initial_pass,contact_active_sensitivity_passed=contact_pass,
        prototype_repairs_verified=bool(initial_pass and contact_pass and prepared['geometry_unchanged'] and all(r['contacts']==0 for r in adoption)),
        candidate_activation_mm=candidate,initially_declared_candidate_mm=plan['candidate_activation_mm'],
        original_activation_mm=original['ipc_dhat_mm'],
        scope='Prototype startup and same-faceted-boundary input refinement; active sensitivity is conditional on the SAME old accepted history, not full paths under each new activation.',
        material_friction_geometry_boundary_unchanged=True,initial_cases=initial,startup_checks=startup,
        active_cases=active,active_adjacent_differences=differences,active_errors_decrease=decreasing,
        original_coarser_family_passed=coarser['passed'] if coarser else contact_pass,
        original_coarser_family=coarser,
        extended_study_plan=None if not extension.exists() else extended,
        adoption_equilibrium_replays=adoption,
        inactive_initial_screen_has_no_contacts=all(r['contacts']==0 for r in inactive),
        inactive_initial_screen_is_not_active_contact_validation=True,
        same_surface_refinement=prepared,
        small_strain_pressure_reduction_relative=1-fine['pressure_MPa']/coarse['pressure_MPa'],
        initial_guard_expected_rejection=json.loads((root/'guard-rejection-assessment.json').read_text()),
        full40to10_path_recomputed=False,mesh_convergence_verified=False,full_path_increment_convergence_verified=False)
    out.write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--active-run',action='append');a=p.parse_args()
    result=assess(a.root,a.out,a.active_run)
    if not result['prototype_repairs_verified']:raise SystemExit(1)
