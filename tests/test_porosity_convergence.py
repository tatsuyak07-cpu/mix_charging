from copy import deepcopy
from tools.assess_porosity_convergence import assess


def case():
    rows=[dict(phase='hold',passed=True,porosity=p,axial_compression=e,cauchy_stress_MPa=[[0,0,0],[0,0,0],[0,0,-e]]) for p,e in ((.3,.2),(.2,.4),(.1,.6))]
    rows.append(dict(phase='recovery',passed=True,porosity=.16,axial_compression=.5))
    return dict(initial_porosity=.4,summary=dict(target_loaded_porosity=.1,no_pressure_cap=True,completed=True,
                attempts=[dict(cycle=dict(loaded_target=True,complete=True,loaded_compression=.6,records=rows))]))


def test_identical_complete_cycles_pass():
    assert assess(dict(a=case(),b=case()))['verified']


def test_identical_failed_prefixes_cannot_be_called_convergence():
    a=case();b=case();b['summary']['completed']=False
    r=assess(dict(a=a,b=b));assert not r['eligible'] and not r['verified'] and not r['comparisons']


def test_initial_geometry_error_is_not_hidden_by_normalizing_porosity():
    a=case();b=case();b['initial_porosity']=.403
    r=assess(dict(a=a,b=b));assert not r['verified']
    assert not r['comparisons'][0]['checks']['initial_porosity']


def test_recovery_and_response_errors_fail_separately():
    a=case();b=case();b['summary']['attempts'][0]['cycle']['records'][-1]['porosity']=.17
    r=assess(dict(a=a,b=b));assert not r['verified']
    assert not r['comparisons'][0]['checks']['recovered_porosity']
    assert r['comparisons'][0]['checks']['cauchy_pressure_path']


def test_identical_complete_runs_are_not_a_physical_convergence_study():
    from tools.assess_porosity_convergence import convergence_gate
    a=case();a['config']=dict(young_mpa=10.,cfl=.2,ramp_s=.008,unload_s=.008,mesh_size_mm=.6,surface_quadrature_tolerance=.001)
    r=convergence_gate(dict(a=a,b=deepcopy(a)))
    assert r['response_agreement'] and not r['verified']
    assert all(v['distinct_levels']==1 for v in r['refinement_axes'].values())


def test_three_controlled_levels_per_axis_are_required():
    from tools.assess_porosity_convergence import convergence_gate
    a=case();a['config']=dict(young_mpa=10.,cfl=.2,ramp_s=.008,unload_s=.008,mesh_size_mm=.6,surface_quadrature_tolerance=.001)
    cases={'base':a}
    for axis,values in dict(mesh_size_mm=(.45,.3),cfl=(.1,.05),ramp_s=(.016,.032),surface_quadrature_tolerance=(.0005,.00025)).items():
        for value in values:
            b=deepcopy(a);b['config'][axis]=value;cases[f'{axis}-{value}']=b
    assert convergence_gate(cases)['verified']
    # A change in material cannot count as a controlled time refinement.
    cases['cfl-0.05']['config']['young_mpa']=20.
    assert not convergence_gate(cases)['verified']
