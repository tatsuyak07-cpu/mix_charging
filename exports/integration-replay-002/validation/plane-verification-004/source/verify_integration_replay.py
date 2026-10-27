"""Contact-only history replay prerequisite; no adaptive packing solver claim.

Refined surface points are evaluated on a fully prescribed, archived motion
from a known inactive interface. No coarse-point history is interpolated.
"""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from packing_fem.area_contact import AreaContact
from packing_fem.barrier_contact import barrier
from packing_fem.contact_audit import plain
from packing_fem.explicit_periodic import Periodic
from tools.verify_plane_contact import fixture


# mm, N, MPa: same contact law as the previous plane verification.
KN, MU, THICKNESS = 20., .3, .02
# All sliding starts after first contact at zero tangential displacement.
WAYPOINTS = [(0., .025), (0., .015), (.0001, .015), (.02, .015),
             (.012, .015), (-.01, .015), (.005, .015)]


def trajectory(subdivisions):
    points = [WAYPOINTS[0]]
    for left, right in zip(WAYPOINTS, WAYPOINTS[1:]):
        points.extend(tuple(np.array(left)+(np.array(right)-left)*j/subdivisions)
                      for j in range(1, subdivisions+1))
    return points


def make(n, phase):
    X, t, _, _, upper = fixture(n, .015, phase*n)
    c = AreaContact(X, t, Periodic(X, 1.), KN, MU, .1,
                    thickness=THICKNESS, quadrature_mode='material', closest_tie_rule='clarke')
    return X, t, upper, c


def step(c, X, upper, phase, slip, gap):
    x = X.copy(); x[upper, 0] += phase+slip
    x[upper, 1] += .37*phase; x[upper, 2] -= .2-gap
    H = np.eye(3); H[:2, 2] = (phase+slip, .37*phase); H[2, 2] = .8+gap
    response = c.evaluate(x, x*0, H, H*0, 1.)
    c.history = response[2]; c.accept_geometry(x, H)
    r = response[-1]
    return dict(force_x_N=float(response[0][upper, 0].sum()),
                force_z_N=float(response[0][upper, 2].sum()),
                normal_energy_Nmm=float(r['normal_energy']),
                tangent_energy_Nmm=float(r['tangent_energy']),
                friction_increment_Nmm=float(r['friction_increment']),
                force_balance_N=float(np.linalg.norm(response[0].sum(axis=0))),
                coulomb_ratio=float(r['max_coulomb_ratio']), contacts=r['contacts'])


def analytic(points):
    """Independent scalar spring/Coulomb return mapping on 1 mm² flat area.

    On this fixed plane the production moving-anchor rule reduces to an
    elastic slip increment, clipped at mu*normal traction / (KN/2).
    """
    spring = 0.; previous_slip = 0.; active = False; result = []
    for slip, gap in points:
        energy, normal, _ = barrier(gap, KN, THICKNESS)
        if gap >= THICKNESS:
            spring = 0.; work = 0.; active = False
        else:
            trial = spring+(slip-previous_slip) if active else 0.
            limit = MU*normal/(KN/2)
            updated = float(np.clip(trial, -limit, limit))
            work = (KN/2)*updated*(trial-updated)
            spring = updated; active = True
        result.append(dict(force_x_N=-(KN/2)*spring, force_z_N=normal,
                           normal_energy_Nmm=energy, tangent_energy_Nmm=.5*(KN/2)*spring**2,
                           friction_increment_Nmm=work))
        previous_slip = slip
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    root = args.output; root.mkdir(parents=True, exist_ok=False)
    rows = []; cases = []
    keys = list(analytic(WAYPOINTS)[0])
    for n in (1, 2, 4):
        for phase in (0., .125):
            for subdivisions in (8, 16):
                name = f'n{n}-phase{phase}-sub{subdivisions}'
                folder = root/name; folder.mkdir()
                points = trajectory(subdivisions); reference = analytic(points)
                X, t, upper, uninterrupted = make(n, phase)
                direct = [step(uninterrupted, X, upper, phase, *point) for point in points]
                # A newly requested integration set is reconstructed from the
                # inactive checkpoint, not initialized at the current contact.
                _, _, _, replay = make(n, phase)
                switch = 3*subdivisions
                for point in points[:switch+1]:
                    at_switch = step(replay, X, upper, phase, *point)
                restored_committed = deepcopy(replay.history)
                restored_applied = deepcopy(replay.accepted_force_history)
                replayed = [at_switch]
                for point in points[switch+1:]:
                    replayed.append(step(replay, X, upper, phase, *point))
                # Negative control: identical current geometry, unknown past.
                _, _, _, cold = make(n, phase)
                cold_state = step(cold, X, upper, phase, *points[switch])
                history = dict(committed_at_insertion=restored_committed,
                               applied_at_insertion=restored_applied,
                               committed_final=replay.history,
                               applied_final=replay.accepted_force_history)
                (folder/'history.json').write_text(json.dumps({name: [dict(key=plain(key), state=plain(state)) for key, state in states.items()] for name, states in history.items()}, indent=2)+'\n')
                np.savez_compressed(folder/'input.npz', X=X, tetrahedra=t,
                                    upper=upper, trajectory=points, area_weights=replay.integration_area0)
                errors = {key: max(abs(a[key]-b[key]) for a, b in zip(direct, reference)) for key in keys}
                replay_error = max(abs(a[key]-b[key]) for a, b in zip(direct[switch:], replayed) for key in keys)
                cold_error = abs(cold_state['force_x_N']-direct[switch]['force_x_N'])
                checks = dict(analytic_force_and_energy=max(errors.values())<1e-10,
                              replay_matches_uninterrupted=replay_error<1e-12,
                              cold_initialization_detected=cold_error>1e-4,
                              action_reaction=max(a['force_balance_N'] for a in direct)<1e-10,
                              coulomb=max(a['coulomb_ratio'] for a in direct)<=MU+1e-12,
                              nonnegative_dissipation=min(a['friction_increment_Nmm'] for a in direct)>-1e-12,
                              starts_inactive=direct[0]['contacts']==0,
                              nonzero_history=any(np.linalg.norm(v['spring'])>1e-8 for v in restored_committed.values()))
                case = plain(dict(name=name, n=n, phase_mm=phase, subdivisions=subdivisions,
                            insertion_step=switch, checks=checks, passed=all(checks.values()),
                            analytic_absolute_errors=errors, replay_maximum_absolute_error=replay_error,
                            cold_force_error_N=cold_error,
                            final_force_x_N=direct[-1]['force_x_N'],
                            total_friction_work_Nmm=sum(a['friction_increment_Nmm'] for a in direct)))
                cases.append(case)
                (folder/'summary.json').write_text(json.dumps(case, indent=2)+'\n')
                for index, (point, actual, expected) in enumerate(zip(points, direct, reference)):
                    rows.append(dict(case=name, step=index, slip_mm=point[0], gap_mm=point[1],
                                     **actual, **{'analytic_'+key: value for key, value in expected.items()}))
                print(json.dumps(case), flush=True)
    sensitivity = {}
    for key in ('final_force_x_N', 'total_friction_work_Nmm'):
        values = [c[key] for c in cases]; sensitivity[key+'_range'] = max(values)-min(values)
    checks = dict(all_cases=all(c['passed'] for c in cases),
                  mesh_phase_step_endpoint_invariance=max(sensitivity.values())<1e-10)
    summary = dict(scope=__doc__, checks=checks, passed=all(checks.values()), cases=cases,
                   sensitivity=sensitivity, units='mm N MPa', friction=MU,
                   limits=['Prescribed flat fixed-normal motion only; not a bulk FEM solve.',
                           'No adaptive integration or replay of a coupled bulk trajectory implemented.',
                           'No general curved-normal, patch-change, or nonuniform-slip verification.',
                           'Quadrature change conservation and narrow contact integration remain unverified.',
                           'No 0.1 MPa equilibrium or porosity prediction.'])
    (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (root/'trace.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    source = root/'source'; source.mkdir()
    for file in ('tools/verify_integration_replay.py', 'tools/verify_plane_contact.py',
                 'packing_fem/area_contact.py', 'packing_fem/barrier_contact.py'):
        (source/Path(file).name).write_bytes(Path(file).read_bytes())
    (root/'manifest.json').write_text(json.dumps({str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
                                                for f in sorted(root.rglob('*')) if f.is_file()}, indent=2)+'\n')
    print(json.dumps(dict(passed=summary['passed'], checks=checks, sensitivity=sensitivity)), flush=True)
    if not summary['passed']: raise SystemExit(1)


if __name__ == '__main__':
    main()
