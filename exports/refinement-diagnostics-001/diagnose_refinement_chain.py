"""Audit saved EE guards without changing contact forces or advancing time."""
import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from packing_fem.barrier_contact import segment_distances
from packing_fem.contact_tracking import closest_segments


def point_gap(points, s):
    a, b, c, d = np.asarray(points)
    v = d - c
    q = a + s * (b - a)
    t = np.clip((q - c) @ v / (v @ v), 0., 1.)
    return float(np.linalg.norm(q - c - t * v))


def active_interval(points, thickness, s):
    """Distance to a convex segment is convex; its strict sublevel is an interval.

    The supplied closest parameter brackets the two boundary roots. A zero
    width means that no point is strictly inside the barrier thickness.
    """
    gap = lambda value: point_gap(points, value) - thickness
    if gap(s) >= 0:
        return None
    left = 0. if gap(0.) < 0 else brentq(gap, 0., s, xtol=1e-14)
    right = 1. if gap(1.) < 0 else brentq(gap, s, 1., xtol=1e-14)
    return float(left), float(right)


def measure(points, thickness):
    points = np.asarray(points, float)
    hit = closest_segments(*points)
    s = float(hit['s'])
    interval = active_interval(points, thickness, s)
    independent = minimize_scalar(lambda v: point_gap(points, v)**2,
                                  bounds=(0., 1.), method='bounded',
                                  options={'xatol': 1e-14})
    independent_gap = min(point_gap(points, independent.x), point_gap(points, 0.), point_gap(points, 1.))
    d = point_gap(points, s)
    return dict(source_length_mm=float(np.linalg.norm(points[1] - points[0])),
                closest_source_parameter=s, closest_target_parameter=float(hit['t']),
                minimum_gap_mm=d, independent_minimum_gap_mm=independent_gap,
                midpoint_gap_mm=point_gap(points, .5),
                closest_point_inside_barrier=bool(d < thickness),
                midpoint_inside_barrier=bool(point_gap(points, .5) < thickness),
                active_interval=interval,
                active_length_mm=0. if interval is None else (interval[1] - interval[0]) * float(np.linalg.norm(points[1] - points[0])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='calculation directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    summary = json.loads((args.input / 'summary.json').read_text())
    config = json.loads((args.input / 'config.json').read_text())
    thickness = config['barrier_thickness_mm']
    rows, sensitivity = [], []
    theta = .47
    R = np.array([[np.cos(theta), -np.sin(theta), 0.], [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
    for i, attempt in enumerate(summary['surface_resolution_attempts'], 1):
        points = np.array(attempt['feature']['points'])
        row = dict(attempt=i, accepted_time_s=attempt['accepted_time_s'],
                   repaired=attempt['repaired'], source_ids=attempt['feature']['ids'][:2],
                   **measure(points, thickness))
        row['vectorized_gap_error_mm'] = abs(row['minimum_gap_mm'] - float(segment_distances(points[None])[0]))
        rows.append(row)
        # Endpoint and pair order, rigid motion, and small coordinate noise.
        variants = {'reverse_source': points[[1, 0, 2, 3]],
                    'reverse_target': points[[0, 1, 3, 2]],
                    'swap_pair': points[[2, 3, 0, 1]],
                    'rigid_motion': points @ R.T + [.31, -.27, .19]}
        for name, p in variants.items():
            m = measure(p, thickness)
            sensitivity.append(dict(attempt=i, variant=name,
                                    gap_error_mm=abs(m['minimum_gap_mm'] - row['minimum_gap_mm']),
                                    guard_active_unchanged=m['closest_point_inside_barrier'] == row['closest_point_inside_barrier']))
        for sign in (-1., 1.):
            p = points.copy()
            p[1, 0] += sign * 1e-12
            m = measure(p, thickness)
            sensitivity.append(dict(attempt=i, variant=f'noise_{sign:+g}',
                                    gap_error_mm=abs(m['minimum_gap_mm'] - row['minimum_gap_mm']),
                                    guard_active_unchanged=m['closest_point_inside_barrier'] == row['closest_point_inside_barrier']))
    # Diagnostic polynomial/root calculations do not authorize contact exclusion.
    checks = dict(
        actual_guard_gaps_reproduced=all(abs(r['minimum_gap_mm'] - a['feature']['distance_mm']) < 1e-10 for r, a in zip(rows, summary['surface_resolution_attempts'])),
        independent_scalar_minimum=all(abs(r['minimum_gap_mm'] - r['independent_minimum_gap_mm']) < 1e-10 for r in rows),
        vectorized_minimum=all(r['vectorized_gap_error_mm'] < 1e-10 for r in rows),
        all_guards_inside_barrier=all(r['closest_point_inside_barrier'] for r in rows),
        all_first_midpoints_outside_barrier=all(not r['midpoint_inside_barrier'] for r in rows),
        active_interval_roots=all(r['active_interval'] is not None and all(abs(point_gap(np.array(a['feature']['points']), v) - thickness) < 1e-10 for v in r['active_interval'] if 0 < v < 1) for r, a in zip(rows, summary['surface_resolution_attempts'])),
        ordering_rigid_motion_noise_stability=all(r['guard_active_unchanged'] and r['gap_error_mm'] < 2e-12 for r in sensitivity),
        cap_stop_reproduced=len(summary['remesh_events']) == config['surface_remesh_max_events'] and not rows[-1]['repaired'],
    )
    result = dict(scope=__doc__, checks=checks, passed=all(checks.values()),
                  barrier_thickness_mm=thickness, attempts=rows, sensitivity=sensitivity,
                  contact_candidate_order_sensitivity_verified=False,
                  full_surface_quadrature_resolution_verified=False,
                  time_advanced=False, physical_solution_verified=False,
                  interpretation='Midpoint insertion misses the active part of each flagged edge. Tests of isolated pairs do not establish stability of patch selection, history transfer, or the full trajectory.')
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    flat = [{k: v for k, v in r.items() if k not in ('source_ids', 'active_interval')} for r in rows]
    with (out / 'refinement-chain.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    shutil.copy2(args.input / 'summary.json', out / 'input-summary.json')
    shutil.copy2(args.input / 'config.json', out / 'input-config.json')
    shutil.copy2(Path(__file__), out / 'diagnose_refinement_chain.py')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    index = [r['attempt'] for r in rows]
    ax[0].semilogy(index, [r['midpoint_gap_mm'] for r in rows], 'o-', label='Inserted midpoint to target edge')
    ax[0].plot(index, [r['minimum_gap_mm'] for r in rows], '.-', label='Closest pair gap')
    ax[0].axhline(thickness, color='red', linestyle=':', label='Barrier thickness')
    ax[0].set(xlabel='Guard attempt', ylabel='Gap [mm]')
    ax[1].semilogy(index, [r['source_length_mm'] for r in rows], 'o-', label='Source edge length')
    ax[1].semilogy(index, [r['active_length_mm'] for r in rows], 'o-', label='Active subinterval length')
    ax[1].set(xlabel='Guard attempt', ylabel='Length [mm]')
    for a in ax:
        a.grid(alpha=.25);a.legend(fontsize=8)
    fig.suptitle('Saved trial-pair geometry only — no pressure integration')
    fig.tight_layout();fig.savefig(out / 'refinement-chain.png', dpi=180);plt.close(fig)
    hashes = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}
    (out / 'sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    with zipfile.ZipFile(out.with_suffix('.zip'), 'x', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(out.iterdir()):
            if p.is_file():archive.write(p, str(Path(out.name) / p.name))
    print(json.dumps(dict(checks=checks, passed=result['passed'], last_attempt=rows[-1]), indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
