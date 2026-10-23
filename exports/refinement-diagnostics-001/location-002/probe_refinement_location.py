"""Compare insertion locations on one saved trial, without state transfer."""
import argparse
import json
from pathlib import Path

import numpy as np

from packing_fem.adaptive_tet import boundary, quality
from packing_fem.contact_geometry import closest_triangle
from packing_fem.contact_tracking import closest_segments
from packing_fem.explicit_material import geometry
from packing_fem.explicit_periodic import Periodic
from packing_fem.surface_remesh import bisect
from packing_fem.surface_stationarity import SurfaceStationarity


def geometry_split(X, x, t, edge, fraction, L):
    """Geometry-only off-centre split of all translated copies of an edge.

    Returned arrays deliberately omit material state/history/mass. They cannot
    be used as a transferred solver state.
    """
    if not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError('An interior insertion fraction is required')
    n = len(X)
    p = bisect(X, x, t, np.repeat(np.eye(3)[None], len(t), axis=0),
               np.zeros(len(t)), geometry(X, t)[0], x * 0, edge, L)
    u = X[edge[1]] - X[edge[0]]
    for node in range(n, len(p['X'])):
        a, b = p['supports'][node]
        f = fraction if (X[b] - X[a]) @ u > 0 else 1. - fraction
        p['X'][node] = (1-f) * X[a] + f * X[b]
        p['x'][node] = (1-f) * x[a] + f * x[b]
    return p['X'], p['x'], p['t']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--calculation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    c = json.loads((args.calculation / 'config.json').read_text())
    summary = json.loads((args.calculation / 'summary.json').read_text())
    replay = json.loads((args.trial / 'summary.json').read_text())
    with np.load(args.trial / 'trial-geometry.npz') as z:
        X, x, t, H = z['reference_points'], z['trial_points'], z['tets'], z['cell']
    feature = summary['failure_feature'];ids = np.array(feature['ids']);shift = np.array(feature['shift'])
    points = x[ids].copy();points[2:] += H @ shift
    hit = closest_segments(*points)
    stationarity = SurfaceStationarity(X, boundary(t), Periodic(X, c['length_mm']).classes, c['length_mm'])
    star = stationarity.star(ids[2:], shift)
    source_star = stationarity.star(ids[:2], np.zeros(3, int))
    rows = []
    for name, f, side in (('midpoint', .5, 0), ('closest_parameter', float(hit['s']), 0), ('target_closest_parameter', float(hit['t']), 1)):
        node = (1-f) * points[2*side] + f * points[2*side+1]
        opposite_star = star if side == 0 else source_star
        distances = [float(np.linalg.norm(node - closest_triangle(node, x[face] + H @ lift)[0])) for face, lift in opposite_star]
        Y, y, newt = geometry_split(X, x, t, ids[2*side:2*side+2], f, c['length_mm'])
        rows.append(dict(location=name, fraction=f, side=side,
                         target_incident_triangle_gap_mm=min(distances),
                         inside_target_star_barrier=min(distances) < c['barrier_thickness_mm'],
                         minimum_quality=float(quality(y, newt).min()),
                         reference_volume_error_mm3=float(abs(geometry(Y, newt)[0].sum() - geometry(X, t)[0].sum())),
                         trial_volume_error_mm3=float(abs(geometry(y, newt)[0].sum() - geometry(x, t)[0].sum()))))
        np.savez_compressed(args.output / f'{name}-geometry-only.npz', reference_points=Y, points=y, tets=newt)
    error = float(np.max(abs(points - np.array(feature['points']))))
    checks = dict(failed_trial_geometry_reproduced=error < 1e-10,
                  same_guard_reproduced=replay['replayed_error']['feature']['ids'] == feature['ids'],
                  complete_target_edge_star=bool(star),
                  midpoint_misses_target_star=not rows[0]['inside_target_star_barrier'],
                  closest_insertion_enters_target_star=rows[1]['inside_target_star_barrier'],
                  target_insertion_enters_source_star=bool(source_star) and rows[2]['inside_target_star_barrier'],
                  trial_geometries_above_quality_floor=all(r['minimum_quality'] >= c['minimum_quality'] for r in rows),
                  volumes_preserved=all(r['reference_volume_error_mm3'] < 1e-9 and r['trial_volume_error_mm3'] < 1e-9 for r in rows))
    result = dict(scope=__doc__, geometry_error_mm=error, target_incident_faces=len(star),
                  rows=rows, checks=checks, passed=all(checks.values()),
                  state_and_friction_transfer_verified=False, force_or_energy_verified=False,
                  contact_guard_after_insertion_verified=False,
                  time_advanced=False, physical_solution_verified=False)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.output / 'probe_refinement_location.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__ == '__main__':
    main()
