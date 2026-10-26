"""CalculiX large-displacement, elastic solid FEM and frictional face contact.

Periodic images share nodal DOFs up to the box translation. Their stiffness
and mass are partitioned across copies so duplicated bodies do not stiffen
the physical cell. Contact pairs themselves are counted exactly once.
"""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .geometry import candidate_pairs, sphere_volume, tet_volumes


def validate_config(c):
    if c.get('model_type','unbonded')!='unbonded':
        raise ValueError('Sintered solids require the sintered backend, not frictional particle contact')
    positive = ('length_mm', 'diameter_mm', 'young_mpa', 'target_mpa',
                'mesh_mm', 'candidate_mm', 'density_tonne_mm3',
                'controller_mass_tonne', 'ramp_s', 'hold_s', 'increment_s',
                'penalty_mpa_mm', 'stress_tolerance', 'energy_tolerance',
                'overlap_tolerance_mm', 'max_dofs')
    for key in positive:
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not 0 < c['porosity'] < 1 or not -1 < c['poisson'] < .5:
        raise ValueError('Invalid porosity or Poisson ratio')
    if not np.isfinite(c['friction']) or c['friction'] < 0:
        raise ValueError('Invalid friction')
    if not np.isfinite(c['damping_per_s']) or c['damping_per_s'] < 0:
        raise ValueError('Invalid damping')
    if c['candidate_mm'] < c['diameter_mm']:
        raise ValueError('Candidate radius must include initial contact')
    if c['increment_s'] > min(c['ramp_s'], c['hold_s']):
        raise ValueError('Increment must not exceed a step duration')
    if not isinstance(c.get('max_increments',20000),int) or c.get('max_increments',20000)<1:
        raise ValueError('max_increments must be a positive integer')


def plan(c, centers, points, tets):
    pairs = candidate_pairs(centers, [c['length_mm']]*3, c['candidate_mm'])
    keys = [(i, (0, 0, 0)) for i in range(len(centers))]
    keys += sorted({(j, s) for _, j, s in pairs if s != (0, 0, 0)})
    n = len(centers)
    mesh_volume = float(tet_volumes(points, tets).sum())
    return pairs, keys, {
        'particles': n, 'periodic_copies': len(keys)-n,
        'tetrahedra': len(keys)*len(tets),
        'nodes': len(keys)*len(points)+15,
        'upper_bound_dofs': 3*(len(keys)*len(points)+15),
        'analytic_initial_porosity': 1-n*sphere_volume(c['diameter_mm'])/c['length_mm']**3,
        'mesh_initial_porosity': 1-n*mesh_volume/c['length_mm']**3,
        'sphere_mesh_volume_relative_error': mesh_volume/sphere_volume(c['diameter_mm'])-1,
        'candidate_contacts': len(pairs),
    }


def write_model(directory, c, centers, points, tets, surface, allow_large=False):
    pairs, keys, estimate = plan(c, centers, points, tets)
    if estimate['upper_bound_dofs'] > c['max_dofs'] and not allow_large:
        raise ValueError(f"Model exceeds DOF budget: {estimate}. Use --allow-large explicitly.")
    directory = Path(directory)
    nn, ne = len(points), len(tets)
    copy_count = Counter(i for i, _ in keys)
    key_index = {key: k for k, key in enumerate(keys)}
    rng = np.random.default_rng(c['seed']+1)
    # Random proper rotations avoid identically oriented surface facets.
    rotations = []
    for _ in centers:
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        q[:, 0] *= np.linalg.det(q)
        rotations.append(q)
    physical_points = np.concatenate([points @ r.T+x for r, x in zip(rotations, centers)])
    all_points = np.concatenate([points @ rotations[i].T + centers[i] + np.array(s)*c['length_mm']
                                 for i, s in keys])
    all_tets = np.concatenate([tets+k*nn for k in range(len(keys))])
    macros = np.arange(len(all_points)+1, len(all_points)+4)
    lines = ['*HEADING', 'Periodic nonbonded spheres; mm N MPa tonne s; no cylinder', '*NODE']
    for k, xyz in enumerate(all_points, 1):
        lines.append(f'{k},'+','.join(f'{v:.12g}' for v in xyz))
    for node in macros:
        lines.append(f'{node},0,0,0')
    controller_nodes = []
    controller_elements = []
    next_node, next_elem = int(macros[-1])+1, len(all_tets)+1
    for axis in range(3):
        ids = np.arange(next_node, next_node+4)
        xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
        xyz += c['length_mm']+20+axis*3
        for node, p in zip(ids, xyz):
            lines.append(f'{node},'+','.join(map(str, p)))
        controller_nodes.append(ids)
        controller_elements.append((next_elem, ids))
        next_node += 4
        next_elem += 1
    for k, (i, _) in enumerate(keys):
        lines.append(f'*ELEMENT,TYPE=C3D{tets.shape[1]},ELSET=B{k}')
        for e, conn in enumerate(tets+k*nn+1, k*ne+1):
            lines.append(f'{e},'+','.join(map(str, conn)))
        lines += [f'*SOLID SECTION,ELSET=B{k},MATERIAL=M{copy_count[i]}']
        lines += [f'*SURFACE,NAME=S{k},TYPE=ELEMENT']
        lines += [f'{e+k*ne+1},S{f}' for e, f in surface]
    for multiplicity in sorted(set(copy_count.values())):
        lines += [f'*MATERIAL,NAME=M{multiplicity}', '*ELASTIC',
                  f"{c['young_mpa']/multiplicity:.12g},{c['poisson']}",
                  '*DENSITY', f"{c['density_tonne_mm3']/multiplicity:.12g}"]
    for axis, (e, nodes) in enumerate(controller_elements):
        lines += [f'*ELEMENT,TYPE=C3D4,ELSET=CTRL{axis}', f'{e},'+','.join(map(str,nodes)),
                  f'*SOLID SECTION,ELSET=CTRL{axis},MATERIAL=CONTROL']
    # Controller tetrahedra translate rigidly: zero elastic strain, finite
    # artificial inertial mass. These are numerical stress-control variables.
    lines += ['*MATERIAL,NAME=CONTROL', '*ELASTIC', '1,0.3', '*DENSITY',
              f"{6*c['controller_mass_tonne']:.12g}"]
    lines += ['*ELSET,ELSET=SOLIDS,GENERATE', f'1,{len(all_tets)},1',
              '*ELSET,ELSET=CONTROLLERS', ','.join(str(e) for e,_ in controller_elements),
              '*ELSET,ELSET=CORE,GENERATE', f'1,{len(centers)*ne},1',
              '*NSET,NSET=CORENODES,GENERATE', f'1,{len(centers)*nn},1',
              '*NSET,NSET=MACRO', ','.join(map(str,macros))]
    lines += ['*EQUATION']
    for k, (i, shift) in enumerate(keys):
        if shift == (0,0,0):
            continue
        for a in range(nn):
            for axis in range(3):
                terms = [(k*nn+a+1,axis+1,1),(i*nn+a+1,axis+1,-1)]
                if shift[axis]:
                    terms.append((int(macros[axis]),axis+1,-shift[axis]))
                lines += [str(len(terms)), ','.join(str(v) for term in terms for v in term)]
    for axis, ids in enumerate(controller_nodes):
        for node in ids:
            lines += ['2', f'{node},{axis+1},1,{macros[axis]},{axis+1},-1']
    lines += ['*BOUNDARY']
    for axis, macro in enumerate(macros):
        for component in range(3):
            if component != axis:
                lines.append(f'{macro},{component+1},{component+1},0')
                for node in controller_nodes[axis]:
                    lines.append(f'{node},{component+1},{component+1},0')
    if not pairs:
        raise ValueError('No candidate contacts: enlarge candidate_mm')
    lines += ['*SURFACE INTERACTION,NAME=GRAIN', '*SURFACE BEHAVIOR,PRESSURE-OVERCLOSURE=LINEAR',
              str(c['penalty_mpa_mm']), '*FRICTION',
              f"{c['friction']},{c['penalty_mpa_mm']/10}",
              '*CONTACT PAIR,INTERACTION=GRAIN,TYPE=SURFACE TO SURFACE']
    for i,j,shift in pairs:
        lines.append(f'S{key_index[(i,(0,0,0))]},S{key_index[(j,shift)]}')
    lines += [f"*DAMPING,ALPHA={c['damping_per_s']},BETA=0"]
    (directory/'model.inp').write_text('\n'.join(lines)+'\n')
    np.savez_compressed(directory/'geometry.npz', points=all_points, tets=all_tets,
                        physical_points=physical_points, template_tets=tets,
                        centers=centers, macros=macros, copy_keys=np.array([(i,*s) for i,s in keys]),
                        pairs=np.array([(i,j,*s) for i,j,s in pairs]))
    metadata = {'config':c, 'estimate':estimate, 'nodes_per_sphere':nn,
                'controller_energy_output': True,
                'elements_per_sphere':ne, 'macro_nodes':macros.tolist(),
                'notes':['Density and controller masses are numerical relaxation parameters, not measured material density.',
                         'Stress-controlled orthogonal periodic box; shear strain fixed, shear stress checked.',
                         'Analytic sphere and curved/linear mesh porosities reported separately.']}
    (directory/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


def dynamic_steps(c, force, label, first=False, previous_force=None):
    """Continuous ramp/hold in ONE dynamic step, preserving contact state.

    Starting a second dynamic step can reset the contact-controlled increment
    and introduce a numerical impulse even under unchanged applied load.
    """
    dt = c['increment_s']
    duration=c['ramp_s']+c['hold_s']
    start=0 if first else (previous_force/force if previous_force is not None else 1)
    out = [f'*AMPLITUDE,NAME=RAMP{label}', f"0,{start},{c['ramp_s']},1",f'{duration},1',
           f"*STEP,NLGEOM,INC={c.get('max_increments',20000)}", '*DYNAMIC,ALPHA=-0.1',
           f'{dt},{duration},{dt*1e-5},{dt}', f'*CLOAD,AMPLITUDE=RAMP{label}',
           f"__MACRO_Z__,3,{-force:.12g}",
           '*NODE PRINT,NSET=MACRO,FREQUENCY=10', 'U',
                '*NODE PRINT,NSET=CORENODES', 'U',
                '*EL PRINT,ELSET=SOLIDS', 'S,EVOL',
                '*EL PRINT,ELSET=SOLIDS,TOTALS=ONLY', 'ELSE,ELKE',
                '*EL PRINT,ELSET=CONTROLLERS,TOTALS=ONLY', 'ELKE',
                '*EL PRINT,ELSET=CORE,TOTALS=ONLY', 'EVOL',
                '*CONTACT PRINT', 'CDIS,CSTR',
                '*CONTACT PRINT,TOTALS=ONLY', 'CELS',
                '*NODE FILE,NSET=CORENODES', 'U,V', '*END STEP']
    return '\n'.join(out)+'\n'
