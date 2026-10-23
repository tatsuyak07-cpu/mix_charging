"""Read accepted CalculiX outputs; failed jobs remain diagnostics only."""
import csv
import json
import re
from pathlib import Path
import numpy as np
from .geometry import candidate_pairs, tet_volumes,tetra_quadrature


def read_dat(path):
    """Yield one output time at a time; do not combine different increments."""
    current_time, snapshot, heading, rows = None, {}, None, []

    def finish():
        if heading is not None:
            snapshot[heading] = np.asarray(rows, dtype=float)

    with Path(path).open() as stream:
        for line in stream:
            match = re.search(r'\btime\s+([\d.EeDd+\-]+)\s*$', line)
            if match:
                finish()
                time = float(match.group(1).replace('D','E'))
                if current_time is not None and time != current_time:
                    yield current_time, snapshot
                    snapshot = {}
                current_time = time
                heading = line[:match.start()].strip().lower()
                rows = []
            elif heading is not None:
                parts = line.split()
                if parts:
                    try:
                        rows.append([float(p.replace('D','E')) for p in parts])
                    except ValueError:
                        pass
    finish()
    if current_time is not None:
        yield current_time, snapshot


def find(snapshot, prefix, set_name=None):
    matches = [v for k,v in snapshot.items()
               if k.startswith(prefix) and (set_name is None or f'for set {set_name.lower()} and' in k)]
    if len(matches) != 1 or matches[0].size == 0:
        raise ValueError(f'Missing/ambiguous current-run output: {prefix}, {set_name}')
    return matches[0]


def frame(time, snapshot, metadata, geometry):
    c = metadata['config']
    macro = find(snapshot, 'displacements', 'MACRO')
    ids = macro[:,0].astype(int)
    lengths = np.array([c['length_mm']+macro[np.where(ids==node)[0][0],axis+1]
                        for axis,node in enumerate(metadata['macro_nodes'])])
    if np.any(lengths <= 0):
        raise ValueError('Periodic cell collapsed or inverted')
    u = find(snapshot, 'displacements', 'CORENODES')
    ncore = len(geometry['physical_points'])
    expected = np.arange(1,ncore+1)
    if len(u) != ncore or not np.array_equal(np.sort(u[:,0].astype(int)),expected):
        raise ValueError('Incomplete displacement output')
    u = u[np.argsort(u[:,0])][:,1:4]
    xyz = geometry['physical_points']+u
    ne = metadata['elements_per_sphere']
    n = metadata['estimate']['particles']
    core_tets = geometry['tets'][:n*ne]
    signed_volume = tet_volumes(xyz,core_tets,signed=True)
    solid = float(signed_volume.sum())
    box = float(lengths.prod())
    # Stress in replicas is scaled by multiplicity; summing all replicas once
    # recovers the physical integrated solid Cauchy stress, not N times it.
    stresses = find(snapshot,'stresses','SOLIDS')
    nq=1 if geometry['tets'].shape[1]==4 else 4
    if len(stresses) != len(geometry['tets'])*nq:
        raise ValueError('Incorrect tetrahedral integration point count')
    stresses = stresses[np.lexsort((stresses[:,1],stresses[:,0]))]
    if not np.array_equal(stresses[:,0].astype(int),np.repeat(np.arange(1,len(geometry['tets'])+1),nq)) or not np.array_equal(
            stresses[:,1].astype(int),np.tile(np.arange(1,nq+1),len(geometry['tets']))):
        raise ValueError('Incomplete stress output')
    # Recover image volumes from the physical grain: periodic images differ
    # only by translation, so their element Jacobians are identical.
    physical_volumes=tetra_quadrature(xyz,core_tets).reshape(n,ne,nq)
    volumes = np.concatenate([physical_volumes[int(key[0])].ravel() for key in geometry['copy_keys']])
    stress = (stresses[:,2:8]*volumes[:,None]).sum(axis=0)/box
    internal = float(find(snapshot,'total internal energy','SOLIDS').ravel()[0])
    kinetic = float(find(snapshot,'total kinetic energy','SOLIDS').ravel()[0])
    controller_kinetic = float(find(snapshot,'total kinetic energy','CONTROLLERS').ravel()[0]) if metadata.get('controller_energy_output') else 0.
    contact_energy = float(find(snapshot,'total contact spring energy').ravel()[0])
    # Penalized contact compliance belongs to the numerical model; include it
    # in the kinetic/elastic diagnostic, but expose it separately.
    elastic = internal+contact_energy
    ratio = (kinetic+controller_kinetic)/max(abs(elastic),1e-30)
    max_overlap = None
    contact_stress = None
    for k,v in snapshot.items():
        if k.startswith('contact stress') and len(v):
            contact_stress=v[:,-3:]
    for k, v in snapshot.items():
        if ('relative contact' in k or 'contact displacements' in k) and len(v):
            # Face contact output may have (element, face, point, normal, t1,t2)
            # or (node, normal, t1,t2): the last three are displacement components.
            # CalculiX 2.20 face contact writes NEGATIVE normal displacement
            # under compressive pressure (verified against p=-k*normal).
            # Use magnitude at compressed points, not an old manual's sign.
            if contact_stress is not None and len(v)==len(contact_stress):
                compressed=contact_stress[:,0]>0
                max_overlap=float(np.max(np.abs(v[compressed,-3]),initial=0))
    unwrapped=geometry['centers']+u.reshape(n,metadata['nodes_per_sphere'],3).mean(axis=1)
    centers = unwrapped % lengths
    layers=max(1,int(np.ceil(c['diameter_mm']/lengths.min())))
    if layers>4:
        raise ValueError('Cell too collapsed for bounded periodic contact audit')
    active = candidate_pairs(centers,lengths,c['diameter_mm']*1.05,layers=layers)
    original = {tuple(map(int,row)) for row in geometry['pairs']}
    # Original core centers may drift across box boundaries; account for the
    # image relabeling instead of falsely reporting a missing contact.
    wraps = np.floor(unwrapped/lengths).astype(int)
    missing = []
    for i,j,s in active:
        shift = np.array(s)+wraps[i]-wraps[j]
        key = (i,j,*map(int,shift))
        if i==j and tuple(shift) < (0,0,0):
            key=(i,j,*map(int,-shift))
        if key not in original:
            missing.append(key)
    target = c['target_mpa']
    status = {
        'stress_target': abs(-stress[2]-target) <= c['stress_tolerance']*target,
        'free_lateral_stress': max(abs(stress[0]),abs(stress[1])) <= c['stress_tolerance']*target,
        'small_shear_stress': max(abs(stress[3:])) <= c['stress_tolerance']*target,
        'small_kinetic_energy': ratio <= c['energy_tolerance'],
        'controller_energy_checked': bool(metadata.get('controller_energy_output')),
        'positive_elements': bool(np.all(signed_volume>0) and np.all(physical_volumes>0)),
        'physical_porosity': 0 < 1-solid/box < 1,
        'complete_contact_candidates': not missing,
        'adjacent_images_sufficient': not any(max(abs(v) for v in key[2:])>1 for key in missing),
        'contact_penetration_checked': max_overlap is not None,
        'small_penetration': max_overlap is not None and max_overlap <= c['overlap_tolerance_mm'],
        'sphere_geometry_resolved': abs(metadata['estimate']['sphere_mesh_volume_relative_error']) <= .01,
        'small_penalty_energy': contact_energy <= .1*max(abs(internal),1e-30),
        'coulomb_bound': contact_stress is not None and bool(np.all(
            np.linalg.norm(contact_stress[:,1:],axis=1) <=
            c['friction']*np.maximum(0,contact_stress[:,0])*(1+1e-5)+1e-6)),
    }
    row = {'time_s':time,'lx_mm':lengths[0],'ly_mm':lengths[1],'lz_mm':lengths[2],
           'axial_strain':1-lengths[2]/c['length_mm'],'porosity':1-solid/box,
           'solid_volume_mm3':solid,'cell_volume_mm3':box,
           'compression_mpa':-stress[2],'sxx_mpa':stress[0],'syy_mpa':stress[1],
           'sxy_mpa':stress[3],'sxz_mpa':stress[4],'syz_mpa':stress[5],
           'kinetic_Nmm':kinetic,'internal_Nmm':internal,'contact_Nmm':contact_energy,
           'controller_kinetic_Nmm':controller_kinetic,
           'kinetic_elastic_ratio':ratio,'max_overlap_mm':max_overlap,
           'missing_candidate_contacts':len(missing)}
    return row,status,xyz


def analyze(directory, job, completed, expected_time, solver_exit_code=None):
    directory=Path(directory)
    for suffix in ('summary.json','history.csv','history.png','last-accepted.vtu'):
        if (directory/f'{job}-{suffix}').exists():
            raise ValueError(f'Refusing to overwrite archived analysis: {job}-{suffix}')
    meta=json.loads((directory/'metadata.json').read_text())
    geometry=np.load(directory/'geometry.npz')
    rows,checks,last_xyz=[],{},None
    history_valid=True
    try:
        for time,snapshot in read_dat(directory/f'{job}.dat'):
            row,checks,last_xyz=frame(time,snapshot,meta,geometry)
            history_valid &= all(checks[key] for key in ('positive_elements','physical_porosity',
                                 'complete_contact_candidates','adjacent_images_sufficient'))
            rows.append(row)
    except (ValueError,OSError) as exc:
        checks['complete_outputs']=False
        error=str(exc)
    else:
        checks['complete_outputs']=bool(rows)
        error=None
    checks['solver_completed']=completed
    checks['history_geometry_and_contacts_valid']=history_valid and bool(rows)
    checks['final_time_reached']=bool(rows) and abs(rows[-1]['time_s']-expected_time)<1e-6*max(expected_time,1)
    # Endpoint kinetic energy can vanish at a turning point. Require a late
    # output window of at least three states and stable cell dimensions.
    hold=meta['config']['hold_s']
    late=[r for r in rows if r['time_s'] >= expected_time-.2*hold]
    checks['stable_late_window']=len(late)>=3 and all(
        r['kinetic_elastic_ratio']<=meta['config']['energy_tolerance'] for r in late)
    if len(late)>=3:
        checks['stable_late_window'] &= all(
            np.ptp([r[f'l{axis}_mm'] for r in late])/meta['config']['length_mm']<1e-4
            for axis in 'xyz')
    result={'status':'diagnostics_only','checks':{k:bool(v) for k,v in checks.items()},
            'last_accepted_increment':rows[-1] if rows else None,'error':error,
            'mesh_and_relaxation_independence':'unverified',
            'full_size_model':meta['estimate']['particles']==6112 and meta['config']['length_mm']==100,
            'full_size_validation':'unverified'}
    result['job']=job
    if solver_exit_code is not None:
        result['solver_exit_code']=solver_exit_code
    if checks and all(checks.values()):
        result['status']='equilibrium_candidate_requires_refinement'
    with (directory/f'{job}-summary.json').open('x') as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if rows:
        with (directory/f'{job}-history.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(9,3.5))
        axes[0].plot([r['axial_strain'] for r in rows],[r['compression_mpa'] for r in rows])
        axes[0].axhline(meta['config']['target_mpa'],color='gray',linestyle='--')
        axes[0].set(xlabel='Axial compression strain',ylabel='Compression stress [MPa]')
        axes[1].plot([r['time_s'] for r in rows],[r['porosity'] for r in rows])
        axes[1].set(xlabel='Numerical relaxation time [s]',ylabel='Mesh porosity')
        fig.suptitle(result['status']);fig.tight_layout()
        fig.savefig(directory/f'{job}-history.png',dpi=160);plt.close(fig)
        import meshio
        n=meta['estimate']['particles'];ne=meta['elements_per_sphere']
        meshio.write(directory/f'{job}-last-accepted.vtu',meshio.Mesh(last_xyz,
                     [('tetra' if geometry['tets'].shape[1]==4 else 'tetra10',geometry['tets'][:n*ne])],
                     point_data={'displacement_mm':last_xyz-geometry['physical_points']}))
    geometry.close()
    return result
