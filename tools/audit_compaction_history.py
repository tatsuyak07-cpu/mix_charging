"""Audit accepted FEM volume history, equilibrium criteria and fixed-cell kinematics."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from packing_fem.explicit_material import geometry,response
from packing_fem.explicit_periodic import Periodic,deformation


def audit(run,out):
    run=Path(run);out=Path(out)
    if out.exists():raise FileExistsError(out)
    c=json.loads((run/'config.json').read_text())
    with (run/'history.csv').open() as stream:history={int(row['step']):row for row in csv.DictReader(stream)}
    rows=[];reference_volume=None
    for path in sorted(run.glob('state-*.npz')):
        step=int(path.stem.split('-')[-1]);record=history.get(step)
        with np.load(path) as stream:d=dict(stream)
        p=Periodic(d['X'],c['length_mm']);H=deformation(d['q'][-6:]);cell_volume=p.L**3*np.linalg.det(H)
        volumes=geometry(d['x'],d['t'])[0];solid_volume=volumes.sum()
        J=response(d['b'],d['ep'],c)[2]
        local=np.bincount(d['parent'],weights=d['V0']*J,minlength=len(d['t']))
        material_volume=float(d['V0'].sum())
        if reference_volume is None:reference_volume=material_volume
        phi=float(1-solid_volume/cell_volume)
        row=dict(step=step,porosity=phi,solid_volume_mm3=float(solid_volume),reference_material_volume_mm3=material_volume,
                 solid_volume_to_initial_ratio=float(solid_volume/reference_volume),cell_volume_mm3=float(cell_volume),
                 minimum_element_volume_mm3=float(volumes.min()),
                 reference_material_relative_error=abs(material_volume/reference_volume-1),
                 quadrature_geometry_relative_error=float(np.max(abs(local-volumes)/volumes)),
                 saved_position_error_mm=float(np.max(abs(d['x']-p.positions(d['q'])))),
                 csv_porosity_error=0. if record is None else abs(phi-float(record['porosity'])))
        passed=(row['minimum_element_volume_mm3']>0 and row['reference_material_relative_error']<1e-7
                and row['quadrature_geometry_relative_error']<1e-7 and row['saved_position_error_mm']<1e-8
                and row['csv_porosity_error']<1e-10 and cell_volume>0 and 0<=phi<=1)
        if c.get('transverse_boundary')=='fixed_cell':
            row['macro_fixed_cell_error']=float(np.max(abs(H-np.diag([1.,1.,H[2,2]]))))
            passed=passed and row['macro_fixed_cell_error']<1e-12
        if record is not None:
            row['free_residual_N']=float(record['free_residual_N']);row['macro_target_error']=float(record['macro_target_error'])
            passed=passed and row['free_residual_N']<c['implicit_force_tolerance_N'] and row['macro_target_error']<c['implicit_macro_tolerance']
        row['passed']=bool(passed);rows.append(row)
    missing=sorted(set(history)-{row['step'] for row in rows})
    report=dict(passed=bool(rows and not missing and all(row['passed'] for row in rows)),missing_steps=missing,
                interpretation='Reference material volume is conserved. Current solid volume includes elastic compression; exact incompressibility is not imposed.',
                spatial_and_increment_convergence_verified=False,states=rows)
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=report['passed'],states=len(rows),last=rows[-1] if rows else None)),flush=True)
    return report['passed']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    if not audit(args.run,args.out):raise SystemExit(1)
