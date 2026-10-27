"""Finite triangle-pair integration against independent 1-D reductions.

Prescribed frictionless pairs only; no periodic search or bulk pressure solve.
"""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from packing_fem.barrier_contact import barrier
from packing_fem.triangle_quadrature import integrate,density,density_batch
from packing_fem.contact_audit import plain


def fixture(kind,width,minimum,phase=0.):
    xy=np.array([[0.,0.],[1.,0.],[0.,1.]])
    coordinate=xy[:,0] if kind=='edge' else xy.sum(1)
    source=np.c_[xy,minimum+(.02-minimum)/width*coordinate]
    target=np.array([[-2.,-2.,0.],[4.,-2.,0.],[-2.,4.,0.]])
    rotation=np.array([[np.cos(phase),-np.sin(phase),0.],[np.sin(phase),np.cos(phase),0.],[0.,0.,1.]])
    translation=np.array([.173,-.219,.137])
    return source@rotation.T+translation,target@rotation.T+translation


def reference(kind,width,minimum):
    def value(u):
        energy,force,_=barrier(minimum+(.02-minimum)*u/width,20.,.02)
        measure=1-u if kind=='edge' else u
        bary=np.array([(1-u)/2,u,(1-u)/2]) if kind=='edge' else np.array([1-u,u/2,u/2])
        return np.r_[energy,force,force*bary]*measure
    return np.array([quad(lambda u:float(value(u)[i]),0,width,epsabs=1e-19,epsrel=1e-11)[0] for i in range(5)])


def frozen_energy(points,weights,source,target):
    return float(weights@density_batch(points,source,target,20.,.02)[0][:,0])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=False)
    rows=[]
    for kind in ('edge','corner'):
        for width in (.02,.002,.0002):
            for minimum in (.005,.0005):
                exact=reference(kind,width,minimum)
                for tolerance in (1e-3,5e-4):
                    name=f'{kind}-w{width}-g{minimum}-tol{tolerance}'
                    folder=root/name;folder.mkdir()
                    phase=.37 if minimum==.0005 else 0.
                    source,target=fixture(kind,width,minimum,phase)
                    response=integrate(source,target,.5,relative_tolerance=tolerance)
                    points=response['barycentric_points'];weights=response['area_weights'];forces=response['force']
                    actual=np.r_[response['energy'],forces[:3,2].sum(),forces[:3,2]]
                    scale=np.r_[exact[:2],np.full(3,exact[1])]
                    error=abs(actual-exact)/scale
                    balance=float(np.linalg.norm(forces.sum(0)))
                    moment=float(np.linalg.norm(np.cross(np.r_[source,target],forces).sum(0)))
                    # Nodal force is the derivative of the SAME frozen
                    # quadrature potential, independent of adaptation errors.
                    direction=np.array([[.1,-.3,.7],[-.2,.4,-.3],[.5,.1,.2],[-.3,.2,.1],[.2,-.1,-.4],[.1,.3,.6]])
                    eps=1e-7*minimum
                    plus=frozen_energy(points,weights,source+eps*direction[:3],target+eps*direction[3:])
                    minus=frozen_energy(points,weights,source-eps*direction[:3],target-eps*direction[3:])
                    work=float(np.sum(forces*direction));fd=-(plus-minus)/(2*eps)
                    work_error=abs(fd-work)/exact[1]
                    area_error=abs(float(weights.sum())+response['stats']['pruned_area']-.5)
                    # One centroid point is a negative control: both thin
                    # footprints lie away from it for every tested width.
                    centroid=density(np.ones(3)/3,source,target,20.,.02)[0]
                    checks=dict(independent_integral=float(max(error))<2e-3,
                                action_reaction=balance/exact[1]<1e-10,
                                angular_momentum=moment/exact[1]<1e-9,
                                frozen_potential_gradient=work_error<2e-5,
                                reference_area_partition=area_error<1e-12,
                                centroid_misses_contact=bool(centroid[0]==0 and exact[0]>0),
                                positive_weights=bool(np.all(weights>0)),
                                valid_material_coordinates=bool(np.all(points>=-1e-12) and np.max(abs(points.sum(1)-1))<1e-12))
                    row=plain(dict(case=name,kind=kind,width_mm=width,minimum_gap_mm=minimum,in_plane_rotation_rad=phase,
                                   tolerance=tolerance,energy_Nmm=response['energy'],normal_force_N=float(actual[1]),
                                   reference_energy_Nmm=float(exact[0]),reference_force_N=float(exact[1]),
                                   maximum_integral_relative_error=float(max(error)),energy_relative_error=float(error[0]),
                                   force_relative_error=float(error[1]),work_relative_error=work_error,
                                   reference_area_error_mm2=area_error,quadrature_points=len(weights),
                                   **response['stats'],checks=checks,passed=all(checks.values())))
                    np.savez_compressed(folder/'quadrature.npz',source=source,target=target,barycentric_points=points,area_weights=weights,nodal_force=forces)
                    (folder/'summary.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
                    print(json.dumps(row),flush=True)
    groups=[]
    for kind in ('edge','corner'):
        for width in (.02,.002,.0002):
            for minimum in (.005,.0005):
                a,b=[r for r in rows if r['kind']==kind and r['width_mm']==width and r['minimum_gap_mm']==minimum]
                groups.append(dict(kind=kind,width_mm=width,minimum_gap_mm=minimum,
                                   force_change_relative=abs(a['normal_force_N']-b['normal_force_N'])/b['reference_force_N'],
                                   energy_change_relative=abs(a['energy_Nmm']-b['energy_Nmm'])/b['reference_energy_Nmm']))
    checks=dict(all_cases=all(r['passed'] for r in rows),tolerance_sensitivity=max(max(g['force_change_relative'],g['energy_change_relative']) for g in groups)<2e-3)
    summary=dict(scope=__doc__,passed=all(checks.values()),checks=checks,cases=rows,tolerance_groups=groups,
                 limits=['Frictionless prescribed finite triangle pairs, no bulk material update.',
                         'No periodic image enumeration, friction history or pressure solver coupling.',
                         'AABB bounds may be loose and cause explicit maximum-depth failure.',
                         'Quadrature order comparison and sample/lower-bound check are screening criteria, not rigorous integral error bounds.',
                         'In-plane rotation tested; general 3-D objectivity not established.',
                         'No 0.1 MPa or porosity prediction.'])
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (root/'comparison.csv').open('w') as stream:
        fields=[k for k in rows[0] if k!='checks'];writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in rows)
    for file in (Path(__file__),Path('packing_fem/triangle_quadrature.py')):(root/file.name).write_bytes(file.read_bytes())
    (root/'manifest.json').write_text(json.dumps({str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in root.rglob('*') if f.is_file()},indent=2)+'\n')
    print(json.dumps(dict(passed=summary['passed'],cases=len(rows),checks=checks)),flush=True)
    if not summary['passed']:raise SystemExit(1)


if __name__=='__main__':main()
