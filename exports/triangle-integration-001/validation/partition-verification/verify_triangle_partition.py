"""Reference-triangle partition invariance of normal contact virtual work."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from packing_fem.triangle_quadrature import integrate
from packing_fem.contact_audit import plain
from tools.verify_triangle_integration import fixture,reference


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=False);cases=[]
    for kind in ('edge','corner'):
        source,target=fixture(kind,.002,.0005,.37)
        baseline=integrate(source,target,.5,relative_tolerance=5e-4)
        exact=reference(kind,.002,.0005)
        for fraction in (.2,.5,.8):
            for tolerance in (1e-3,5e-4):
                name=f'{kind}-f{fraction}-tol{tolerance}';folder=root/name;folder.mkdir()
                unit=np.eye(3);mid=(1-fraction)*unit[1]+fraction*unit[2]
                children=[np.array([unit[0],unit[1],mid]),np.array([unit[0],mid,unit[2]])]
                force=np.zeros((6,3));energy=0.;points=[];weights=[]
                for B in children:
                    response=integrate(B@source,target,.5*abs(np.linalg.det(B)),relative_tolerance=tolerance)
                    force[:3]+=B.T@response['force'][:3];force[3:]+=response['force'][3:]
                    energy+=response['energy'];points.extend(response['barycentric_points']@B);weights.extend(response['area_weights'])
                force_error=float(np.max(abs(force-baseline['force']))/exact[1])
                energy_error=abs(energy-baseline['energy'])/exact[0]
                exact_error=max(abs(energy-exact[0])/exact[0],abs(force[:3,2].sum()-exact[1])/exact[1])
                checks=dict(partition_force=force_error<2e-3,partition_energy=energy_error<2e-3,
                            independent_reference=exact_error<2e-3,action_reaction=float(np.linalg.norm(force.sum(0)))/exact[1]<1e-10)
                row=plain(dict(case=name,kind=kind,fraction=fraction,tolerance=tolerance,energy_Nmm=energy,
                               source_normal_force_N=float(force[:3,2].sum()),partition_force_relative_error=force_error,
                               partition_energy_relative_error=float(energy_error),independent_relative_error=float(exact_error),
                               quadrature_points=len(weights),checks=checks,passed=all(checks.values())))
                np.savez_compressed(folder/'quadrature.npz',source=source,target=target,barycentric_points=points,area_weights=weights,nodal_force=force)
                (folder/'summary.json').write_text(json.dumps(row,indent=2)+'\n');cases.append(row);print(json.dumps(row),flush=True)
    summary=dict(scope=__doc__,passed=all(r['passed'] for r in cases),cases=cases,
                 limits=['Frictionless prescribed pair; no friction history or coupled solver remeshing.',
                         'Child contact forces pulled back by transpose of exact linear geometry interpolation.',
                         'Adaptive quadrature sets may differ; only reference-integral accuracy is screened.'])
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (root/'verify_triangle_partition.py').write_bytes(Path(__file__).read_bytes())
    (root/'manifest.json').write_text(json.dumps({str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in root.rglob('*') if f.is_file()},indent=2)+'\n')
    print(json.dumps(dict(passed=summary['passed'],cases=len(cases))),flush=True)
    if not summary['passed']:raise SystemExit(1)


if __name__=='__main__':main()
