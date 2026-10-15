"""Compare two actual linear elastic FEM solves on the identical saved mesh."""
import argparse
import hashlib
import json
from pathlib import Path
import meshio
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference',type=Path);p.add_argument('run',type=Path)
    a=p.parse_args()
    old=json.loads((a.reference/'sintered-metadata.json').read_text())
    new=json.loads((a.run/'sintered-metadata.json').read_text())
    previous=json.loads((a.reference/'sintered-summary.json').read_text())
    current=json.loads((a.run/'sintered-summary.json').read_text())
    digests=[hashlib.sha256((r/'sintered-geometry.npz').read_bytes()).hexdigest() for r in [a.reference,a.run]]
    ratio=old['config']['young_mpa']/new['config']['young_mpa']
    u0=meshio.read(a.reference/'sintered-deformed.vtu').point_data['displacement_mm']
    u1=meshio.read(a.run/'sintered-deformed.vtu').point_data['displacement_mm']
    relative=float(np.max(np.abs(u1-ratio*u0))/np.max(np.abs(u1)))
    with np.load(a.run/'sintered-geometry.npz') as g:points=g['points']
    h=new['config']['mesh_mm'];L=new['config']['length_mm'];n=int(round(L/h))
    lookup=np.full((n+1,)*3,-1,dtype=np.int64)
    lookup[tuple(np.rint(points/h).astype(int).T)]=np.arange(len(points))
    E=np.array(current['macro_strain']);periodic_error=0
    for axis in range(3):
        low=np.take(lookup,0,axis=axis);high=np.take(lookup,n,axis=axis)
        valid=(low>=0)&(high>=0)
        periodic_error=max(periodic_error,float(np.max(np.abs(u1[high[valid]]-u1[low[valid]]-L*E[:,axis]))))
    config_match=all(old['config'][k]==v for k,v in new['config'].items() if k!='young_mpa')
    # Isotropic elastic trace(strain_solid)=(1-2nu)*trace(stress_solid)/E.
    phi0=current['initial_porosity'];sigma=np.array(current['mean_stress_mpa'])
    first_order=(1-phi0)*np.trace(E)-(1-2*new['config']['poisson'])*sigma[:3].sum()/new['config']['young_mpa']
    volume_change=current['final_porosity']-phi0
    report={'identical_geometry':digests[0]==digests[1],'only_modulus_changed':config_match,
            'expected_displacement_ratio':ratio,'displacement_scaling_relative_max_error':relative,
            'periodic_displacement_max_residual_mm':periodic_error,
            'axial_strain_ratio':current['axial_compression_strain']/previous['axial_compression_strain'],
            'first_order_porosity_change':first_order,'integrated_porosity_change':volume_change,
            'porosity_change_difference':volume_change-first_order,
            'checks':{'identical_geometry':digests[0]==digests[1],'only_modulus_changed':config_match,
                      'linear_modulus_scaling':relative<2e-5,'periodic_displacements':periodic_error<1e-6}}
    out=a.run/'modulus-comparison.json'
    with out.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
    if not all(report['checks'].values()):raise SystemExit(2)

if __name__=='__main__':main()
