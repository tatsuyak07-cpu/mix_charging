"""Identical material points must have identical density at any batch size."""
import argparse,json
from pathlib import Path
import numpy as np
from packing_fem.contact_atlas import density,scales
from packing_fem.surface_patch import integrate


def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    f=json.loads(a.summary.read_text())['failure_feature'];poses=f['poses']+f['controls'];rules=[];rows=[]
    for pose in poses:
        rules.append(integrate(np.asarray(pose['source']),np.asarray(pose['targets']),f['reference_area_mm2'],kn=f['kn'],thickness=f['thickness'],tolerance=f['tolerance']/16,vertex_map=np.asarray(pose['mapping']),groups=pose['groups'],pointwise=True,error_budget='normalized',absolute_energy_budget=f['energy_budget']/16,absolute_force_budget=f['force_budget']/16))
    union=np.unique(np.concatenate([r['bary'] for r in rules]),axis=0);lookup={tuple(p):i for i,p in enumerate(union)}
    for i,(pose,rule) in enumerate(zip(poses,rules)):
        separate=density(rule['bary'],pose,f['kn'],f['thickness']);pooled=density(union,pose,f['kn'],f['thickness']);indices=np.array([lookup[tuple(p)] for p in rule['bary']]);difference=pooled[indices]-separate
        weighted=rule['weights']@difference;norm=scales(rule['value'],f['energy_budget'],f['force_budget'],f['tolerance'])
        row=dict(pose=i,points=len(indices),pooled_points=len(union),max_density_difference=float(abs(difference).max()),integral_normalized_difference=float((abs(weighted)/norm).max()))
        rows.append(row);print(json.dumps(row),flush=True)
    (a.output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
