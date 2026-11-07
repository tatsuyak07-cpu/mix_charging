"""Generate a reproducible random periodic sinter graph and exact input files."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from packing_fem.random_packing import generate


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--seed-start',type=int);p.add_argument('--seed-budget',type=int,default=20);a=p.parse_args()
    c=json.loads(Path(a.config).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    (out/'requested-config.json').write_text(json.dumps(c,indent=2)+'\n')
    start=c['packing_seed'] if a.seed_start is None else a.seed_start
    for seed in range(start,start+a.seed_budget):
        c['packing_seed']=seed;directory=out/f'seed-{seed}'
        try:packing=generate(c,directory)
        except ValueError as exc:
            directory.mkdir(exist_ok=True);(directory/'rejected.json').write_text(json.dumps(dict(seed=seed,reason=str(exc)),indent=2));print(seed,str(exc),flush=True);continue
        np.savez_compressed(out/'packing.npz',**{k:v for k,v in packing.items() if k!='metadata'})
        (out/'packing.json').write_text(json.dumps(packing['metadata'],indent=2)+'\n');c['length_mm']=packing['length_mm']
        (out/'resolved-config.json').write_text(json.dumps(c,indent=2)+'\n')
        with (out/'centers.csv').open('w',newline='') as stream:
            w=csv.writer(stream);w.writerow(['particle','x_mm','y_mm','z_mm'])
            for i,center in enumerate(packing['fractional_centers']*packing['length_mm']):w.writerow([i,*center])
        with (out/'bonds.csv').open('w',newline='') as stream:
            w=csv.writer(stream);w.writerow(['particle_i','particle_j','image_x','image_y','image_z','surface_gap_mm'])
            for (i,j),shift in zip(packing['bonds'],packing['bond_images']):
                gap=np.linalg.norm(packing['fractional_centers'][j]+shift-packing['fractional_centers'][i])*packing['length_mm']-c['diameter_mm']
                w.writerow([i,j,*shift,gap])
        print(json.dumps(packing['metadata']),flush=True);return
    raise RuntimeError('No geometrically valid packing within the declared seed budget')

if __name__=='__main__':main()
