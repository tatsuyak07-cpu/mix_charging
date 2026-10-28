"""Increase nominal compressive load on the verified periodic test geometry."""
import argparse,hashlib,json,os,shutil
from pathlib import Path
import numpy as np
from packing_fem.incompatible import write_model,run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('exports/periodic-modes-001/C3D8I_n40'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--target-mpa',type=float,default=.1)
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--timeout',type=float,default=1200)
    a=p.parse_args()
    if a.target_mpa<=0 or a.steps<1:raise ValueError('Positive load and step count required')
    a.output.mkdir(parents=True,exist_ok=False);os.environ['OMP_NUM_THREADS']='1'
    c=json.loads((a.source/'config.json').read_text())
    c.update(control='force',target_mpa=a.target_mpa,compression_steps=a.steps,element_type='C3D8I')
    for name in ('sintered-geometry.npz','estimate.json'):shutil.copyfile(a.source/name,a.output/name)
    (a.output/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    with np.load(a.output/'sintered-geometry.npz') as g:points,elements=g['points'],g['elements']
    write_model(a.output,c,points,elements,np.rint(points/c['mesh_mm']).astype(int),json.loads((a.output/'estimate.json').read_text()),True)
    (a.output/'provenance.json').write_text(json.dumps({'source':str(a.source),'geometry_sha256':hashlib.sha256((a.output/'sintered-geometry.npz').read_bytes()).hexdigest(),'input_sha256':hashlib.sha256((a.output/'plastic.inp').read_bytes()).hexdigest(),'stress_definition':'nominal axial compressive stress; Cauchy stress measured separately','scope':'25 mm exploratory cell, no self-contact, uncalibrated material, mesh convergence unverified'},indent=2)+'\n')
    try:print(json.dumps(run(a.output,a.timeout)),flush=True)
    except Exception as exc:
        (a.output/'analysis-error.json').write_text(json.dumps({'error':repr(exc),'accepted_result':False},indent=2)+'\n')
        raise

if __name__=='__main__':main()
