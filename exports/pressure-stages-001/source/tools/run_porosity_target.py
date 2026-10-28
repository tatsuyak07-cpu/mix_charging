"""Compress the periodic sintered trial toward a target porosity.

A displacement envelope is a numerical search bound, not an assumed mapping
from axial compression to porosity. A separate monitor stops on observed target
or on existing audit failure. No unobserved porosity is imputed.
"""
import argparse,hashlib,json,os
from pathlib import Path
from packing_fem import sintered
from packing_fem.incompatible import write_model,run


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--timeout',type=float,default=1200)
    a=p.parse_args();c=json.loads(a.config.read_text())
    if not 0<c['target_final_porosity']<c['porosity']<1:raise ValueError('Require 0 < final porosity < initial porosity < 1')
    if c.get('control')!='displacement':raise ValueError('Target search uses displacement control')
    a.output.mkdir(parents=True,exist_ok=False);os.environ['OMP_NUM_THREADS']='1'
    (a.output/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    try:
        points,elements,indices,estimate=sintered.prepare(a.output,c,True)
        write_model(a.output,c,points,elements,indices,estimate,True)
        (a.output/'provenance.json').write_text(json.dumps({'requested_initial_porosity':c['porosity'],'actual_initial_porosity':estimate['initial_porosity'],'target_final_porosity':c['target_final_porosity'],'geometry_sha256':hashlib.sha256((a.output/'sintered-geometry.npz').read_bytes()).hexdigest(),'input_sha256':hashlib.sha256((a.output/'plastic.inp').read_bytes()).hexdigest(),'scope':'Small periodic explicit sintered-particle trial, no self-contact, uncalibrated material; not the original 100 mm RVE'},indent=2)+'\n')
        print(json.dumps(run(a.output,a.timeout)),flush=True)
    except Exception as exc:
        (a.output/'analysis-error.json').write_text(json.dumps({'error':repr(exc),'accepted_result':False},indent=2)+'\n');raise

if __name__=='__main__':main()
