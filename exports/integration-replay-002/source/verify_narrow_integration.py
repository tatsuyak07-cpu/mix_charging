"""Manufactured 1-D strip integration; not a surface/volume FEM algorithm.

The known quadratic separation supplies an exact interval lower bound. This
bound is a verification fixture, not a bound for arbitrary moving FE surfaces.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from packing_fem.barrier_contact import barrier

G8,W8=leggauss(8)
G16,W16=leggauss(16)
THICKNESS=.02


def adaptive(function,left,right,tolerance,budget,depth=0):
    """Nested order comparison with bounded recursion and explicit failure."""
    center=(left+right)/2; radius=(right-left)/2
    coarse=radius*np.array([function(center+radius*s) for s in G8]).T@W8
    fine=radius*np.array([function(center+radius*s) for s in G16]).T@W16
    if np.all(abs(fine-coarse)<=tolerance*abs(fine)+budget):
        return fine,24,depth
    if depth>=24:raise ValueError('Adaptive quadrature recursion limit')
    a,na,da=adaptive(function,left,center,tolerance,budget/2,depth+1)
    b,nb,db=adaptive(function,center,right,tolerance,budget/2,depth+1)
    return a+b,24+na+nb,max(da,db)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=False)
    rows=[];cases=[]
    for width in (.02,.002,.0002):
        for center in (.37,.501,.529):
            for minimum in (.005,.0005):
                curvature=(THICKNESS-minimum)/(width/2)**2
                gap=lambda s:minimum+curvature*(s-center)**2
                def value(s):
                    energy,force,_=barrier(gap(s),20.,THICKNESS)
                    return np.array([energy,force])
                # Exact activation interval from the known quadratic bound.
                left,right=center-width/2,center+width/2
                reference=np.array([quad(lambda s:float(value(s)[j]),left,right,
                                         epsabs=1e-14,epsrel=1e-12,points=[center])[0] for j in (0,1)])
                name=f'w{width}-c{center}-g{minimum}'
                errors=[]
                for tolerance in (1e-5,1e-6):
                    estimate,count,depth=adaptive(value,left,right,tolerance,np.array([1e-18,1e-16]))
                    error=abs(estimate-reference)/abs(reference);errors.append(float(max(error)))
                    rows.append(dict(case=name,method='bound_adaptive',resolution=tolerance,
                                     width_mm=width,center_mm=center,minimum_gap_mm=minimum,
                                     energy_Nmm=float(estimate[0]),force_N=float(estimate[1]),
                                     energy_relative_error=float(error[0]),force_relative_error=float(error[1]),
                                     evaluations=count,maximum_depth=depth))
                missed=[]
                for n in (16,32,64,128):
                    estimate=sum((value((i+.5)/n) for i in range(n)),np.zeros(2))/n
                    error=abs(estimate-reference)/abs(reference)
                    missed.append(bool(estimate[1]==0.))
                    rows.append(dict(case=name,method='uniform_midpoint',resolution=n,
                                     width_mm=width,center_mm=center,minimum_gap_mm=minimum,
                                     energy_Nmm=float(estimate[0]),force_N=float(estimate[1]),
                                     energy_relative_error=float(error[0]),force_relative_error=float(error[1]),
                                     evaluations=n,maximum_depth=0))
                cases.append(dict(name=name,width_mm=width,center_mm=center,minimum_gap_mm=minimum,
                                  reference_energy_Nmm=float(reference[0]),reference_force_N=float(reference[1]),
                                  adaptive_maximum_relative_error=max(errors),
                                  uniform_missed_resolutions=[n for n,m in zip((16,32,64,128),missed) if m]))
    # Same normalized profile: the exact force/energy scale with strip width.
    scaling_error=0.
    for center in (.37,.501,.529):
        for minimum in (.005,.0005):
            group=[c for c in cases if c['center_mm']==center and c['minimum_gap_mm']==minimum]
            for key in ('reference_energy_Nmm','reference_force_N'):
                values=[c[key]/c['width_mm'] for c in group]
                scaling_error=max(scaling_error,(max(values)-min(values))/min(values))
    checks=dict(adaptive_reference_agreement=max(c['adaptive_maximum_relative_error'] for c in cases)<1e-6,
                exact_width_scaling=scaling_error<1e-9,
                uniform_rule_negative_control=any(c['uniform_missed_resolutions'] for c in cases))
    result=dict(scope=__doc__,passed=all(checks.values()),checks=checks,cases=cases,
                width_scaling_relative_error=scaling_error,
                limits=['Exact separation bounds are provided by a manufactured quadratic gap.',
                        'No general triangle search, friction history transfer, contact force distribution, or pressure solver coupling.',
                        'A posteriori order comparison alone does not guarantee detection of an unsampled contact patch.',
                        'Reference shares the existing barrier law; it verifies integration, not the law.'])
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (root/'integration.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for file in (Path(__file__),Path('packing_fem/barrier_contact.py')):
        (root/file.name).write_bytes(file.read_bytes())
    (root/'manifest.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in root.iterdir() if f.is_file()},indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],checks=checks,cases=len(cases),maximum_adaptive_relative_error=max(c['adaptive_maximum_relative_error'] for c in cases),width_scaling_relative_error=scaling_error)),flush=True)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
