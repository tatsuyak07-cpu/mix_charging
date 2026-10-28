"""Independent near-activation integral and saved-RVE quadrature verification.

No bulk trajectory or friction history is advanced. Decimal densities and a
one-dimensional integral supply the manufactured-plane reference; RVE cases
use a tolerance-refinement study, not an exact reference claim.
"""
import argparse,csv,hashlib,itertools,json,time
from decimal import Decimal,localcontext
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from packing_fem.explicit_periodic import Periodic,deformation
from packing_fem.embedded_contact import EmbeddedContact
from tools import experimental_surface_budget as patch


def errors(value,reference):
    scale=max(float(np.linalg.norm(reference[1:10].reshape(3,3).sum(0))),
              float(np.linalg.norm(reference[10:].reshape(-1,3),axis=1).sum()),1e-16)
    return dict(energy_error=float(abs(value[0]-reference[0])/max(abs(reference[0]),1e-18)),
                force_error=float(abs(value[1:]-reference[1:]).max()/scale),
                physical_energy_relative_error=float(abs(value[0]-reference[0])/abs(reference[0])) if reference[0] else None)


def applied(rule,source,targets,groups,mapping):
    return rule['weights']@patch.grouped_field(rule['bary'],source,targets,groups,10.,.02,mapping,True)[0]


def plane_reference(source):
    """60-digit scalar density, reduced exact triangle moments, scaled quad."""
    with localcontext() as ctx:
        ctx.prec=60;D=Decimal.from_float;h=D(.02);kn=D(10.);z0=D(float(source[0,2]));dz=D(float(source[1,2]))-z0
        width=(h-z0)/dz;depth=(h-z0)/h;es=kn*h*h*depth**3;fs=kn*h*depth**2
        def density(s,kind):
            g=z0+(h-z0)*D(float(s));r=(g/h)**2
            if r>=1:return 0.
            log=r.ln();one=1-r
            if kind==0:return float((-kn*h*h/8*one*one*log)/es)
            return float((-kn*g/4*(2*one*log-one*one/r))/fs)
        w=float(width)
        moments=[lambda x:(1-x)**2/2,lambda x:x*(1-x),lambda x:(1-x)**2/2,
                 lambda x:(2-x)*(1-x)/6-(1-x)**2/12,
                 lambda x:(x+2)*(1-x)/6,lambda x:(1-x)**2/12+2*(1-x)/6]
        reference=np.zeros(19)
        integral,err=quad(lambda s:(1-w*s)*density(s,0),0,1,epsabs=1e-12,epsrel=1e-12)
        reference[0]=w*float(es)*integral;uncertainty=w*float(es)*err
        for i,moment in enumerate(moments):
            integral,_=quad(lambda s:moment(w*s)*density(s,1),0,1,epsabs=1e-12,epsrel=1e-12)
            reference[3+3*i]=(1 if i<3 else -1)*w*float(fs)*integral
        return reference,dict(active_width=w,quad_energy_error_estimate_Nmm=uncertainty,decimal_digits=60)


def manufactured(root):
    rows=[];targets=np.array([[[-2.,-2.,0.],[4.,-2.,0.],[-2.,4.,0.]]]);mapping=np.array([[0,1,2]])
    for depth,width in itertools.product((1e-2,1e-4,1e-6),(1.,.1,.01)):
        z0=.02*(1-depth);source=np.array([[0.,0.,z0],[1.,0.,z0+.02*depth/width],[0.,1.,z0]])
        reference,details=plane_reference(source)
        for name,options in [('legacy_component',{}),('legacy_resultant',dict(force_error_norm='resultant')),
                             ('normalized',dict(force_error_norm='resultant',normalized_floor=True)),
                             ('normalized_quarter_tolerance',dict(force_error_norm='resultant',normalized_floor=True))]:
            tol=.0000625 if name.endswith('quarter_tolerance') else .00025
            begin=time.perf_counter();r=patch.integrate(source,targets,.5,kn=10.,tolerance=tol,**options)
            value=applied(r,source,targets,[[0]],mapping);err=errors(value,reference)
            rows.append(dict(family='affine_plane',depth_fraction=depth,requested_active_width=width,method=name,tolerance=tol,
                             energy_Nmm=float(value[0]),reference_energy_Nmm=float(reference[0]),points=len(r['weights']),
                             energy_allowance_Nmm=r['energy_allowance'],estimated_energy_error_Nmm=r['energy_error_estimate'],
                             compression_energy_difference_Nmm=r['compression_energy_difference'],
                             passed=max(err['energy_error'],err['force_error'])<.001,wall_s=time.perf_counter()-begin,**err,**details))
            print(json.dumps(rows[-1]),flush=True)
    write_csv(root/'manufactured.csv',rows)
    return rows


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def capture_fixtures(root,run):
    # Small final-state arrays suffice; do not deserialize the large active
    # friction checkpoint merely to reconstruct a geometric test fixture.
    c=json.loads((run/'config.json').read_text());L=c['length_mm']
    with np.load(run/'last-state.npz') as saved:
        X=saved['reference_points'];t=saved['tets'];q0=saved['generalized_position']
        np.testing.assert_array_equal(saved['points'],Periodic(X,L).positions(q0))
    periodic=Periodic(X,L)
    contact=EmbeddedContact(X,t,periodic,c['contact_penalty_mpa_mm'],c['friction'],c['contact_skin_mm'],thickness=c['barrier_thickness_mm'])
    folder=root/'fixtures';folder.mkdir();fixtures=[]
    for compression in (0.,1e-5,1e-4):
        q=q0.copy();q[-4]-=compression;x=periodic.positions(q);H=deformation(q[-6:])*L
        inv=np.linalg.inv(H);frac=x[contact.nodes]@inv.T;pad=contact.skin*np.linalg.norm(inv,axis=1)
        lo=np.ceil(frac.min(0)-frac.max(0)-pad).astype(int);hi=np.floor(frac.max(0)-frac.min(0)+pad).astype(int)
        shifts=list(itertools.product(*[range(a,b+1) for a,b in zip(lo,hi)]))
        guard=contact.inner.guard;guard.trial_shifts=shifts;guard.neighbor_builds=int(compression*1e8)
        guard.neighbor_geometry=(contact.P@x,H);contact._candidate_build=-1
        contact._candidates((1521,1522,1652,1653))
        for face in (1521,1522,1652,1653):
            keys=sorted(set(contact._face_candidates[face]));targets=np.array([x[contact.faces[j]]+H@shift for j,shift in keys])
            groups=patch.components(np.array([contact.faces[j] for j,_ in keys]),[shift for _,shift in keys],X,L)
            identities=[(int(periodic.classes[node]),*np.rint((X[node]+L*np.asarray(shift))/(L*1e-10)).astype(np.int64)) for j,shift in keys for node in contact.faces[j]]
            unique=sorted(set(identities));mapping=np.array([unique.index(k) for k in identities]).reshape(-1,3)
            tri=X[contact.faces[face]];A=float(np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))/2)
            filename=f'case-{len(fixtures):03d}.npz'
            np.savez_compressed(folder/filename,source=x[contact.faces[face]],targets=targets,mapping=mapping)
            fixtures.append(dict(file=filename,face=face,extra_macro_compression=compression,reference_area_mm2=A,groups=groups,keys=[[j,list(map(int,s))] for j,s in keys]))
    (folder/'index.json').write_text(json.dumps(fixtures,indent=2)+'\n')
    (folder/'provenance.json').write_text(json.dumps(dict(run=str(run),source_sha256={name:hashlib.sha256((run/name).read_bytes()).hexdigest() for name in ('last-state.npz','config.json')},
                  scope='Geometry-only controls: accepted onset pose and two manufactured affine increments; no equilibrium or friction transfer claim.'),indent=2)+'\n')
    return folder


def rve(root,folder):
    fixtures=json.loads((folder/'index.json').read_text());rows=[];frozen={};reference_values={}
    for fixture in fixtures:
        with np.load(folder/fixture['file']) as z:source=z['source'];targets=z['targets'];mapping=z['mapping']
        groups=fixture['groups'];common=dict(kn=10.,vertex_map=mapping,groups=groups,pointwise=True,force_error_norm='resultant',normalized_floor=True)
        rules=[];values=[]
        for tol in (.00025,.0000625,.000015625):
            begin=time.perf_counter();rule=patch.integrate(source,targets,fixture['reference_area_mm2'],tolerance=tol,**common)
            value=applied(rule,source,targets,groups,mapping);rules.append(rule);values.append(value)
            print(json.dumps(dict(event='rve_level',face=fixture['face'],compression=fixture['extra_macro_compression'],tolerance=tol,points=len(rule['weights']),wall_s=time.perf_counter()-begin)),flush=True)
        reference=values[-1];reference_values[fixture['file']]=reference
        for tol,rule,value in zip((.00025,.0000625,.000015625),rules,values):
            err=errors(value,reference);row=dict(family='saved_rve',face=fixture['face'],extra_macro_compression=fixture['extra_macro_compression'],method='normalized_refinement',tolerance=tol,points=len(rule['weights']),energy_Nmm=float(value[0]),reference_energy_Nmm=float(reference[0]),**err)
            row['passed']=max(err['energy_error'],err['force_error'])<.001;rows.append(row)
        face=fixture['face']
        if fixture['extra_macro_compression']==0.:
            frozen[face]=rules[0]
        else:
            old=frozen[face];value=applied(old,source,targets,groups,mapping);err=errors(value,reference)
            rows.append(dict(family='saved_rve',face=face,extra_macro_compression=fixture['extra_macro_compression'],method='frozen_onset_rule',tolerance=.00025,
                             points=len(old['weights']),energy_Nmm=float(value[0]),reference_energy_Nmm=float(reference[0]),passed=max(err['energy_error'],err['force_error'])<.001,**err))
        np.savez_compressed(root/f"rule-{fixture['file']}",bary=rules[0]['bary'],weights=rules[0]['weights'],reference_value=reference)
        write_csv(root/'rve.csv',rows)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run',type=Path);p.add_argument('--fixtures',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    analytic=manufactured(a.output)
    folder=a.fixtures or (capture_fixtures(a.output,a.run) if a.run else None)
    actual=rve(a.output,folder) if folder else []
    normalized=[r for r in analytic if r['method'].startswith('normalized')]
    refined=[r for r in actual if r['method']=='normalized_refinement' and r['tolerance']>.000015625]
    frozen=[r for r in actual if r['method']=='frozen_onset_rule']
    summary=dict(scope=__doc__,manufactured_normalized_passed=all(r['passed'] for r in normalized),
                 rve_refinement_passed=all(r['passed'] for r in refined) if refined else None,
                 frozen_rule_failures=sum(not r['passed'] for r in frozen),frozen_rule_cases=len(frozen),
                 manufactured_cases=analytic,rve_cases=actual,wall_s=time.perf_counter()-start,
                 production_solver_modified=False,pressure_0_1_MPa_reached=False,
                 limits=['No friction-history integration, history projection or coupled pressure advance in this verification.',
                         'The finest RVE quadrature is a numerical reference, not an exact solution.',
                         'A frozen-rule failure is a diagnostic, not a passed coupled replay.'])
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if not summary['manufactured_normalized_passed'] or summary['rve_refinement_passed'] is False:raise SystemExit(1)


if __name__=='__main__':main()
