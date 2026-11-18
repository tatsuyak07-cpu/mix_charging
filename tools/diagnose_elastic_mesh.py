"""Initial elastic tangent diagnostic: no contact, plastic evolution or rigid support.

This is a separate linearized material test, never a replacement production case.
The inactive cylinder DOFs are omitted because no contact couples them to the solid.
"""
import argparse,gc,json
from pathlib import Path
import numpy as np
from packing_fem.rigid_inclusion import make_periodic
from packing_fem.explicit_material import geometry
from packing_fem.implicit_material import evaluate
from packing_fem.sparse_factor import factorize
from packing_fem.uniform_tet import refine


def solve(X,t,c,out,coarse=None):
    out=Path(out);out.mkdir(exist_ok=False,parents=True)
    p=make_periodic(X,c['length_mm'],c);V,_,D=geometry(X,t)
    _,gm,Km,st=evaluate(X,t,np.tile(np.eye(3),(len(t),1,1)),np.zeros(len(t)),np.linalg.inv(D),V,
                       dict(c,implicit_project_material_tangent=False),True)
    if st['ep'].max()!=0:raise ValueError('Initial tangent is not elastic')
    K=(p.A.T@Km@p.A).tocsr();del Km;gc.collect()
    strain=-1e-6;q=np.zeros(p.ndof);q[p.prescribed]=strain;n=p.body_nmicro
    rhs=-np.asarray(K[:n,p.prescribed].toarray()).ravel()*strain
    factor=factorize(K[:n,:n],c);q[:n]=factor.solve(rhs);factor.close();del factor
    g=K@q;energy=float(q@g)/2;modulus=float(g[p.prescribed]/(p.L**3*strain))
    residual=float(abs(g[:n]).max());energy_error=abs(energy-.5*g[p.prescribed]*strain)/max(abs(energy),1e-30)
    report=dict(nodes=len(X),tetrahedra=len(t),reference_volume_mm3=float(V.sum()),
        effective_C33_MPa=modulus,free_linear_residual_N=residual,energy_Nmm=energy,
        reaction_energy_relative_error=energy_error,contact_included=False,plastic_evolution_included=False,
        scope='Initial linear elastic tangent only; fixed transverse cell; cylinder has no contact coupling and no support is introduced.')
    if coarse is not None:
        oldX,old_displacement,edges,old_energy=coarse
        displacement=np.vstack([old_displacement,old_displacement[edges].mean(axis=1)])
        prolong=p.remap_positions(X+displacement,q[-6:]);prolong_energy=float(prolong@(K@prolong))/2
        report.update(prolonged_coarse_energy_Nmm=prolong_energy,
            nested_space_energy_relative_error=abs(prolong_energy/old_energy-1),
            refined_to_coarse_energy_ratio=energy/old_energy)
        if report['nested_space_energy_relative_error']>1e-7 or energy>old_energy*(1+1e-7):raise ValueError('Nested-space elastic energy check failed')
    if residual>1e-8 or energy_error>1e-7:raise ValueError('Linear equilibrium or energy check failed')
    np.savez_compressed(out/'solution.npz',X=X,t=t,q=q,displacement=p.positions(q)-X)
    (out/'assessment.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    return report,p.positions(q)-X


def run(out):
    out=Path(out);out.mkdir(exist_ok=False,parents=True);rows=[]
    for level in range(3):
        source=Path(f'results/rigid-stage4-001/small-load-screen-001/level-{level:03d}')
        c=json.loads((source/'config.json').read_text())
        with np.load(source/'state-0000.npz') as state:X=state['X'];t=state['t']
        report,u=solve(X,t,c,out/f'level-{level:03d}');rows.append(report)
        if level==0:coarseX=X.copy();coarseT=t.copy();coarseU=u.copy();coarseC=c.copy();coarseEnergy=report['energy_Nmm']
    Y,s,edges=refine(coarseX,coarseT)
    refined,_=solve(Y,s,coarseC,out/'same-boundary-refined',coarse=(coarseX,coarseU,edges,coarseEnergy))
    report=dict(original_levels=rows,same_faceted_boundary_refinement=refined,
        original_coarsest_vs_finest_relative=abs(rows[0]['effective_C33_MPa']/rows[-1]['effective_C33_MPa']-1),
        same_boundary_refinement_relative_change=abs(refined['effective_C33_MPa']/rows[0]['effective_C33_MPa']-1),
        diagnostic_only=True,full_compaction_convergence_verified=False)
    (out/'assessment.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();run(a.out)
