"""Bounded conservative reweighting with shared L1 contact-error budgets.

Preserve coarse periodic contact virtual work, normal energy, committed and
applied friction energies, and parent area/polynomial moments. No new sample
history is initialized/interpolated. Infeasible projections reject/replay.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix,vstack
from .explicit_periodic import affine_basis


def response_audit(weights,contexts,tolerance):
    """The same positive excess/L1 accounting as the parent/global audit."""
    groups={};reports=[];passed=True
    for ctx in contexts:
        diff=abs(ctx['matrix']@weights-ctx['exact'])
        excess=np.maximum(diff-tolerance*ctx['scales'],0.)
        energy=float(excess[0]);force=float(excess[1:].sum())
        EB=ctx.get('energy_budget',0.);FB=ctx.get('force_budget',0.)
        passed=passed and energy<=EB+1e-30 and force<=FB+1e-30
        key=ctx.get('budget_group')
        if key is not None:
            g=groups.setdefault(key,[0.,0.,ctx['global_energy_budget'],ctx['global_force_budget']])
            if g[2:]!=[ctx['global_energy_budget'],ctx['global_force_budget']]:raise ValueError('Inconsistent global response budget')
            g[0]+=energy;g[1]+=force
        reports.append(dict(energy_excess=energy,force_excess_L1=force))
    passed=passed and all(e<=EB+1e-30 and f<=FB+1e-30 for e,f,EB,FB in groups.values())
    return dict(passed=bool(passed),contexts=reports,groups={str(k):dict(energy_excess=e,force_excess_L1=f,energy_limit=EB,force_limit=FB) for k,(e,f,EB,FB) in groups.items()})


def project(weights,conservation,contexts,tolerance=.001,factor=2.,max_entries=12000000,conservation_scales=None,reference_weights=None):
    w=np.asarray(weights,float);C=np.asarray(conservation,float);n=len(w)
    if n==0 or not np.isfinite(w).all() or np.any(w<=0) or factor<=1 or not np.isfinite(factor):raise ValueError('Positive samples and weight factor >1 required')
    fixed_reference=w if reference_weights is None else np.asarray(reference_weights,float)
    if fixed_reference.shape!=w.shape or not np.isfinite(fixed_reference).all() or np.any(fixed_reference<=0):raise ValueError('Invalid fixed reference weights')
    if (len(C)+sum(len(c['exact']) for c in contexts))*n>max_entries or n*n>max_entries:raise ValueError('Conservative quadrature matrix budget exhausted')
    if not np.isfinite(tolerance) or tolerance<=0:raise ValueError('Positive quadrature tolerance required')
    for c in contexts:
        if min(c.get('energy_budget',0.),c.get('force_budget',0.),c.get('global_energy_budget',0.),c.get('global_force_budget',0.))<0:raise ValueError('Negative response budget')
    C=C*w;target=C.sum(1)
    scale=np.maximum(np.maximum(np.max(abs(C),axis=1),abs(target)),1e-30) if conservation_scales is None else np.asarray(conservation_scales,float)
    if scale.shape!=target.shape or not np.isfinite(scale).all() or np.any(scale<=0):raise ValueError('Invalid conservation scales')
    from scipy.linalg import svd
    if len(C):
        _,singular,V=svd(C/scale[:,None],full_matrices=len(C)<n)
        rank=int(np.sum(singular>1e-12*singular[0])) if len(singular) and singular[0]>0 else 0
        Z=V[rank:].T
    else:Z=np.eye(n)
    free=Z.shape[1]
    # A certified old rule is admissible even exactly on the audit boundary.
    baseline=response_audit(w,contexts,tolerance)
    within=np.all(w>=fixed_reference/factor) and np.all(w<=fixed_reference*factor)
    if baseline['passed'] and within:
        return w.copy(),dict(relative_weight_factor=factor,conservation_error=0.,conservation_nullity=free,relative_area_weight_change=0.,new_samples=0,l1_audit=baseline,baseline_admissible=True)
    # Variables: conservation-nullspace modes, weight-change L1 epigraph,
    # and nonnegative response excesses. Absolute force budgets constrain SUMS
    # of excesses, never equally apportioned per Cartesian component.
    sizes=[len(c['exact']) for c in contexts];offsets=np.r_[0,np.cumsum(sizes)].astype(int)
    total=free+n+int(offsets[-1]);rows=[];upper=[];bounds=[(None,None)]*free+[(0,None)]*n
    def row(coeff,rhs):rows.append(csr_matrix(coeff));upper.extend(np.atleast_1d(rhs))
    groups={};guard=1-1e-6
    for j,ctx in enumerate(contexts):
        k=sizes[j];EB=ctx.get('energy_budget',0.);FB=ctx.get('force_budget',0.)
        normalizer=np.maximum(ctx['scales'],np.r_[max(EB,ctx.get('global_energy_budget',0.)),np.full(k-1,max(FB,ctx.get('global_force_budget',0.)))])
        start=free+n+offsets[j];A=np.zeros((k,total));A[:,:free]=(ctx['matrix']*w/normalizer[:,None])@Z
        A[np.arange(k),start+np.arange(k)]=-1
        base=ctx['matrix']@w;relative=guard*tolerance*ctx['scales']
        row(A,(ctx['exact']+relative-base)/normalizer)
        B=A.copy();B[:,:free]*=-1;row(B,(-ctx['exact']+relative+base)/normalizer)
        bounds.extend([(0,guard*EB/normalizer[0])]+[(0,None) if FB else (0,0)]*(k-1))
        if FB:
            v=np.zeros(total);v[start+1:start+k]=normalizer[1:];row(v/FB,guard)
        key=ctx.get('budget_group')
        if key is not None:
            g=groups.setdefault(key,dict(energy=np.zeros(total),force=np.zeros(total),EB=ctx['global_energy_budget'],FB=ctx['global_force_budget']))
            g['energy'][start]=normalizer[0];g['force'][start+1:start+k]=normalizer[1:]
    for g in groups.values():
        for name,budget in (('energy',g['EB']),('force',g['FB'])):
            if budget:row(g[name]/budget,guard)
            elif np.any(g[name]):row(g[name],0.)
    A=np.zeros((n,total));A[:,:free]=Z
    row(A,factor*fixed_reference/w-1);row(-A,1-fixed_reference/(factor*w))
    A[:,free:free+n]=-np.eye(n);row(A,np.zeros(n))
    A[:,:free]*=-1;row(A,np.zeros(n))
    objective=np.r_[np.zeros(free),w/w.sum(),np.zeros(int(offsets[-1]))]
    result=linprog(objective,A_ub=vstack(rows).tocsr(),b_ub=np.asarray(upper),bounds=bounds,method='highs-ipm',options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10,'time_limit':60.})
    if not result.success:raise ValueError('Conservative existing-sample redistribution infeasible: '+result.message)
    change=Z@result.x[:free];proposed=w*(1+change)
    error=float(np.max(abs(C@change)/scale)) if len(C) else 0.
    if error>1e-9 or np.any(proposed<=0) or np.min(proposed/fixed_reference)<1/factor-1e-9 or np.max(proposed/fixed_reference)>factor+1e-9:raise ValueError('Conservative quadrature failed independent conservation audit')
    audit=response_audit(proposed,contexts,tolerance)
    if not audit['passed']:raise ValueError('Redistribution failed retained-response L1 audit')
    return proposed,dict(relative_weight_factor=factor,conservation_error=error,conservation_nullity=free,relative_area_weight_change=float(np.sum(abs(proposed-w))/w.sum()),new_samples=0,l1_audit=audit,baseline_admissible=False)


def redistribute(old,new,proposal,additional_contexts=(),reference_weights=None):
    """Construct linear conservation constraints from the applied potential."""
    from .parent_quadrature import historical
    contexts=list(new.parent_contexts.values());nodes=np.array(sorted({int(n) for c in contexts for n in c['nodes']}),int);index={int(n):i for i,n in enumerate(nodes)};w=new.inner.integration_area0[nodes].copy();n=len(nodes)
    if not n:raise ValueError('No existing parent samples to redistribute')
    N=len(new.X);R=csr_matrix((proposal['weights'].ravel(),(np.repeat(np.arange(N),2),proposal['supports'].ravel())),shape=(N,len(old.X)))
    work=np.zeros((old.periodic.ndof,n));energies=np.zeros((4,n));constraints=[];normal=[]
    for record in new.inner.trial_audit['records']:
        node=int(record['support'][0])
        if node not in index:continue
        k=index[node];aug=np.zeros_like(new.inner.X);np.add.at(aug,record['support'],record['local_force']/w[k])
        force=np.asarray(R.T@(new.P.T@aug));extra=np.einsum('aij,ai->j',affine_basis(new.L*record['images']),record['local_force']/w[k])
        work[:,k]+=old.periodic.force(force,extra)
        energies[3,k]+=.5*record['kt']*float(record['spring']@record['spring'])/w[k]
    for history,row in ((new.history,1),(new.inner.accepted_force_history,2)):
        for key,state in history.items():
            if key[1] in index:energies[row,index[key[1]]]+=state['energy']/w[index[key[1]]]
    def expand(ctx):
        M=np.zeros((len(ctx['exact']),n))
        for i,node in enumerate(ctx['nodes']):M[:,index[int(node)]]=ctx['matrix'][:,i]
        return dict(ctx,matrix=M)
    for leader,ctx in new.parent_contexts.items():
        expanded=expand(ctx)
        expanded.update(budget_group='accepted',global_energy_budget=new.absolute_energy_budget/2,global_force_budget=new.absolute_force_budget/2)
        normal.append(expanded)
        for i,node in enumerate(ctx['nodes']):energies[0,index[int(node)]]+=ctx['matrix'][0,i]
        group=new.parent_groups[leader];tri=new.X[list(group['vertices'])];E=(tri[1:]-tri[0]).T;positions=(new.P@new.X)[ctx['nodes']]
        uv=(positions-tri[0])@np.linalg.pinv(E).T;B=np.c_[1-uv.sum(1),uv]
        local=np.array([np.ones(len(B)),B[:,0],B[:,1],B[:,0]**2,B[:,0]*B[:,1],B[:,1]**2]);M=np.zeros((6,n))
        for i,node in enumerate(ctx['nodes']):M[:,index[int(node)]]=local[:,i]
        constraints.append(M)
        rule=new.rules[str(group['faces'][0])]
        for name in ('parent_atlas','parent_controls'):
            for pose in rule.get(name,[]):normal.append(expand(historical(new,leader,pose)))
    for ctx in additional_contexts:
        expanded=expand(ctx)
        expanded.update(budget_group='trial',global_energy_budget=new.absolute_energy_budget/2,global_force_budget=new.absolute_force_budget/2)
        normal.append(expanded)
    # A vanishing Cartesian component is judged against the physical force
    # block, not relative to its own cancellation roundoff. Keep EVERY row.
    work[-6:]/=old.L
    force_scale=max(float(np.linalg.norm(work@w)),float(np.max(np.linalg.norm(work*w,axis=1))),1e-16)
    energy_scales=np.maximum(np.maximum(np.max(abs(energies*w),axis=1),abs(energies@w)),1e-18)
    moment_scales=np.concatenate([np.full(6,new.parent_groups[k]['area']) for k in new.parent_contexts])
    conservation_scales=np.r_[np.full(len(work),force_scale),energy_scales,moment_scales]
    reference=None if reference_weights is None else np.array([reference_weights[int(node)] for node in nodes])
    proposed,report=project(w,np.r_[work,energies,*constraints],normal,new.quadrature_tolerance,new.redistribution_factor,new.atlas_max_entries,conservation_scales=conservation_scales,reference_weights=reference)
    ratio=dict(zip(map(int,nodes),proposed/w))
    for face,(point_nodes,B,weights) in list(new.face_points.items()):
        updated=np.array([weight*ratio.get(int(node),1.) for node,weight in zip(point_nodes,weights)])
        new.face_points[face]=(point_nodes,B,updated);new.rules[str(face)]['weights']=updated.tolist()
    new.inner.quadrature_weights[nodes]=proposed;new.inner.guard.node_area0[nodes]=proposed
    from .parent_quadrature import groups
    groups(new.X,new.faces,new.rules,new.L)
    for history in (new.history,new.inner.accepted_force_history):
        for key,state in history.items():state['energy']*=ratio.get(int(key[1]),1.)
    return report
