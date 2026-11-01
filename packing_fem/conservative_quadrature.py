"""Bounded positive reweighting of EXISTING material samples at remesh only.

Preserve coarse periodic contact virtual work, normal energy, committed and
applied friction energies, and parent area/polynomial moments. No new sample
history is initialized/interpolated. Infeasible projections reject/replay.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix,hstack,vstack,eye
from .explicit_periodic import affine_basis


def project(weights,conservation,contexts,tolerance=.001,factor=2.,max_entries=12000000,conservation_scales=None):
    w=np.asarray(weights,float);C=np.asarray(conservation,float);n=len(w)
    if n==0 or np.any(w<=0) or factor<=1 or not np.isfinite(factor):raise ValueError('Positive samples and weight factor >1 required')
    if (len(C)+sum(len(c['exact']) for c in contexts))*n>max_entries:raise ValueError('Conservative quadrature matrix budget exhausted')
    C=C*w;target=C.sum(1)
    scale=np.maximum(np.maximum(np.max(abs(C),axis=1),abs(target)),1e-30) if conservation_scales is None else np.asarray(conservation_scales,float)
    if scale.shape!=target.shape or np.any(scale<=0):raise ValueError('Invalid conservation scales')
    # Parameterize around the old rule in the mass/moment constraint nullspace.
    # This avoids ill-conditioned duplicate equality rows in the LP presolver.
    # EVERY original conservation row is independently checked afterwards.
    from scipy.linalg import svd
    if n*n>max_entries:raise ValueError('Conservative quadrature nullspace budget exhausted')
    if len(C):
        _,singular,V=svd(C/scale[:,None],full_matrices=len(C)<n)
        rank=int(np.sum(singular>1e-12*singular[0])) if len(singular) and singular[0]>0 else 0
        Z=V[rank:].T
    else:Z=np.eye(n)
    free=Z.shape[1];rows=[];upper=[]
    for ctx in contexts:
        M=np.asarray(ctx['matrix'])*w;reference=ctx['exact'];scales=ctx['scales'];absolute=np.r_[ctx.get('energy_budget',0.),np.full(len(reference)-1,ctx.get('force_budget',0.)/max(1,len(reference)-1))]
        allowance=.5*(tolerance*scales+absolute);normalizer=np.maximum(scales,absolute/tolerance)
        projected=(M/normalizer[:,None])@Z;baseline=M.sum(1)
        rows.extend([csr_matrix(projected),csr_matrix(-projected)])
        upper.extend([(reference+allowance-baseline)/normalizer,(-reference+allowance+baseline)/normalizer])
    # Positive bounded weights and the L1 change objective.
    rows.extend([csr_matrix(Z),csr_matrix(-Z),csr_matrix(Z),csr_matrix(-Z)])
    upper.extend([np.full(n,factor-1),np.full(n,1-1/factor),np.zeros(n),np.zeros(n)])
    A=vstack(rows).tocsr();b=np.r_[*upper]
    slack=vstack([csr_matrix((A.shape[0]-2*n,n)),-eye(n),-eye(n)]).tocsr()
    if free==0:
        if np.any(b< -1e-12):raise ValueError('Conservative existing-sample redistribution infeasible: no free modes')
        change=np.zeros(n)
    else:
        result=linprog(np.r_[np.zeros(free),w/w.sum()],A_ub=hstack([A,slack]),b_ub=b,
                       bounds=[(None,None)]*free+[(0,None)]*n,method='highs-ipm',
                       options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10,'time_limit':60.})
        if not result.success:raise ValueError('Conservative existing-sample redistribution infeasible: '+result.message)
        change=Z@result.x[:free]
    proposed=w*(1+change)
    error=float(np.max(abs(C@change)/scale)) if len(C) else 0.
    if error>1e-9 or np.any(proposed<=0) or np.min(proposed/w)<1/factor-1e-9 or np.max(proposed/w)>factor+1e-9:raise ValueError('Conservative quadrature failed independent conservation audit')
    for ctx in contexts:
        absolute=np.r_[ctx.get('energy_budget',0.),np.full(len(ctx['exact'])-1,ctx.get('force_budget',0.)/max(1,len(ctx['exact'])-1))]
        if np.any(abs(ctx['matrix']@proposed-ctx['exact'])>tolerance*ctx['scales']+absolute+1e-30):raise ValueError('Redistribution failed retained-response audit')
    return proposed,dict(relative_weight_factor=factor,conservation_error=error,conservation_nullity=free,relative_area_weight_change=float(np.sum(abs(proposed-w))/w.sum()),new_samples=0)


def redistribute(old,new,proposal):
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
        expanded['energy_budget']=min(ctx['energy_budget'],new.absolute_energy_budget/(2*len(contexts)))
        expanded['force_budget']=min(ctx['force_budget'],new.absolute_force_budget/(2*len(contexts)))
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
    # A vanishing Cartesian component is judged against the physical force
    # block, not relative to its own cancellation roundoff. Keep EVERY row.
    work[-6:]/=old.L
    force_scale=max(float(np.linalg.norm(work@w)),float(np.max(np.linalg.norm(work*w,axis=1))),1e-16)
    energy_scales=np.maximum(np.maximum(np.max(abs(energies*w),axis=1),abs(energies@w)),1e-18)
    moment_scales=np.concatenate([np.full(6,new.parent_groups[k]['area']) for k in new.parent_contexts])
    conservation_scales=np.r_[np.full(len(work),force_scale),energy_scales,moment_scales]
    proposed,report=project(w,np.r_[work,energies,*constraints],normal,new.quadrature_tolerance,new.redistribution_factor,new.atlas_max_entries,conservation_scales=conservation_scales)
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
