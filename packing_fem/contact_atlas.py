"""Positive empirical surface cubature of ALL retained counterexample poses.

This fits normal-potential integrals, not bulk solutions or displacement fields.
New point histories MUST still be computed by full coupled inactive replay.
Unseen poses always require the independent production surface audit.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.linalg import qr
from scipy.sparse import csr_matrix,hstack,vstack
from . import surface_patch as patch


def scales(value,energy_budget=0.,force_budget=0.,tolerance=.001):
    value=np.asarray(value)
    force=max(float(np.linalg.norm(value[1:10].reshape(3,3).sum(0))),float(np.linalg.norm(value[10:].reshape(-1,3),axis=1).sum()),1e-16)
    return np.r_[max(abs(value[0]),1e-18,energy_budget/tolerance),np.full(len(value)-1,max(force,force_budget/(tolerance*(len(value)-1))))]


def density(bary,pose,kn,thickness):
    return patch.grouped_field(bary,np.asarray(pose['source'],float),np.asarray(pose['targets'],float),pose['groups'],kn,thickness,np.asarray(pose['mapping'],int),True,pose.get('tie_rule','stop'))[0]


def refine_weights(matrix,rhs,weights):
    """Refine a feasible positive LP basis without relaxing moment tolerances."""
    support=np.flatnonzero(weights>0);A=matrix[:,support]
    scale=np.maximum(np.max(abs(A),axis=1),abs(rhs));scale=np.maximum(scale,1e-30)
    balanced=A/scale[:,None]
    _,_,pivot=qr(balanced.T,mode='economic',pivoting=True)
    selected=pivot[:len(support)];square=balanced[selected];target=rhs[selected]/scale[selected]
    try:
        values=np.linalg.solve(square,target)
        for _ in range(3):
            residual=np.asarray(target.astype(np.longdouble)-square.astype(np.longdouble)@values.astype(np.longdouble),float)
            values+=np.linalg.solve(square,residual)
        if values.min()>=-1e-13:
            candidate=np.zeros_like(weights);candidate[support]=np.maximum(values,0.)
            if np.max(abs(matrix@candidate-rhs))<np.max(abs(matrix@weights-rhs)):weights=candidate
    except np.linalg.LinAlgError:pass
    # Enforce the exact reference-area polynomial moments with a weighted
    # projection. All physical responses are audited again after this step.
    C=matrix[:6];target_moments=rhs[:6]
    for _ in range(3):
        try:correction=C.T@np.linalg.solve((C*weights)@C.T,target_moments-C@weights)
        except np.linalg.LinAlgError:break
        if np.min(1+correction)<=0:break
        weights=weights*(1+correction)
    return weights


def minimax_fit(matrix,rhs):
    """Smallest response residual with exact area moments and positive mass."""
    count=matrix.shape[1];response=csr_matrix(matrix[6:]);slack=csr_matrix(-np.ones((len(rhs)-6,1)))
    return linprog(np.r_[np.zeros(count),1.],
        A_eq=hstack([csr_matrix(matrix[:6]),csr_matrix((6,1))]),b_eq=rhs[:6],
        A_ub=vstack([hstack([response,slack]),hstack([-response,slack])]),b_ub=np.r_[rhs[6:],-rhs[6:]],
        bounds=(0,None),method='highs',options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10,'time_limit':60.})


def fit_accepts(matrix,rhs,weights,references,tolerance,energy_budget,force_budget):
    residual=abs(matrix[6:]@weights-rhs[6:]);offset=0
    for reference in references:
        count=len(reference);difference=residual[offset:offset+count]*scales(reference,energy_budget,force_budget,tolerance)
        baseline=scales(reference)
        if max(0.,difference[0]-tolerance*baseline[0])>energy_budget+1e-30:return False
        if np.maximum(difference[1:]-tolerance*baseline[1:],0.).sum()>force_budget+1e-30:return False
        offset+=count
    return True


def enrich_minimax(bary,matrix,rhs,evaluate_columns,limit,iterations=8,max_points=None,max_entries=None,diagnostics=None,acceptance=None):
    """Dual-guided material-coordinate enrichment; no response row is removed."""
    trace=[] if diagnostics is None else diagnostics
    result=minimax_fit(matrix,rhs);rounds=0
    for _ in range(iterations):
        if not result.success:break
        if acceptance is not None:
            if acceptance(matrix,rhs,result.x[:-1]):break
        elif result.x[-1]<=limit:break
        support=np.flatnonzero(result.x[:-1]>0)
        if not len(support):break
        activity=np.max(abs(matrix[6:,support]),axis=0)*result.x[support]
        anchor=bary[support[np.argsort(activity)[-48:]]]
        cloud=[]
        for fraction in (1e-10,1e-8,1e-6,1e-4,.01,.1,.5):
            for vertex in np.eye(3):cloud.extend((1-fraction)*anchor+fraction*vertex)
        for i in range(len(anchor)):
            for fraction in (.25,.5,.75):cloud.extend((1-fraction)*anchor[i]+fraction*anchor[i+1:])
        cloud=np.unique(np.asarray(cloud).reshape(-1,3),axis=0)
        existing=set(map(tuple,bary));cloud=np.array([point for point in cloud if tuple(point) not in existing]).reshape(-1,3)
        if not len(cloud):break
        room=len(cloud)
        if max_points is not None:room=min(room,max_points-len(bary))
        if max_entries is not None:room=min(room,max_entries//len(rhs)-len(bary))
        if room<=0:break
        def safe_columns(points):
            try:return points,evaluate_columns(points)
            except patch.IntegrationRefinement:
                if len(points)==1:return np.empty((0,3)),np.empty((len(rhs),0))
                half=len(points)//2;a,A=safe_columns(points[:half]);b,B=safe_columns(points[half:])
                return np.r_[a,b],np.c_[A,B]
        dual=result.ineqlin.marginals;half=len(rhs)-6
        chosen_points=np.empty((0,3));chosen_columns=np.empty((len(rhs),0));chosen_cost=np.empty(0)
        minimum=0.;valid_count=0
        batch=min(512,max_entries//len(rhs)) if max_entries is not None else 512
        for start in range(0,len(cloud),max(1,batch)):
            valid,columns=safe_columns(cloud[start:start+max(1,batch)])
            if not len(valid):continue
            valid_count+=len(valid)
            reduced=-columns[:6].T@result.eqlin.marginals-columns[6:].T@(dual[:half]-dual[half:])
            minimum=min(minimum,float(reduced.min()));take=np.flatnonzero(reduced< -1e-9)
            chosen_points=np.r_[chosen_points,valid[take]];chosen_columns=np.c_[chosen_columns,columns[:,take]];chosen_cost=np.r_[chosen_cost,reduced[take]]
            keep=np.argsort(chosen_cost)[:min(32,room)]
            chosen_points=chosen_points[keep];chosen_columns=chosen_columns[:,keep];chosen_cost=chosen_cost[keep]
        trace.append(dict(minimum_reduced_cost=minimum,proposed=len(cloud),singular_candidates=len(cloud)-valid_count,selected=len(chosen_points),residual=float(result.x[-1])))
        if not len(chosen_points):break
        trial_bary=np.r_[bary,chosen_points];trial_matrix=np.c_[matrix,chosen_columns]
        trial=minimax_fit(trial_matrix,rhs)
        if not trial.success or trial.x[-1]>result.x[-1]+1e-10:break
        bary,matrix,result=trial_bary,trial_matrix,trial;rounds+=1
    return bary,matrix,result,rounds


def build(poses,area,kn,thickness,tolerance=.001,previous_bary=None,max_poses=32,max_points=80000,max_entries=12000000,energy_budget=0.,force_budget=0.):
    if not poses or len(poses)>max_poses:raise patch.IntegrationRefinement('Contact pose-atlas budget exceeded; no past pose discarded')
    references=[];candidates=[]
    for pose in poses:
        r=patch.integrate(np.asarray(pose['source'],float),np.asarray(pose['targets'],float),area,kn=kn,thickness=thickness,
                          tolerance=tolerance/16,vertex_map=np.asarray(pose['mapping'],int),groups=pose['groups'],pointwise=True,error_budget='normalized',absolute_energy_budget=energy_budget/16,absolute_force_budget=force_budget/16,tie_rule=pose.get('tie_rule','stop'))
        references.append(r['value']);candidates.append(r['bary'])
    # Try the previously certified positive support plus the new pose first.
    # If this support is insufficient, use the union of every reference rule.
    pools=[np.r_[np.asarray(previous_bary).reshape(-1,3),candidates[-1]]] if previous_bary is not None else []
    pools.append(np.concatenate(candidates));last_error='No positive cubature found'
    for pool in pools:
        bary=np.unique(pool,axis=0);rows=6+sum(len(v) for v in references)
        if len(bary)>max_points or rows*len(bary)>max_entries:
            last_error=f'Contact atlas matrix budget exceeded: points={len(bary)}/{max_points}, entries={rows*len(bary)}/{max_entries}; no reference pose discarded';continue
        matrix=[np.ones(len(bary)),bary[:,0],bary[:,1],bary[:,0]**2,bary[:,0]*bary[:,1],bary[:,1]**2]
        rhs=[1.,1/3,1/3,1/6,1/12,1/6]
        for pose,value in zip(poses,references):
            scale=scales(value,energy_budget,force_budget,tolerance);values=density(bary,pose,kn,thickness)*area
            matrix.extend((values/scale).T);rhs.extend(value/scale)
        matrix=np.asarray(matrix);rhs=np.asarray(rhs)
        def columns(points):
            values=[np.ones(len(points)),points[:,0],points[:,1],points[:,0]**2,points[:,0]*points[:,1],points[:,1]**2]
            for pose,value in zip(poses,references):values.extend((area*density(points,pose,kn,thickness)/scales(value,energy_budget,force_budget,tolerance)).T)
            return np.asarray(values)
        # Force balance and material moments introduce exact/near row
        # dependencies. Identify independent rows after equilibration, solve in
        # physical response scales, then
        # audit EVERY original row and every applied physical response below.
        row_scale=np.maximum(np.max(abs(matrix),axis=1),abs(rhs));row_scale=np.maximum(row_scale,1e-30)
        equilibrated=matrix/row_scale[:,None]
        _,R,pivot=qr(equilibrated.T,mode='economic',pivoting=True)
        diagonal=abs(np.diag(R));rank=int(np.sum(diagonal>max(diagonal.max(),1.)*1e-11));selected=pivot[:rank]
        for fit_mode in ('independent_equalities','minimax_responses_exact_moments'):
            if fit_mode=='independent_equalities':
                result=linprog(np.zeros(len(bary)),A_eq=matrix[selected],b_eq=rhs[selected],bounds=(0,None),method='highs',
                               options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10,'time_limit':60.})
            else:
                enrichment_diagnostics=[]
                bary,matrix,result,enrichment_rounds=enrich_minimax(bary,matrix,rhs,columns,tolerance*5/8,max_points=max_points,max_entries=max_entries,diagnostics=enrichment_diagnostics,acceptance=lambda M,b,w:fit_accepts(M,b,w,references,tolerance,energy_budget,force_budget))
            if not result.success:last_error='Positive contact cubature rejected: '+result.message;continue
            if fit_mode=='independent_equalities':enrichment_rounds=0;enrichment_diagnostics=[]
            raw=result.x[:len(bary)];solution=refine_weights(matrix,rhs,np.maximum(raw,0.))
            keep=solution>0;points=bary[keep];weights=area*solution[keep]
            moment_error=float(np.max(abs(matrix[:6]@solution-rhs[:6])))
            audits=[]
            for pose,value in zip(poses,references):
                applied=weights@density(points,pose,kn,thickness);error=abs(applied-value)/scales(value,energy_budget,force_budget,tolerance)
                audits.append(dict(energy_error=float(error[0]),force_error=float(error[1:].max())))
            # This is ONLY a proposal: retained reference responses must meet
            # the same mixed criterion, then the entire coupled trajectory
            # independently rechecks actual poses and the global error sum.
            if moment_error>1e-11 or not fit_accepts(matrix,rhs,solution,references,tolerance,energy_budget,force_budget):
                last_error=f'Positive contact cubature failed independent applied-rule audit: moment={moment_error}, responses={audits}, enrichment={enrichment_diagnostics}';continue
            return dict(bary=points,weights=weights,atlas=poses,audits=audits,
                        enrichment_diagnostics=enrichment_diagnostics,enrichment_rounds=enrichment_rounds,fit_mode=fit_mode,proposal_only=True,requires_coupled_replay=True,fit_acceptance=dict(relative=tolerance,absolute_energy=energy_budget,absolute_force_l1=force_budget),candidate_points=len(bary),moment_error=moment_error,reference_tolerance=tolerance/16)
    raise patch.IntegrationRefinement(last_error)
