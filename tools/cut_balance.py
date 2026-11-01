"""Necessary area-balance conditions for unchanged positive surface samples.

For a cut from the opposite vertex to edge parameter f, sample i is on the
left if s_i = B_ib/(B_ia+B_ib) < f. Child area is f*A, whereas inherited mass
is a step function. This module enumerates its admissible intervals exactly
up to floating arithmetic; the actual remesh/pose audit remains mandatory.
"""
import numpy as np


def merge(intervals):
    result=[]
    for low,high in sorted(intervals):
        if low>high:continue
        if result and low<=result[-1][1]:result[-1][1]=max(high,result[-1][1])
        else:result.append([float(low),float(high)])
    return result


def intersect(left,right):
    result=[];i=j=0;left=merge(left);right=merge(right)
    while i<len(left) and j<len(right):
        low=max(left[i][0],right[j][0]);high=min(left[i][1],right[j][1])
        if low<=high:result.append([low,high])
        if left[i][1]<right[j][1]:i+=1
        else:j+=1
    return merge(result)


def balance(bary,weights,ia,ib,area,area_tolerance):
    B=np.asarray(bary,float);w=np.asarray(weights,float)
    if B.ndim!=2 or B.shape[1]!=3 or w.shape!=(len(B),) or not len(w):raise ValueError('Triangle samples required')
    if min(area,area_tolerance)<=0 or np.any(w<=0) or not np.isfinite(B).all() or not np.isfinite(w).all():raise ValueError('Finite positive mass/area required')
    if ia==ib or ia not in range(3) or ib not in range(3):raise ValueError('Distinct edge vertices required')
    if B.min()<-1e-12 or np.max(abs(B.sum(1)-1))>1e-10:raise ValueError('Invalid barycentric coordinates')
    denominator=B[:,ia]+B[:,ib];apex=denominator<=0
    s=np.clip(B[~apex,ib]/denominator[~apex],0.,1.)
    positions,labels=np.unique(s,return_inverse=True)
    masses=np.bincount(labels,weights=w[~apex],minlength=len(positions))
    cumulative=np.r_[0.,np.cumsum(masses)];boundaries=np.r_[0.,positions,1.]
    apex_mass=float(w[apex].sum());offsets=[0.,apex_mass] if apex_mass else [0.]
    intervals=[];roots=[];total=float(w.sum())
    for offset in offsets:
        for i,left_mass in enumerate(cumulative+offset):
            if not 0<left_mass<total:continue # both children need samples
            low,high=boundaries[i:i+2]
            # BOTH children must satisfy the same absolute area criterion.
            a=max(0.,low,(left_mass-area_tolerance)/area,1-(total-left_mass+area_tolerance)/area)
            b=min(1.,high,(left_mass+area_tolerance)/area,1-(total-left_mass-area_tolerance)/area)
            if a<=b and b>0 and a<1:intervals.append([a,b])
            f=float(left_mass/area)
            if 0<f<1 and low<=f<=high:roots.append(f)
    return dict(intervals=merge(intervals),roots=sorted(set(roots)),positions=positions.tolist(),cumulative_fraction=(cumulative/area).tolist(),
                apex_mass_fraction=apex_mass/area,area_mm2=float(area),area_tolerance_mm2=float(area_tolerance),total_mass_mm2=total)


def residual(bary,weights,ia,ib,fraction,area):
    B=np.asarray(bary);w=np.asarray(weights);denominator=B[:,ia]+B[:,ib]
    normal=denominator>0;left=normal&(B[:,ib]<=fraction*denominator)
    # Boundary assignment is checked by refine_rules; here report both apex sides.
    masses=[float(w[left].sum())]
    if np.any(~normal):masses.append(masses[0]+float(w[~normal].sum()))
    return min(max(abs(m-fraction*area),abs(w.sum()-m-(1-fraction)*area)) for m in masses)
