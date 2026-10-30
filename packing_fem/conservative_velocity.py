"""Periodic velocity projection preserving physical P, angular momentum and KE.

Project to the affine momentum-constraint space in the new mass metric, then
scale only its momentum-null component to the old kinetic energy. Infeasible
energy/constraint data are rejected; prescribed velocities are never changed.
"""
import numpy as np


def moments(x,mass,velocity):
    return np.r_[(mass[:,None]*velocity).sum(0),np.cross(x,mass[:,None]*velocity).sum(0)]


def project(periodic,x,mass,guess,target_moments,target_energy):
    if target_energy<0 or not np.isfinite(target_energy):raise ValueError('Invalid target kinetic energy')
    # Rows are exact work-conjugate loads for physical P and L. force()
    # includes the zero-momentum frame's translation Schur complement.
    rows=[]
    for axis in np.eye(3):rows.append(periodic.force(mass[:,None]*axis))
    for axis in np.eye(3):rows.append(periodic.force(mass[:,None]*np.cross(axis,x)))
    A=np.asarray(rows);free=periodic.free
    vfixed=np.zeros(periodic.ndof)
    if periodic.prescribed_active:vfixed[periodic.prescribed]=guess[periodic.prescribed]
    center=vfixed.copy();center[free]=-periodic.solve_free_mass(np.asarray(periodic.M@vfixed)[free])
    # Balance units of P and L before the six-by-six rank test.
    scales=np.r_[np.ones(3),np.full(3,periodic.L)]
    C=A[:,free]/scales[:,None];target=(np.asarray(target_moments)-A@center)/scales
    U=np.column_stack([periodic.solve_free_mass(row) for row in C])
    S=C@U;S=(S+S.T)/2
    values,vectors=np.linalg.eigh(S);cut=128*np.finfo(float).eps*max(float(values.max()),1e-300)
    keep=values>cut
    inverse=(vectors[:,keep]/values[keep])@vectors[:,keep].T
    minimum=center.copy();minimum[free]+=U@(inverse@target)
    residual=guess[free]-minimum[free];residual-=U@(inverse@(C@residual))
    minimum_energy=float(.5*minimum@(periodic.M@minimum))
    residual_energy=float(.5*residual@(periodic.Mff@residual))
    roundoff=1024*np.finfo(float).eps*max(target_energy,minimum_energy,1e-300)
    if minimum_energy>target_energy+roundoff:raise ValueError('Momentum/prescribed constraints require more kinetic energy than available')
    available=max(0.,target_energy-minimum_energy)
    if residual_energy<=roundoff and available>roundoff:raise ValueError('No resolved momentum-null velocity component for energy transfer')
    result=minimum.copy()
    if residual_energy>0:result[free]+=np.sqrt(available/residual_energy)*residual
    actual_velocity=periodic.velocities(result);actual=moments(x,mass,actual_velocity)
    scale=max(np.sqrt(2*mass.sum()*max(target_energy,1e-300)),1e-300)
    error=float(np.linalg.norm((actual-target_moments)/scales)/scale)
    energy=float(.5*result@(periodic.M@result));energy_error=abs(energy-target_energy)/max(target_energy,1e-300)
    if error>1e-10 or energy_error>1e-10:raise ValueError('Conservative velocity projection failed independent physical audit')
    return result,dict(method='momentum_energy',momentum_scaled_error=error,kinetic_relative_error=energy_error,
                       constraint_rank=int(keep.sum()),minimum_kinetic_energy_Nmm=minimum_energy,target_kinetic_energy_Nmm=target_energy)
