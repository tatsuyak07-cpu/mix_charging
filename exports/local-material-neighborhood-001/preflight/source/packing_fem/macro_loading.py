"""Work-conjugate loads for the six symmetric periodic cell coordinates.

Stress is positive in tension, reference volume in mm3, stresses in MPa.
The Cauchy load is a follower load; its path work must be integrated, rather
than inferred from a constant reference-area pressure potential.
"""
import numpy as np

PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
BASIS = np.zeros((6, 3, 3))
for k, (i, j) in enumerate(PAIRS):
    BASIS[k, i, j] = BASIS[k, j, i] = 1.


def first_piola(F, sigma):
    F = np.asarray(F, float)
    sigma = np.asarray(sigma, float)
    if F.shape != (3, 3) or sigma.shape != (3, 3) or not np.all(np.isfinite(F)) or not np.all(np.isfinite(sigma)):
        raise ValueError('Finite 3 by 3 deformation and stress required')
    if np.linalg.det(F) <= 0:
        raise ValueError('Positive cell determinant required')
    if not np.allclose(sigma, sigma.T, rtol=0, atol=1e-12):
        raise ValueError('Symmetric Cauchy stress required')
    return np.linalg.det(F) * sigma @ np.linalg.inv(F).T


def generalized_load(F, sigma, reference_volume):
    if not np.isfinite(reference_volume) or reference_volume <= 0:
        raise ValueError('Positive reference volume required')
    return reference_volume * np.einsum('kij,ij->k', BASIS, first_piola(F, sigma))


def recover_cauchy(F, generalized_force, reference_volume):
    """Recover symmetric work-equivalent cell stress, including contact virial.

The input is the *positive* resisting macro force (-internal generalized
force). Valid for the symmetric-positive-definite cell gauge. This is a cell
work diagnostic, not a volume average of skeleton stress alone.
"""
    F = np.asarray(F, float)
    if not np.allclose(F, F.T, atol=1e-12, rtol=0) or np.linalg.eigvalsh(F).min() <= 0:
        raise ValueError('Symmetric positive definite cell gauge required')
    operator = np.column_stack([generalized_load(F, B, reference_volume) for B in BASIS])
    coefficients = np.linalg.solve(operator, np.asarray(generalized_force))
    return np.einsum('k,kij->ij', coefficients, BASIS)


def uniaxial_load(F, pressure, reference_volume):
    return generalized_load(F, np.diag([0., 0., -pressure]), reference_volume)


def diagnostic_cauchy(F, force, reference_volume):
    """Unavailable outside the admissible gauge; never break failure archival."""
    try:
        return recover_cauchy(F, force, reference_volume).tolist()
    except (ValueError, np.linalg.LinAlgError):
        return None
