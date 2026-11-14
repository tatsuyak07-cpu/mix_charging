"""Conservative endpoint approach, including the first increment after restart."""


def limited_increment(increment, porosity, compression, target, minimum, slope=None):
    if porosity >= .2:
        return increment
    if slope is None or slope <= 0:
        slope = (1-porosity)/max(1-compression, 1e-10)
    return min(increment, max(minimum, .5*(porosity-target)/slope))


def volume_limited_increment(increment, porosity, compression, target, minimum,
                             rigid_fraction=0., safety=.95):
    """Predict the target cell volume with the CURRENT solid volume held fixed.

    This is a numerical load predictor, not a material-volume constraint.
    The driver measures the solved porosity and rejects endpoint overshoots.
    """
    if not 0 < safety <= 1:
        raise ValueError('Endpoint safety must be in (0,1]')
    if porosity >= .2:
        return increment
    effective_height = 1-compression-rigid_fraction
    if effective_height <= 0 or not 0 < target < 1:
        raise ValueError('Invalid target or cell volume outside the rigid inclusion')
    predicted = safety*effective_height*(porosity-target)/(1-target)
    return min(increment, max(minimum, predicted))
