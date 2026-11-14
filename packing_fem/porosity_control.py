"""Conservative endpoint approach, including the first increment after restart."""


def limited_increment(increment, porosity, compression, target, minimum, slope=None):
    if porosity >= .2:
        return increment
    if slope is None or slope <= 0:
        slope = (1-porosity)/max(1-compression, 1e-10)
    return min(increment, max(minimum, .5*(porosity-target)/slope))
