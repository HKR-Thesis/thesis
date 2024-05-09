from numba import njit

@njit
def custom_digitize(value, bins):
    for i in range(len(bins)):
        if value < bins[i]:
            return i
    return len(bins) - 1