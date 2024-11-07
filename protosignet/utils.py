import contextlib
import os

import numpy as np


def makedirs(dp):
    """Create directories as needed for the desired directory path.

    Args:
        dp (str): directory path desired
    """
    with contextlib.suppress(OSError):
        os.makedirs(dp)


def rescale_01(x, xmin, xmax):
    """Rescales data between 0 and 1.

    Args:
        x (1D array): data
        xmin (float): minimum possible value
        xmax (float): maximum possible value

    Returns:
        rescaled (1D array): rescaled data
    """
    return (x - xmin) / (xmax - xmin)


def calc_hypervolume2D(pf_obj, ref):
    """Calculate the hypervolume indicator for an optimization problem with 2 objectives.

    Args:
        pf_obj (m x 2 array): m nondominated front individuals by j=2 objectives
        ref (1 x 2 array): reference point for HV calculation.
            ref value 1: the minimum value possible for objective 1
            ref value 2: the minimum value possible for objective 2

    Returns:
        hv (float): hypervolume value
    """
    pf_obj = np.unique(pf_obj, axis=0)  # get unique obj scores and sort by obj 1
    df1 = np.diff([ref[0], *list(pf_obj[:, 0])])  # rectangle widths
    df2 = np.abs(pf_obj[:, 1] - ref[1])  # rectangle heights
    hv = (df1 * df2).sum()
    return hv
