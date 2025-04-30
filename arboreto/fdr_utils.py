
import pandas as pd
import numpy as np
from numba import njit, prange, set_num_threads, types


@njit(nopython=True, nogil=True)
def _merge_sorted_arrays(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Merge two sorted 1D NumPy arrays into a single sorted array.

    :param a: a 1D NumPy array sorted in ascending order.
    :param b: a 1D NumPy array sorted in ascending order.
    :return: a 1D NumPy array containing all elements from `a` and `b`, merged in sorted order.
    """

    lenA,lenB = a.shape[0], b.shape[0]
    # Get searchsorted indices
    idx = np.searchsorted(a,b)
    # Offset each searchsorted indices with ranged array to get new positions of b in output array
    b_pos = np.arange(lenB) + idx
    lenTotal = lenA+lenB
    mask = np.ones(lenTotal,dtype=types.boolean)
    out = np.empty(lenTotal,dtype=types.float64)
    mask[b_pos] = False
    out[b_pos] = b
    out[mask] = a
    return out


@njit(nopython=True, parallel=True, nogil=True)
def _pairwise_wasserstein_dists(sorted_matrix, num_threads):
    """
    Compute the pairwise 1D Wasserstein distances between all columns of a sorted matrix.

    Each column in `sorted_matrix` represents a sorted 1D empirical distribution. The function computes the
    Wasserstein distance between each pair of columns, assuming uniform sample weights.

    :param sorted_matrix: a 2D NumPy array where each column is a sorted in ascending order.
    :param num_threads: number of threads to use for parallel execution. If -1, uses the default setting.
    :return: distance matrix, i.e. a 2D NumPy array containing the pairwise Wasserstein distances.
    """

    if num_threads != -1:
        set_num_threads(num_threads)
    num_cols = sorted_matrix.shape[1]
    num_rows = sorted_matrix.shape[0]
    distance_mat = np.zeros((num_cols, num_cols))
    for col1 in prange(num_cols):
        for col2 in range(col1 + 1, num_cols):
            all_values = _merge_sorted_arrays(sorted_matrix[:, col1], sorted_matrix[:, col2])
            # Compute the differences between pairs of successive values of u and v.
            deltas = np.diff(all_values)
            # Get the respective positions of the values of u and v among the values of
            # both distributions.
            col1_cdf_indices = np.searchsorted(sorted_matrix[:, col1], all_values[:-1], 'right')
            col2_cdf_indices = np.searchsorted(sorted_matrix[:, col2], all_values[:-1], 'right')
            # Calculate the CDFs of u and v using their weights, if specified.
            col1_cdf = col1_cdf_indices / num_rows
            col2_cdf = col2_cdf_indices / num_rows
            # Compute the value of the integral based on the CDFs.
            distance = np.sum(np.multiply(np.abs(col1_cdf - col2_cdf), deltas))
            distance_mat[col1, col2] = distance
            distance_mat[col2, col1] = distance
    return distance_mat


def compute_wasserstein_distance_matrix(expression_mat : pd.DataFrame, num_threads: int = -1):
    """
    Compute the pairwise 1D Wasserstein distance matrix between columns of a gene expression matrix.

    Each column in the input DataFrame is treated as a 1D empirical distribution.
    The function sorts the values in each column and computes the pairwise Wasserstein distances
    using uniform sample weights.

    :param expression_mat: a pandas DataFrame where each column is a gene's expression profile across samples.
    :param num_threads: number of threads to use for parallel execution. If -1, uses the default setting.
    :return: a pandas DataFrame containing the symmetric matrix of pairwise Wasserstein distances.
    """
    numpy_mat = expression_mat.to_numpy()
    numpy_mat = np.sort(numpy_mat, axis=0)
    distance_mat = _pairwise_wasserstein_dists(numpy_mat, num_threads)
    distance_mat = pd.DataFrame(distance_mat, columns=expression_mat.columns)
    return distance_mat



