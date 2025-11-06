"""
Distance metrics computation for F-matrix analysis.

Provides various distance metrics including:
- Nei's genetic distance
- F-based distances (1-F, 1/F, etc.)
"""

import numpy as np


def compute_nei_distance(F_matrix):
    """
    Compute Nei's standard genetic distance from F-matrix.

    Nei's standard genetic distance (1972):
        D_ij = -ln(F_ij / sqrt(F_ii * F_jj))

    where F_ij is the probability of identity by descent.

    Args:
        F_matrix: (M, M) F-matrix (IBD probability matrix)

    Returns:
        distance_matrix: (M, M) Nei's genetic distance

    References:
        Nei, M. (1972). Genetic distance between populations.
        The American Naturalist, 106(949), 283-292.
    """
    M = F_matrix.shape[0]
    distance = np.zeros((M, M))

    for i in range(M):
        for j in range(M):
            if i == j:
                distance[i, j] = 0.0  # Self-distance is 0
            else:
                # Nei's formula: D = -ln(J_xy / sqrt(J_x * J_y))
                J_ij = F_matrix[i, j]
                J_ii = F_matrix[i, i]
                J_jj = F_matrix[j, j]

                # Avoid log(0) and division by zero
                numerator = max(J_ij, 1e-10)
                denominator = max(np.sqrt(J_ii * J_jj), 1e-10)

                distance[i, j] = -np.log(numerator / denominator)

    return distance


def compute_f_distance(F_matrix, method='1-F'):
    """
    Compute distance from F-matrix using various methods.

    Args:
        F_matrix: (M, M) F-matrix (IBD probability matrix)
        method: Distance metric to use
            '1-F': Simple complement (1 - F_ij)
                   Interprets F as similarity, distance = dissimilarity
            '1/F': Inverse (1 / F_ij)
                   Used in existing probability_of_identity.py
                   Larger F → smaller distance
            'nei': Nei's genetic distance
                   Standard population genetics metric
            'sqrt': sqrt(1 - F_ij)
                   Sometimes used in population genetics
            '-log': -log(F_ij)
                   Additive distance metric

    Returns:
        distance_matrix: (M, M) distance matrix

    Examples:
        >>> F = np.array([[1.0, 0.9, 0.7],
        ...               [0.9, 1.0, 0.8],
        ...               [0.7, 0.8, 1.0]])
        >>> d = compute_f_distance(F, method='1-F')
        >>> d[0, 1]  # Distance between agent 0 and 1
        0.1
    """
    if method == '1-F':
        # Simple complement: higher F → lower distance
        distance = 1.0 - F_matrix
        # Ensure diagonal is 0
        np.fill_diagonal(distance, 0.0)
        return distance

    elif method == '1/F':
        # Inverse: used in existing code
        # Higher F → lower distance
        distance = 1.0 / (F_matrix + 1e-10)
        # Diagonal should be 1 (self-identity has F=1)
        np.fill_diagonal(distance, 0.0)
        return distance

    elif method == 'nei':
        return compute_nei_distance(F_matrix)

    elif method == 'sqrt':
        # Square root of dissimilarity
        distance = np.sqrt(np.maximum(1.0 - F_matrix, 0))
        np.fill_diagonal(distance, 0.0)
        return distance

    elif method == '-log':
        # Additive metric: -log(F)
        distance = -np.log(np.maximum(F_matrix, 1e-10))
        np.fill_diagonal(distance, 0.0)
        return distance

    else:
        raise ValueError(f"Unknown method: {method}. "
                        f"Choose from: '1-F', '1/F', 'nei', 'sqrt', '-log'")


def compare_distance_methods(F_matrix, methods=None):
    """
    Compare multiple distance methods on the same F-matrix.

    Args:
        F_matrix: (M, M) F-matrix
        methods: List of method names, or None for all methods

    Returns:
        dict: {method_name: distance_matrix}
    """
    if methods is None:
        methods = ['1-F', '1/F', 'nei', 'sqrt', '-log']

    results = {}
    for method in methods:
        results[method] = compute_f_distance(F_matrix, method=method)

    return results
