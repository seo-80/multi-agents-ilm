"""
Concentric distribution analysis for IBD F-matrix.

Analyzes whether parameter combinations lead to concentric distributions
using symbolic F-matrix solutions.
"""

import numpy as np
import sys
import os

# Add paths
parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

from sympy import symbols, lambdify
from IBD_analysis.src.f_matrix_symbolic import load_results_by_case, get_case_name
from IBD_analysis.src.distance_metrics import compute_f_distance
from ilm import networks


def is_concentric_distribution(distance_matrix):
    """
    Check if the distance matrix shows a concentric distribution.

    A distribution is concentric if there exists a base agent (not center)
    that is closer to an agent on the opposite side than to the center.

    Args:
        distance_matrix: (M, M) distance matrix

    Returns:
        bool: True if concentric distribution detected
    """
    center = len(distance_matrix) // 2

    for base in range(len(distance_matrix)):
        if base == center:
            continue

        for reference in range(len(distance_matrix)):
            # Check if reference is on opposite side from base
            is_opposite_side = (base - center) * (reference - center) < 0

            # If base is closer to opposite-side reference than to center
            if is_opposite_side and distance_matrix[base][reference] < distance_matrix[base][center]:
                return True

    return False


def evaluate_f_matrix_symbolic(M, case_name, N_val, m_val, alpha_val):
    """
    Evaluate symbolic F-matrix at specific parameter values.

    Args:
        M: Number of agents
        case_name: "case1", "case2", "case3", or "case4"
        N_val, m_val, alpha_val: Parameter values

    Returns:
        F_matrix: (M, M) numpy array with evaluated F values

    Raises:
        FileNotFoundError: If symbolic results for M and case_name not found
    """
    # Load symbolic results
    results = load_results_by_case(M, case_name)

    F_symbolic = results['F_matrix']

    # Define symbolic variables
    N_sym, m_sym, alpha_sym = symbols('N m alpha')

    # Evaluate each element
    F_evaluated = np.zeros((M, M), dtype=float)
    for i in range(M):
        for j in range(M):
            expr = F_symbolic[i, j]
            f_lambda = lambdify((N_sym, m_sym, alpha_sym), expr, 'numpy')
            F_evaluated[i, j] = float(f_lambda(N_val, m_val, alpha_val))

    return F_evaluated


def evaluate_f_matrix_numerical(M, case_name, N_val, m_val, alpha_val,
                                max_iter=50000, tol=1e-6):
    """
    Compute F-matrix numerically for cases where symbolic solution unavailable.

    Args:
        M: Number of agents
        case_name: "case1", "case2", "case3", or "case4"
        N_val, m_val, alpha_val: Parameter values
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        F_matrix: (M, M) numpy array
    """
    # Parse case_name to get boolean flags
    case_configs = {
        'case1': (False, False),
        'case2': (True, False),
        'case3': (False, True),
        'case4': (True, True),
    }
    center_prestige, centralized_neologism_creation = case_configs[case_name]

    # Build W matrix
    if not center_prestige:
        network_args = {"bidirectional_flow_rate": m_val}
    else:
        network_args = {"outward_flow_rate": m_val}

    W = networks.network(M, args=network_args)

    # Build alpha vector
    if not centralized_neologism_creation:
        alphas = np.ones(M) * alpha_val
    else:
        alphas = np.zeros(M)
        alphas[M // 2] = alpha_val

    # Compute mutation rates
    mu = alphas / (N_val + alphas)

    # Update function
    def update_f(f, W, mu, N):
        """F-matrix update (simplified form without drift)."""
        D = np.diag(1 - mu)
        G = W @ D @ f @ D @ W.T

        # Off-diagonal: F_ij = G_ij
        # Diagonal: F_ii = 1/N + (1-1/N)*G_ii
        f_new = G.copy()
        np.fill_diagonal(f_new, 1/N + (1 - 1/N) * np.diag(G))

        return f_new

    # Run until convergence
    f = np.eye(M)
    for _ in range(max_iter):
        f_new = update_f(f, W, mu, N_val)
        diff = np.max(np.abs(f_new - f))
        if diff < tol:
            return f_new
        f = f_new

    print(f"Warning: Did not converge within {max_iter} iterations (diff={diff:.2e})")
    return f


def analyze_concentric_for_parameters(N, m, alpha, M=3,
                                     center_prestige=False,
                                     centralized_neologism_creation=False,
                                     distance_method='nei',
                                     use_symbolic=True,
                                     verbose=False):
    """
    Analyze if parameters lead to concentric distribution.

    Args:
        N, m, alpha: Parameter values
        M: Number of agents (must be odd)
        center_prestige: Center-outward asymmetric model
        centralized_neologism_creation: Only center creates innovations
        distance_method: '1-F', 'nei', 'sqrt', '-log'
        use_symbolic: If True, use symbolic solution (requires pre-computed results).
                     If False, use numerical computation.
        verbose: Print debug information

    Returns:
        dict with:
            - 'is_concentric': bool
            - 'F_matrix': (M, M) array
            - 'distance_matrix': (M, M) array
            - 'parameters': dict
            - 'method_used': 'symbolic' or 'numerical'
    """
    if M % 2 == 0:
        raise ValueError("M must be odd")

    case_name = get_case_name(center_prestige, centralized_neologism_creation)

    # Use symbolic or numerical based on flag
    if use_symbolic:
        F_matrix = evaluate_f_matrix_symbolic(M, case_name, N, m, alpha)
        method_used = 'symbolic'
        if verbose:
            print(f"  Used symbolic solution for M={M}, {case_name}")
    else:
        F_matrix = evaluate_f_matrix_numerical(M, case_name, N, m, alpha)
        method_used = 'numerical'

    # Compute distance
    distance_matrix = compute_f_distance(F_matrix, method=distance_method)

    # Check concentric
    is_concentric = is_concentric_distribution(distance_matrix)

    return {
        'is_concentric': is_concentric,
        'F_matrix': F_matrix,
        'distance_matrix': distance_matrix,
        'parameters': {
            'N': N,
            'm': m,
            'alpha': alpha,
            'M': M,
            'center_prestige': center_prestige,
            'centralized_neologism_creation': centralized_neologism_creation,
        },
        'distance_method': distance_method,
        'method_used': method_used,
    }


def test_single_parameter():
    """Test function for single parameter combination."""
    result = analyze_concentric_for_parameters(
        N=100, m=0.01, alpha=0.001,
        M=3,
        center_prestige=False,
        centralized_neologism_creation=False,
        distance_method='nei',
        use_symbolic=True,
        verbose=True
    )

    print("\nTest Result:")
    print(f"  Is concentric: {result['is_concentric']}")
    print(f"  Method used: {result['method_used']}")
    print(f"  F-matrix diagonal: {np.diag(result['F_matrix'])}")
    print(f"  Distance matrix:")
    print(result['distance_matrix'])

    return result


if __name__ == "__main__":
    test_single_parameter()
