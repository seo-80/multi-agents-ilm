"""
Numerical Verification of Symbolic F-Matrix Solutions

This script verifies that the symbolic F-matrix solutions match numerical
computations from the existing probability_of_identity.py implementation.
"""

import numpy as np
import sys
import os

# Add parent directory to path to import modules
parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

from IBD_analysis.src.f_matrix_symbolic import load_results_by_case
from ilm import networks
from sympy import symbols, lambdify


def update_f_numerical(f, W, mu, N):
    """
    Numerical F-matrix update using the NEW equations.

    For i ≠ j:
        F_ij = Σ_k Σ_l W_ik W_jl (1-μ_k)(1-μ_l) F_kl

    For i = j:
        F_ii = 1/N + (1-1/N) Σ_k Σ_l W_ik W_il (1-μ_k)(1-μ_l) F_kl

    This is different from probability_of_identity.py which includes drift term.
    """
    M = len(mu)
    f_new = np.zeros_like(f)

    # Compute using matrix operations for efficiency
    D = np.diag(1 - mu)
    G = W @ D @ f @ D @ W.T

    # Off-diagonal: F_ij = G_ij
    f_new = G.copy()

    # Diagonal: F_ii = 1/N + (1-1/N)*G_ii
    np.fill_diagonal(f_new, 1/N + (1 - 1/N) * np.diag(G))

    return f_new


def run_until_convergence(f_init, W, mu, N, tol=1e-10, max_iter=10000):
    """Run F-matrix update until convergence."""
    f = f_init.copy()
    for i in range(max_iter):
        f_new = update_f_numerical(f, W, mu, N)
        diff = np.max(np.abs(f_new - f))
        if diff < tol:
            return f_new, i+1
        f = f_new
    print(f"Warning: Did not converge within {max_iter} iterations. Final diff={diff}")
    return f, max_iter


def verify_case(M, case_name, N_val=100.0, m_val=0.01, alpha_val=0.001, verbose=True):
    """
    Verify a specific case by comparing symbolic and numerical solutions.

    Args:
        M: Number of agents
        case_name: "case1", "case2", "case3", or "case4"
        N_val: Numerical value for population size
        m_val: Numerical value for coupling strength
        alpha_val: Numerical value for innovation parameter
        verbose: Print detailed output

    Returns:
        Boolean indicating if verification passed
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Verifying M={M}, {case_name}")
        print(f"Parameters: N={N_val}, m={m_val}, α={alpha_val}")
        print(f"{'='*70}\n")

    # Load symbolic results
    try:
        results = load_results_by_case(M, case_name)
    except FileNotFoundError:
        print(f"Error: Results file for M={M}, {case_name} not found.")
        print("Please run f_matrix_symbolic.py first to generate results.")
        return False

    F_symbolic = results['F_matrix']
    metadata = results['metadata']
    center_prestige = metadata['center_prestige']
    centralized_neologism_creation = metadata['centralized_neologism_creation']

    if verbose:
        print("Symbolic solution loaded successfully.")

    # Build W matrix numerically
    if not center_prestige:
        # Symmetric bidirectional
        network_args = {"bidirectional_flow_rate": m_val}
    else:
        # Center-outward asymmetric
        network_args = {"outward_flow_rate": m_val}

    W_numerical = networks.network(M, args=network_args)

    # Build alpha vector
    if not centralized_neologism_creation:
        alphas = np.ones(M) * alpha_val
    else:
        alphas = np.zeros(M)
        alphas[M // 2] = alpha_val  # Center agent

    # Compute mutation rates
    mu_numerical = alphas / (N_val + alphas)

    if verbose:
        print("Running numerical computation...")

    # Compute numerical solution
    f_init = np.eye(M)
    F_numerical, n_iters = run_until_convergence(f_init, W_numerical, mu_numerical, N_val)

    if verbose:
        print(f"Numerical computation converged in {n_iters} iterations.")

    # Evaluate symbolic solution at parameter values
    if verbose:
        print("Evaluating symbolic solution at parameter values...")

    N_sym, m_sym, alpha_sym = symbols('N m alpha')

    F_evaluated = np.zeros((M, M), dtype=float)
    for i in range(M):
        for j in range(M):
            # Convert symbolic expression to numerical function
            expr = F_symbolic[i, j]
            f_lambda = lambdify((N_sym, m_sym, alpha_sym), expr, 'numpy')
            F_evaluated[i, j] = float(f_lambda(N_val, m_val, alpha_val))

    # Compare
    diff = np.abs(F_numerical - F_evaluated)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    if verbose:
        print(f"\nComparison results:")
        print(f"  Maximum difference: {max_diff:.2e}")
        print(f"  Mean difference:    {mean_diff:.2e}")
        print(f"\nF_numerical (first 3x3):")
        print(F_numerical[:min(3,M), :min(3,M)])
        print(f"\nF_symbolic evaluated (first 3x3):")
        print(F_evaluated[:min(3,M), :min(3,M)])
        print(f"\nDifference (first 3x3):")
        print(diff[:min(3,M), :min(3,M)])

    # Check if verification passed
    threshold = 1e-6
    passed = max_diff < threshold

    if passed:
        print(f"\n✓ Verification PASSED (max diff {max_diff:.2e} < {threshold:.2e})")
    else:
        print(f"\n✗ Verification FAILED (max diff {max_diff:.2e} >= {threshold:.2e})")

    return passed


def main():
    """Run verification for all available cases."""
    import argparse

    parser = argparse.ArgumentParser(description='Verify symbolic F-matrix solutions numerically')
    parser.add_argument('--M', type=int, nargs='+', default=[3],
                       help='Agent counts to verify, e.g., --M 3 5')
    parser.add_argument('--cases', type=str, nargs='+',
                       default=['case1', 'case2', 'case3', 'case4'],
                       help='Cases to verify, e.g., --cases case1 case2')
    parser.add_argument('--N', type=float, default=100.0,
                       help='Population size for numerical evaluation')
    parser.add_argument('--m', type=float, default=0.01,
                       help='Coupling strength for numerical evaluation')
    parser.add_argument('--alpha', type=float, default=0.001,
                       help='Innovation parameter for numerical evaluation')

    args = parser.parse_args()

    results = {}

    for M in args.M:
        for case_name in args.cases:
            key = f"M{M}_{case_name}"
            passed = verify_case(M, case_name, args.N, args.m, args.alpha, verbose=True)
            results[key] = passed

    # Summary
    print(f"\n{'='*70}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*70}\n")

    for key, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {key:20s} {status}")

    print()

    all_passed = all(results.values())
    if all_passed:
        print("All verifications passed!")
        return 0
    else:
        print("Some verifications failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
