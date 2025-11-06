"""
Compare IBD_analysis F-matrix with symbolic_analysis expected distances.

Relationship:
    E[d_ij] = 1 - F_ij

where:
    E[d_ij]: Expected distance from symbolic_analysis (different data probability)
    F_ij: IBD probability from IBD_analysis (same ancestry probability)

For N=1, the mutation rate definitions match:
    μ_i = α_i/(1 + α_i)
"""

import sys
import os

parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'symbolic_analysis', 'src'))

from sympy import symbols, simplify, lambdify, Abs
import numpy as np
from IBD_analysis.src.f_matrix_symbolic import load_results_by_case as load_ibd_results
from m_agent_stationary_symbolic import load_results_by_case as load_symbolic_results
from analyze_distances import compute_distance_expectations


def compare_ibd_with_symbolic(M, case_name, verbose=True):
    """
    Compare F-matrix from IBD_analysis with 1-distance from symbolic_analysis.

    Args:
        M: Number of agents
        case_name: "case1", "case2", "case3", or "case4"
        verbose: Print detailed output

    Returns:
        Boolean indicating if comparison passed
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Comparing IBD_analysis vs symbolic_analysis")
        print(f"M={M}, {case_name}")
        print(f"{'='*70}\n")

    # Load IBD_analysis results
    if verbose:
        print("Loading IBD_analysis results...")
    try:
        ibd_results = load_ibd_results(M, case_name)
    except FileNotFoundError:
        print(f"Error: IBD_analysis results for M={M}, {case_name} not found.")
        return False

    F_matrix = ibd_results['F_matrix']

    # Load symbolic_analysis results
    if verbose:
        print("Loading symbolic_analysis results...")
    try:
        sym_results = load_symbolic_results(M, case_name)
    except FileNotFoundError:
        print(f"Error: symbolic_analysis results for M={M}, {case_name} not found.")
        return False

    states = sym_results['states']
    pi = sym_results['pi']

    # Compute expected distances
    if verbose:
        print("Computing expected distances from symbolic_analysis...")
    expected_distances = compute_distance_expectations(states, pi, M)

    # Define symbolic variables
    N_sym, m_sym, alpha_sym = symbols('N m alpha')

    if verbose:
        print("\nComparing symbolic expressions (N=1 case)...\n")

    # Compare for all pairs
    max_diff_expr = 0
    all_match = True

    comparison_results = []

    for i in range(1, M + 1):
        for j in range(i + 1, M + 1):
            # Get F_ij from IBD_analysis (0-indexed)
            f_ij_expr = F_matrix[i-1, j-1]

            # Evaluate at N=1
            f_ij_at_N1 = f_ij_expr.subs(N_sym, 1)
            f_ij_at_N1 = simplify(f_ij_at_N1)

            # Get E[d_ij] from symbolic_analysis
            e_d_ij = expected_distances[(i, j)]

            # Compare: E[d_ij] should equal 1 - F_ij
            one_minus_f = simplify(1 - f_ij_at_N1)

            # Symbolic difference
            diff_expr = simplify(e_d_ij - one_minus_f)

            # Test numerical evaluation to check if symbolically equal
            m_val, alpha_val = 0.01, 0.001
            try:
                diff_eval = diff_expr.subs({m_sym: m_val, alpha_sym: alpha_val}).n()
                if diff_eval == 0:
                    diff_num = 0.0
                else:
                    diff_num = abs(float(diff_eval))
            except (TypeError, AttributeError, ValueError):
                # If still cannot convert, assume no match and print debug info
                if verbose:
                    print(f"    Warning: Cannot numerically evaluate difference")
                    print(f"    diff_expr type: {type(diff_expr)}")
                    print(f"    diff_expr: {diff_expr}")
                diff_num = 1.0  # Assume mismatch

            if diff_num > 1e-10:
                all_match = False
                if verbose:
                    print(f"  ({i},{j}): MISMATCH (diff = {diff_num:.2e})")
                    print(f"    E[d_{{{i}{j}}}] = {e_d_ij}")
                    print(f"    1-F_{{{i}{j}}}(N=1) = {one_minus_f}")
                    print(f"    Difference = {diff_expr}")
            else:
                if verbose:
                    print(f"  ({i},{j}): MATCH ✓ (diff = {diff_num:.2e})")

            comparison_results.append({
                'pair': (i, j),
                'E[d_ij]': e_d_ij,
                '1-F_ij': one_minus_f,
                'diff_symbolic': diff_expr,
                'diff_numeric': diff_num,
                'match': diff_num < 1e-10
            })

    # Numerical verification for specific parameter values
    if verbose:
        print(f"\n{'='*70}")
        print("Numerical verification (m=0.01, α=0.001, N=1)")
        print(f"{'='*70}\n")

    m_val, alpha_val, N_val = 0.01, 0.001, 1.0

    max_numeric_diff = 0

    for i in range(1, M + 1):
        for j in range(i + 1, M + 1):
            # F_ij from IBD_analysis
            f_ij_expr = F_matrix[i-1, j-1]
            f_ij_func = lambdify((N_sym, m_sym, alpha_sym), f_ij_expr, 'numpy')
            f_ij_val = float(f_ij_func(N_val, m_val, alpha_val))

            # E[d_ij] from symbolic_analysis
            e_d_ij_expr = expected_distances[(i, j)]
            e_d_ij_func = lambdify((m_sym, alpha_sym), e_d_ij_expr, 'numpy')
            e_d_ij_val = float(e_d_ij_func(m_val, alpha_val))

            # Compare
            one_minus_f_val = 1.0 - f_ij_val
            diff = abs(e_d_ij_val - one_minus_f_val)

            max_numeric_diff = max(max_numeric_diff, diff)

            if verbose:
                status = "✓" if diff < 1e-6 else "✗"
                print(f"  ({i},{j}): E[d] = {e_d_ij_val:.8f}, 1-F = {one_minus_f_val:.8f}, diff = {diff:.2e} {status}")

    # Summary
    if verbose:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}\n")
        print(f"  Maximum numerical difference: {max_numeric_diff:.2e}")
        print(f"  All symbolic expressions match: {all_match}")
        print()

    threshold = 1e-6
    passed = max_numeric_diff < threshold

    if passed:
        print(f"✓ VERIFICATION PASSED (max diff {max_numeric_diff:.2e} < {threshold:.2e})")
        print(f"  Confirmed: E[d_ij] = 1 - F_ij(N=1)")
    else:
        print(f"✗ VERIFICATION FAILED (max diff {max_numeric_diff:.2e} >= {threshold:.2e})")

    return passed


def main():
    """Run comparison for all available cases."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Compare IBD_analysis F-matrix with symbolic_analysis distances'
    )
    parser.add_argument('--M', type=int, nargs='+', default=[3],
                       help='Agent counts to compare, e.g., --M 3 5')
    parser.add_argument('--cases', type=str, nargs='+',
                       default=['case1', 'case2', 'case3', 'case4'],
                       help='Cases to compare, e.g., --cases case1 case2')

    args = parser.parse_args()

    results = {}

    for M in args.M:
        for case_name in args.cases:
            key = f"M{M}_{case_name}"
            passed = compare_ibd_with_symbolic(M, case_name, verbose=True)
            results[key] = passed

    # Summary
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}\n")

    for key, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {key:20s} {status}")

    print()

    all_passed = all(results.values())
    if all_passed:
        print("All comparisons passed!")
        print("Confirmed: E[d_ij] = 1 - F_ij(N=1) for all cases.")
        return 0
    else:
        print("Some comparisons failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
