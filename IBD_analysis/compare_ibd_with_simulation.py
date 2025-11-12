"""
Compare IBD_analysis F-matrix with numerical simulation results.

This script compares:
    1. IBD_analysis F-matrix (symbolic, evaluated at specific parameters)
    2. Numerical simulation F-matrix (from data/naive_simulation/raw/similarity_dot_*.npy)

Both represent the same quantity: F_ij = Probability(agent i and j share IBD)

The comparison directly evaluates:
    |F_theory - F_simulation|

where F_theory is from the symbolic IBD analysis and F_simulation is the
stationary average from numerical agent-based simulations.
"""

import sys
import os
import numpy as np
import argparse
from pathlib import Path

parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, parent_dir)

from sympy import symbols, lambdify
from IBD_analysis.src.f_matrix_symbolic import load_results_by_case


def get_simulation_dir_name(flow_type, nonzero_alpha, m, M, N, alpha):
    """
    Construct simulation directory name from parameters.

    Args:
        flow_type: "bidirectional_flow" or "outward_flow"
        nonzero_alpha: "nonzero_alpha_all" or "nonzero_alpha_center"
        m: coupling strength
        M: number of agents
        N: data pool size (N_i)
        alpha: innovation rate

    Returns:
        Directory name string
    """
    # Convert m to fraction format (e.g., 0.01 -> 0.01)
    m_str = f"{m:g}"
    alpha_str = f"{alpha:g}"

    return f"{flow_type}-{nonzero_alpha}_fr_{m_str}_agents_{M}_N_i_{N}_alpha_{alpha_str}"


def case_to_simulation_params(case_name):
    """
    Map case name to simulation parameters.

    Args:
        case_name: "case1", "case2", "case3", or "case4"

    Returns:
        (flow_type, nonzero_alpha)
    """
    case_map = {
        'case1': ('bidirectional_flow', 'nonzero_alpha_evenly'),
        'case2': ('outward_flow', 'nonzero_alpha_evenly'),
        'case3': ('bidirectional_flow', 'nonzero_alpha_center'),
        'case4': ('outward_flow', 'nonzero_alpha_center'),
    }
    return case_map[case_name]


def load_simulation_f_matrix(M, N, m, alpha, case_name, data_dir='data/naive_simulation/raw'):
    """
    Load simulation F-matrix (IBD probability) data from similarity_dot files.

    Args:
        M: number of agents
        N: data pool size
        m: coupling strength
        alpha: innovation rate
        case_name: case name
        data_dir: base directory for simulation data

    Returns:
        numpy array of shape (time_steps, M, M) containing F-matrices
    """
    from glob import glob
    import re

    flow_type, nonzero_alpha = case_to_simulation_params(case_name)
    dir_name = get_simulation_dir_name(flow_type, nonzero_alpha, m, M, N, alpha)

    sim_dir = Path(data_dir) / dir_name

    if not sim_dir.exists():
        raise FileNotFoundError(f"Simulation directory not found: {sim_dir}")

    # Find all similarity_dot_*.npy files
    similarity_files = sorted(glob(str(sim_dir / 'similarity_dot_*.npy')),
                             key=lambda x: int(re.search(r'similarity_dot_(\d+)\.npy', x).group(1)))

    if not similarity_files:
        raise FileNotFoundError(f"No similarity_dot files found in: {sim_dir}")

    # Load all F-matrices
    f_matrices = []
    for f in similarity_files:
        f_mat = np.load(f)
        f_matrices.append(f_mat)

    f_matrices = np.array(f_matrices)

    return f_matrices


def compute_stationary_f_matrix(f_matrices, burnin_fraction=0.5):
    """
    Compute stationary F-matrix from time series by averaging over latter half.

    Args:
        f_matrices: array of shape (time_steps, M, M)
        burnin_fraction: fraction of data to discard as burn-in

    Returns:
        Stationary F-matrix of shape (M, M)
    """
    n_steps = f_matrices.shape[0]
    burnin_steps = int(n_steps * burnin_fraction)

    # Average over stationary period
    stationary_f = f_matrices[burnin_steps:].mean(axis=0)

    return stationary_f


def evaluate_f_matrix(F_matrix, N, m, alpha):
    """
    Evaluate symbolic F-matrix at specific parameter values.

    Args:
        F_matrix: Symbolic matrix from IBD_analysis
        N: data pool size
        m: coupling strength
        alpha: innovation rate

    Returns:
        Numerical F-matrix as numpy array
    """
    N_sym, m_sym, alpha_sym = symbols('N m alpha')

    M = F_matrix.shape[0]
    F_numeric = np.zeros((M, M))

    for i in range(M):
        for j in range(M):
            f_expr = F_matrix[i, j]
            f_func = lambdify((N_sym, m_sym, alpha_sym), f_expr, 'numpy')
            F_numeric[i, j] = float(f_func(N, m, alpha))

    return F_numeric




def compare_ibd_with_simulation(M, N, m, alpha, case_name, verbose=True,
                                 burnin_fraction=0.5, data_dir='data/naive_simulation/raw'):
    """
    Compare IBD_analysis results with numerical simulation.

    Args:
        M: number of agents
        N: data pool size
        m: coupling strength
        alpha: innovation rate
        case_name: case name
        verbose: print detailed output
        burnin_fraction: fraction to discard as burn-in
        data_dir: base directory for simulation data

    Returns:
        Dictionary with comparison results
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Comparing IBD_analysis with Numerical Simulation")
        print(f"{'='*70}")
        print(f"Parameters: M={M}, N={N}, m={m}, α={alpha}, {case_name}")
        print(f"{'='*70}\n")

    # Load IBD_analysis results
    if verbose:
        print("Loading IBD_analysis results...")
    try:
        ibd_results = load_results_by_case(M, case_name)
        if verbose:
            print(f"  ✓ Loaded from: {ibd_results.get('file_path', 'N/A')}")
    except FileNotFoundError as e:
        print(f"Error: IBD_analysis results not found for M={M}, {case_name}")
        print(f"  {e}")
        return None

    F_matrix_sym = ibd_results['F_matrix']

    # Evaluate F-matrix numerically
    # Note: alpha is alpha_per_data, but IBD_analysis uses α_total = alpha_per_data * N
    alpha_total = alpha * N
    if verbose:
        print(f"\nEvaluating F-matrix at N={N}, m={m}, α_per_data={alpha} (α_total={alpha_total})...")
    F_theory = evaluate_f_matrix(F_matrix_sym, N, m, alpha_total)

    # Load simulation results
    if verbose:
        print(f"\nLoading simulation F-matrices...")
    try:
        f_matrices_sim = load_simulation_f_matrix(M, N, m, alpha, case_name, data_dir)
        if verbose:
            print(f"  ✓ Loaded F-matrix data with shape {f_matrices_sim.shape}")
            print(f"  Time steps: {f_matrices_sim.shape[0]}")
    except FileNotFoundError as e:
        print(f"Error: Simulation data not found")
        print(f"  {e}")
        return None

    # Compute stationary F-matrix from simulation
    if verbose:
        print(f"\nComputing stationary F-matrix (burn-in: {burnin_fraction*100}%)...")
    F_sim = compute_stationary_f_matrix(f_matrices_sim, burnin_fraction)

    # Compare results
    if verbose:
        print(f"\n{'='*70}")
        print("Comparison Results")
        print(f"{'='*70}\n")

        print("F-matrix from theory (IBD_analysis):")
        print(F_theory)
        print()

        print("F-matrix from simulation (stationary average):")
        print(F_sim)
        print()

    # Compute differences
    diff_f = np.abs(F_theory - F_sim)
    max_diff = np.max(diff_f)
    mean_diff = np.mean(diff_f)

    # Relative error (avoid division by zero)
    epsilon = 1e-10
    rel_error = np.abs(F_theory - F_sim) / (np.abs(F_sim) + epsilon)
    max_rel_error = np.max(rel_error)
    mean_rel_error = np.mean(rel_error)

    if verbose:
        print("Absolute difference (|F_theory - F_sim|):")
        print(diff_f)
        print(f"  Max: {max_diff:.6e}")
        print(f"  Mean: {mean_diff:.6e}")
        print()

        print("Relative error (|F_theory - F_sim| / F_sim):")
        print(rel_error)
        print(f"  Max: {max_rel_error:.6e}")
        print(f"  Mean: {mean_rel_error:.6e}")
        print()

    # Pairwise comparison
    if verbose:
        print(f"\n{'='*70}")
        print("Pairwise Comparison")
        print(f"{'='*70}\n")

        for i in range(M):
            for j in range(i+1, M):
                theory_val = F_theory[i, j]
                sim_val = F_sim[i, j]
                diff = abs(theory_val - sim_val)
                rel = diff / (abs(sim_val) + epsilon)

                status = "✓" if rel < 0.05 else "⚠" if rel < 0.1 else "✗"
                print(f"  F[{i},{j}]: Theory={theory_val:.6f}, Sim={sim_val:.6f}, "
                      f"Diff={diff:.6e}, RelErr={rel:.2%} {status}")

    # Summary
    if verbose:
        print(f"\n{'='*70}")
        print("Summary")
        print(f"{'='*70}\n")

        threshold = 0.05  # 5% relative error threshold
        if mean_rel_error < threshold:
            print(f"✓ GOOD AGREEMENT (mean rel. error {mean_rel_error:.2%} < {threshold:.0%})")
        elif mean_rel_error < 2 * threshold:
            print(f"⚠ MODERATE AGREEMENT (mean rel. error {mean_rel_error:.2%})")
        else:
            print(f"✗ POOR AGREEMENT (mean rel. error {mean_rel_error:.2%} >= {2*threshold:.0%})")
        print()

    return {
        'F_theory': F_theory,
        'F_simulation': F_sim,
        'F_timeseries': f_matrices_sim,
        'absolute_difference': diff_f,
        'relative_error': rel_error,
        'max_abs_diff': max_diff,
        'mean_abs_diff': mean_diff,
        'max_rel_error': max_rel_error,
        'mean_rel_error': mean_rel_error,
    }


def main():
    """Run comparison for specified parameters."""
    parser = argparse.ArgumentParser(
        description='Compare IBD_analysis with numerical simulation results'
    )

    # Required parameters
    parser.add_argument('--M', type=int, required=True,
                       help='Number of agents')
    parser.add_argument('--N', type=int, required=True,
                       help='Data pool size (N_i)')
    parser.add_argument('--m', type=float, required=True,
                       help='Coupling strength')
    parser.add_argument('--alpha', type=float, required=True,
                       help='Innovation rate per data point (alpha_per_data in simulation)')
    parser.add_argument('--case', type=str, nargs='+', required=True,
                       help='Model case(s): case1, case2, case3, case4, or "all" for all cases')

    # Optional parameters
    parser.add_argument('--burnin', type=float, default=0.5,
                       help='Burn-in fraction for stationary average (default: 0.5)')
    parser.add_argument('--data-dir', type=str,
                       default='data/naive_simulation/raw',
                       help='Base directory for simulation data')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Determine which cases to run
    all_cases = ['case1', 'case2', 'case3', 'case4']
    if 'all' in args.case:
        cases_to_run = all_cases
    else:
        cases_to_run = []
        for case in args.case:
            if case not in all_cases:
                print(f"Error: Invalid case '{case}'. Must be one of: {all_cases} or 'all'")
                return 1
            cases_to_run.append(case)

    # Run comparison for each case
    results = {}
    failed_cases = []

    for case_name in cases_to_run:
        result = compare_ibd_with_simulation(
            M=args.M,
            N=args.N,
            m=args.m,
            alpha=args.alpha,
            case_name=case_name,
            verbose=args.verbose,
            burnin_fraction=args.burnin,
            data_dir=args.data_dir
        )

        if result is None:
            failed_cases.append(case_name)
        else:
            results[case_name] = result

    # Print summary if multiple cases were run
    if len(cases_to_run) > 1:
        print(f"\n{'='*70}")
        print("Overall Summary")
        print(f"{'='*70}\n")

        for case_name in cases_to_run:
            if case_name in results:
                result = results[case_name]
                mean_rel_err = result['mean_rel_error']
                max_rel_err = result['max_rel_error']

                if mean_rel_err < 0.05:
                    status = "✓ GOOD"
                elif mean_rel_err < 0.1:
                    status = "⚠ MODERATE"
                else:
                    status = "✗ POOR"

                print(f"{case_name}: {status} (mean rel. error: {mean_rel_err:.2%}, max: {max_rel_err:.2%})")
            else:
                print(f"{case_name}: ✗ FAILED (missing data)")

        print()

        if failed_cases:
            print(f"Failed cases: {', '.join(failed_cases)}")
            return 1

        # Return success if all cases have reasonable agreement
        all_good = all(results[case]['mean_rel_error'] < 0.1 for case in results)
        return 0 if all_good else 1
    else:
        # Single case - use original behavior
        if failed_cases:
            print("\nComparison failed due to missing data.")
            return 1

        result = results[cases_to_run[0]]
        if result['mean_rel_error'] < 0.1:  # 10% threshold
            return 0
        else:
            return 1


if __name__ == "__main__":
    sys.exit(main())
