#!/usr/bin/env python
"""
Compare symbolic F-matrix solutions with Monte Carlo simulation results.

This script loads:
1. Symbolic F-matrix solutions from IBD_analysis/results/M{M}_{case}.pkl
2. Simulation results from data/raw/{subdir}/mean_f_matrix.npy

And compares them to validate the symbolic solutions.
"""

import os
import sys
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional
import pandas as pd

# Add IBD_analysis to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'IBD_analysis'))
from src.f_matrix_symbolic import load_results

# Try to import config, use default if not available
try:
    from config import get_data_raw_dir
except ModuleNotFoundError:
    def get_data_raw_dir():
        """Default data directory fallback."""
        return os.path.join(os.path.dirname(__file__), 'data', 'raw')


# ============================================================================
# Parameter Mapping
# ============================================================================

def get_case_parameters(case_name: str) -> Tuple[str, str]:
    """
    Map case name to simulation parameters.

    Args:
        case_name: One of 'case1', 'case2', 'case3', 'case4'

    Returns:
        (flow_type, nonzero_alpha) tuple

    Mapping:
        case1 (center_prestige=True, centralized_neologism=True):
            -> flow_type="outward", nonzero_alpha="center"
        case2 (center_prestige=True, centralized_neologism=False):
            -> flow_type="outward", nonzero_alpha="evenly"
        case3 (center_prestige=False, centralized_neologism=True):
            -> flow_type="bidirectional", nonzero_alpha="center"
        case4 (center_prestige=False, centralized_neologism=False):
            -> flow_type="bidirectional", nonzero_alpha="evenly"
    """
    case_map = {
        'case1': ('outward', 'center'),
        'case2': ('outward', 'evenly'),
        'case3': ('bidirectional', 'center'),
        'case4': ('bidirectional', 'evenly'),
    }

    if case_name not in case_map:
        raise ValueError(f"Unknown case: {case_name}. Must be one of {list(case_map.keys())}")

    return case_map[case_name]


def get_simulation_dir(M: int, N_i: int, coupling_strength: float,
                       alpha_per_data: float, flow_type: str,
                       nonzero_alpha: str,
                       data_raw_dir: Optional[str] = None) -> str:
    """
    Construct the simulation data directory path.

    Args:
        M: Number of agents
        N_i: Data per agent
        coupling_strength: Coupling strength (m)
        alpha_per_data: Alpha per data point
        flow_type: 'outward' or 'bidirectional'
        nonzero_alpha: 'center' or 'evenly'
        data_raw_dir: Base data directory (default: from config)

    Returns:
        Path to simulation directory
    """
    if data_raw_dir is None:
        data_raw_dir = get_data_raw_dir()

    # Construct subdirectory name (matches naive_simulation.py logic)
    if flow_type == "bidirectional":
        flow_prefix = "bidirectional_flow-"
    else:
        flow_prefix = "outward_flow-"

    subdir = f"{flow_prefix}nonzero_alpha_{nonzero_alpha}_fr_{coupling_strength}_agents_{M}_N_i_{N_i}_alpha_{alpha_per_data}"

    return os.path.join(data_raw_dir, subdir)


# ============================================================================
# Loading Functions
# ============================================================================

def load_symbolic_f_matrix(M: int, case_name: str,
                           m_val: float, alpha_val: float, N_val: int,
                           results_dir: Optional[str] = None) -> np.ndarray:
    """
    Load symbolic F-matrix and evaluate it numerically.

    Args:
        M: Number of agents
        case_name: Case identifier ('case1', 'case2', 'case3', 'case4')
        m_val: Coupling strength value
        alpha_val: Alpha value (total alpha for central agent or per-agent)
        N_val: Data count N
        results_dir: Results directory (default: IBD_analysis/results)

    Returns:
        M×M numpy array of F-matrix values
    """
    if results_dir is None:
        results_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'IBD_analysis', 'results'
        )

    # Load symbolic results
    filepath = os.path.join(results_dir, f"M{M}_{case_name}.pkl")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Symbolic solution not found: {filepath}\n"
            f"Run: python IBD_analysis/analyze_concentric_parameter_space.py "
            f"--M {M} --cases {case_name} --precompute-symbolic --no-plots"
        )

    results = load_results(filepath)
    F_symbolic = results['F_matrix']

    # Get symbolic variables
    from sympy import symbols
    m, alpha, N = symbols('m alpha N')

    # Substitute numerical values
    F_numerical = F_symbolic.subs({m: m_val, alpha: alpha_val, N: N_val})

    # Convert to numpy array - evaluate each element to float
    M_size = F_numerical.shape[0]
    F_array = np.zeros((M_size, M_size), dtype=float)

    for i in range(M_size):
        for j in range(M_size):
            # Evaluate the symbolic expression to a numerical value
            val = F_numerical[i, j]
            # Use evalf() to evaluate, then convert to float
            F_array[i, j] = float(val.evalf())

    return F_array


def load_simulation_f_matrix(sim_dir: str) -> np.ndarray:
    """
    Load time-averaged F-matrix from simulation results.

    Args:
        sim_dir: Simulation directory containing mean_f_matrix.npy

    Returns:
        M×M numpy array of mean F-matrix values
    """
    f_matrix_path = os.path.join(sim_dir, "mean_f_matrix.npy")

    if not os.path.exists(f_matrix_path):
        raise FileNotFoundError(
            f"Simulation F-matrix not found: {f_matrix_path}\n"
            f"Run simulation first, or use --plot option to generate mean_f_matrix.npy:\n"
            f"  ./run_naive_simulations_parallel.sh -p <param> -m <max_t>\n"
            f"  ./run_naive_simulations_parallel.sh -P  # Generate plots and averages"
        )

    F_sim = np.load(f_matrix_path)
    return F_sim


# ============================================================================
# Comparison Functions
# ============================================================================

def compare_matrices(F_symbolic: np.ndarray, F_simulation: np.ndarray,
                     case_name: str, params: Dict) -> Dict:
    """
    Compare symbolic and simulation F-matrices.

    Args:
        F_symbolic: Symbolic F-matrix (M×M)
        F_simulation: Simulation F-matrix (M×M)
        case_name: Case identifier
        params: Parameter dictionary

    Returns:
        Dictionary with comparison statistics
    """
    M = F_symbolic.shape[0]

    # Compute differences
    diff = F_simulation - F_symbolic
    abs_diff = np.abs(diff)

    # Avoid division by zero for relative error
    relative_error = np.zeros_like(abs_diff)
    mask = F_symbolic > 1e-10
    relative_error[mask] = abs_diff[mask] / F_symbolic[mask]

    # Statistics
    stats = {
        'case': case_name,
        'M': M,
        'params': params,
        'mean_absolute_error': np.mean(abs_diff),
        'max_absolute_error': np.max(abs_diff),
        'mean_relative_error': np.mean(relative_error[mask]) if mask.any() else np.nan,
        'max_relative_error': np.max(relative_error[mask]) if mask.any() else np.nan,
        'rmse': np.sqrt(np.mean(diff**2)),
        'correlation': np.corrcoef(F_symbolic.flatten(), F_simulation.flatten())[0, 1],
        'F_symbolic': F_symbolic,
        'F_simulation': F_simulation,
        'difference': diff,
        'absolute_difference': abs_diff,
        'relative_error': relative_error,
    }

    return stats


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_comparison(stats: Dict, output_dir: str):
    """
    Create comprehensive comparison plots.

    Args:
        stats: Comparison statistics dictionary
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)

    case = stats['case']
    M = stats['M']
    F_sym = stats['F_symbolic']
    F_sim = stats['F_simulation']
    diff = stats['difference']
    abs_diff = stats['absolute_difference']
    rel_error = stats['relative_error']

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))

    # 1. Symbolic F-matrix heatmap
    ax1 = plt.subplot(2, 4, 1)
    sns.heatmap(F_sym, annot=True, fmt='.4f', cmap='Blues',
                cbar_kws={'label': 'F-matrix value'}, ax=ax1)
    ax1.set_title(f'Symbolic F-matrix\n{case}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Agent j')
    ax1.set_ylabel('Agent i')

    # 2. Simulation F-matrix heatmap
    ax2 = plt.subplot(2, 4, 2)
    sns.heatmap(F_sim, annot=True, fmt='.4f', cmap='Blues',
                cbar_kws={'label': 'F-matrix value'}, ax=ax2)
    ax2.set_title(f'Simulation F-matrix (mean)\n{case}', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Agent j')
    ax2.set_ylabel('Agent i')

    # 3. Absolute difference heatmap
    ax3 = plt.subplot(2, 4, 3)
    sns.heatmap(abs_diff, annot=True, fmt='.4f', cmap='Reds',
                cbar_kws={'label': 'Absolute difference'}, ax=ax3)
    ax3.set_title(f'Absolute Difference\n|Sim - Symbolic|', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Agent j')
    ax3.set_ylabel('Agent i')

    # 4. Relative error heatmap
    ax4 = plt.subplot(2, 4, 4)
    mask = F_sym > 1e-10
    rel_error_display = np.where(mask, rel_error, np.nan)
    sns.heatmap(rel_error_display, annot=True, fmt='.4f', cmap='Oranges',
                cbar_kws={'label': 'Relative error'}, ax=ax4)
    ax4.set_title(f'Relative Error\n|Sim - Symbolic| / Symbolic', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Agent j')
    ax4.set_ylabel('Agent i')

    # 5. Scatter plot: Simulation vs Symbolic
    ax5 = plt.subplot(2, 4, 5)
    ax5.scatter(F_sym.flatten(), F_sim.flatten(), alpha=0.6, s=50)

    # Add diagonal line
    min_val = min(F_sym.min(), F_sim.min())
    max_val = max(F_sym.max(), F_sim.max())
    ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect match')

    ax5.set_xlabel('Symbolic F-matrix', fontsize=11)
    ax5.set_ylabel('Simulation F-matrix', fontsize=11)
    ax5.set_title(f'Scatter Plot\nCorr: {stats["correlation"]:.4f}', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.axis('equal')

    # 6. Error distribution histogram
    ax6 = plt.subplot(2, 4, 6)
    ax6.hist(diff.flatten(), bins=30, edgecolor='black', alpha=0.7)
    ax6.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax6.set_xlabel('Error (Simulation - Symbolic)', fontsize=11)
    ax6.set_ylabel('Frequency', fontsize=11)
    ax6.set_title(f'Error Distribution\nMean: {stats["mean_absolute_error"]:.6f}',
                  fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. Distance from center agent
    ax7 = plt.subplot(2, 4, 7)
    center_idx = M // 2
    ax7.plot(F_sym[center_idx, :], 'o-', label='Symbolic', linewidth=2, markersize=8)
    ax7.plot(F_sim[center_idx, :], 's-', label='Simulation', linewidth=2, markersize=8)
    ax7.set_xlabel('Agent j', fontsize=11)
    ax7.set_ylabel(f'F[{center_idx}, j]', fontsize=11)
    ax7.set_title(f'F-matrix from Center Agent {center_idx}', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_xticks(range(M))

    # 8. Statistics summary text
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')

    summary_text = f"""
Comparison Statistics
{'='*40}

Case: {case}
M (agents): {M}

Parameters:
  m (coupling): {stats['params']['m']:.4f}
  α (alpha): {stats['params']['alpha']:.4f}
  N (data/agent): {stats['params']['N']}
  Flow type: {stats['params']['flow_type']}
  Alpha dist: {stats['params']['nonzero_alpha']}

Error Metrics:
  Mean Abs Error: {stats['mean_absolute_error']:.6f}
  Max Abs Error: {stats['max_absolute_error']:.6f}
  RMSE: {stats['rmse']:.6f}

  Mean Rel Error: {stats['mean_relative_error']:.4%}
  Max Rel Error: {stats['max_relative_error']:.4%}

  Correlation: {stats['correlation']:.6f}
"""

    ax8.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center', transform=ax8.transAxes)

    plt.tight_layout()

    # Save figure
    filename = f"comparison_M{M}_{case}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved comparison plot: {filepath}")


def create_summary_table(all_stats: list, output_dir: str):
    """
    Create summary table comparing all cases.

    Args:
        all_stats: List of comparison statistics dictionaries
        output_dir: Output directory
    """
    # Create DataFrame
    rows = []
    for stats in all_stats:
        rows.append({
            'Case': stats['case'],
            'M': stats['M'],
            'm': stats['params']['m'],
            'alpha': stats['params']['alpha'],
            'N': stats['params']['N'],
            'Flow': stats['params']['flow_type'],
            'Alpha_dist': stats['params']['nonzero_alpha'],
            'MAE': stats['mean_absolute_error'],
            'Max_AE': stats['max_absolute_error'],
            'RMSE': stats['rmse'],
            'Mean_Rel_Err': stats['mean_relative_error'],
            'Max_Rel_Err': stats['max_relative_error'],
            'Correlation': stats['correlation'],
        })

    df = pd.DataFrame(rows)

    # Save to CSV
    csv_path = os.path.join(output_dir, 'comparison_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved summary table: {csv_path}")

    # Print to console
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compare symbolic F-matrix solutions with Monte Carlo simulations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare single case with default parameters
  python compare_symbolic_simulation.py --M 5 --cases case1

  # Compare all cases for M=5
  python compare_symbolic_simulation.py --M 5 --cases case1 case2 case3 case4

  # Custom parameters
  python compare_symbolic_simulation.py --M 7 --cases case1 --m 0.01 --alpha 0.001 --N 99

  # Custom output directory
  python compare_symbolic_simulation.py --M 5 --cases case1 --output-dir my_results
"""
    )

    # Required arguments
    parser.add_argument('--M', type=int, required=True,
                       help='Number of agents (must match symbolic solution)')
    parser.add_argument('--cases', nargs='+', required=True,
                       choices=['case1', 'case2', 'case3', 'case4'],
                       help='Case(s) to compare')

    # Parameter arguments
    parser.add_argument('--m', '--coupling-strength', type=float, default=0.01,
                       dest='coupling_strength',
                       help='Coupling strength (default: 0.01)')
    parser.add_argument('--alpha', '--alpha-per-data', type=float, default=0.001,
                       dest='alpha_per_data',
                       help='Alpha per data point (default: 0.001)')
    parser.add_argument('--N', '--N-i', type=int, default=99,
                       dest='N_i',
                       help='Number of data per agent (default: 99)')

    # Directory arguments
    parser.add_argument('--symbolic-dir', type=str, default=None,
                       help='Symbolic results directory (default: IBD_analysis/results)')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Simulation data directory (default: from config)')
    parser.add_argument('--output-dir', type=str,
                       default='comparison_results',
                       help='Output directory for comparison results')

    # Display options
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Validate M is odd
    if args.M % 2 == 0:
        raise ValueError(f"M must be odd (got M={args.M})")

    print("\n" + "="*80)
    print("SYMBOLIC vs SIMULATION F-MATRIX COMPARISON")
    print("="*80)
    print(f"M (agents): {args.M}")
    print(f"Cases: {', '.join(args.cases)}")
    print(f"Parameters: m={args.coupling_strength}, alpha_per_data={args.alpha_per_data}, N_i={args.N_i}")
    print("="*80 + "\n")

    # Prepare results storage
    all_stats = []

    # Process each case
    for case_name in args.cases:
        print(f"\nProcessing {case_name}...")
        print("-" * 40)

        # Get case parameters
        flow_type, nonzero_alpha = get_case_parameters(case_name)

        # Calculate alpha value for symbolic solution
        # For centralized (center), alpha is total alpha at center
        # For evenly distributed, alpha is per-agent alpha
        if nonzero_alpha == 'center':
            alpha_symbolic = args.alpha_per_data * args.N_i  # Total alpha at center
        else:
            alpha_symbolic = args.alpha_per_data * args.N_i  # Per-agent alpha

        print(f"  Flow type: {flow_type}")
        print(f"  Alpha distribution: {nonzero_alpha}")
        print(f"  Alpha (symbolic): {alpha_symbolic}")

        # Load symbolic F-matrix
        try:
            F_symbolic = load_symbolic_f_matrix(
                M=args.M,
                case_name=case_name,
                m_val=args.coupling_strength,
                alpha_val=alpha_symbolic,
                N_val=args.N_i,
                results_dir=args.symbolic_dir
            )
            print(f"  ✓ Loaded symbolic F-matrix")
        except FileNotFoundError as e:
            print(f"  ✗ Error: {e}")
            continue

        # Get simulation directory
        sim_dir = get_simulation_dir(
            M=args.M,
            N_i=args.N_i,
            coupling_strength=args.coupling_strength,
            alpha_per_data=args.alpha_per_data,
            flow_type=flow_type,
            nonzero_alpha=nonzero_alpha,
            data_raw_dir=args.data_dir
        )
        print(f"  Simulation dir: {sim_dir}")

        # Load simulation F-matrix
        try:
            F_simulation = load_simulation_f_matrix(sim_dir)
            print(f"  ✓ Loaded simulation F-matrix")
        except FileNotFoundError as e:
            print(f"  ✗ Error: {e}")
            continue

        # Check dimensions match
        if F_symbolic.shape != F_simulation.shape:
            print(f"  ✗ Error: Dimension mismatch!")
            print(f"     Symbolic: {F_symbolic.shape}")
            print(f"     Simulation: {F_simulation.shape}")
            continue

        # Compare matrices
        params = {
            'm': args.coupling_strength,
            'alpha': alpha_symbolic,
            'N': args.N_i,
            'flow_type': flow_type,
            'nonzero_alpha': nonzero_alpha,
        }

        stats = compare_matrices(F_symbolic, F_simulation, case_name, params)
        all_stats.append(stats)

        print(f"  ✓ Comparison complete")
        print(f"     MAE: {stats['mean_absolute_error']:.6f}")
        print(f"     RMSE: {stats['rmse']:.6f}")
        print(f"     Correlation: {stats['correlation']:.6f}")

        # Create plots
        plot_comparison(stats, args.output_dir)

    # Create summary
    if all_stats:
        print("\n" + "="*80)
        print("Creating summary...")
        create_summary_table(all_stats, args.output_dir)
        print(f"\nAll results saved to: {args.output_dir}")
        print("="*80 + "\n")
    else:
        print("\n" + "="*80)
        print("No successful comparisons were made.")
        print("Please check that both symbolic solutions and simulation data exist.")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
