"""
Parameter space analysis for concentric distribution.

Sweep through parameter space using exponential scales to identify
regions where concentric distributions occur.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from itertools import product
from tqdm import tqdm
import argparse

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.concentric_analysis import analyze_concentric_for_parameters


def generate_exponential_values(base, powers):
    """
    Generate values as base^power for each power.

    Args:
        base: Base value (e.g., 2)
        powers: Iterable of power values

    Returns:
        list: [base^p for p in powers]

    Examples:
        >>> generate_exponential_values(2, range(0, 5))
        [1, 2, 4, 8, 16]
        >>> generate_exponential_values(2, range(-3, 0))
        [0.125, 0.25, 0.5]
    """
    return [base ** p for p in powers]


def parameter_sweep_concentric(
    # N parameters (population size)
    N_base=2,
    N_powers=range(5, 11),  # [2^5, ..., 2^10] = [32, ..., 1024]

    # m parameters (coupling strength)
    m_base=2,
    m_powers=range(-10, 0),  # [2^-10, ..., 2^-1] = [~0.001, ..., 0.5]

    # alpha parameters (innovation rate)
    alpha_base=2,
    alpha_powers=range(-15, -5),  # [2^-15, ..., 2^-6]

    # Model parameters
    M_values=[3],  # Support general M, but default to 3
    cases=['case1', 'case2', 'case3', 'case4'],
    distance_methods=['nei', '1-F', '1/F'],

    # Computation settings
    use_symbolic=True,
    verbose=False,

    # Output
    output_dir='IBD_analysis/results/concentric_analysis'
):
    """
    Sweep parameter space with exponential scaling.

    Args:
        N_base, N_powers: N = N_base^power for power in N_powers
        m_base, m_powers: m = m_base^power for power in m_powers
        alpha_base, alpha_powers: alpha = alpha_base^power for power in alpha_powers
        M_values: List of M values to test
        cases: List of case names
        distance_methods: List of distance metrics
        use_symbolic: Use symbolic solutions when available
        verbose: Print detailed progress
        output_dir: Output directory for results

    Returns:
        DataFrame with results

    Examples:
        >>> # Test small parameter space
        >>> df = parameter_sweep_concentric(
        ...     N_powers=range(5, 7),      # [32, 64]
        ...     m_powers=range(-5, -3),    # [0.03125, 0.0625]
        ...     alpha_powers=range(-10, -8), # [~0.001, ~0.002]
        ...     M_values=[3],
        ...     cases=['case1']
        ... )
    """
    # Generate parameter values
    N_values = generate_exponential_values(N_base, N_powers)
    m_values = generate_exponential_values(m_base, m_powers)
    alpha_values = generate_exponential_values(alpha_base, alpha_powers)

    if verbose:
        print(f"Parameter ranges:")
        print(f"  N: {len(N_values)} values from {min(N_values):.6g} to {max(N_values):.6g}")
        print(f"  m: {len(m_values)} values from {min(m_values):.6g} to {max(m_values):.6g}")
        print(f"  α: {len(alpha_values)} values from {min(alpha_values):.6g} to {max(alpha_values):.6g}")
        print(f"  M: {M_values}")
        print(f"  Cases: {cases}")
        print(f"  Distance methods: {distance_methods}")

    # Case configurations
    case_configs = {
        'case1': (False, False),
        'case2': (True, False),
        'case3': (False, True),
        'case4': (True, True),
    }

    results = []

    # Total combinations
    total = len(list(product(
        N_values, m_values, alpha_values,
        M_values, cases, distance_methods
    )))

    if verbose:
        print(f"\nTotal parameter combinations: {total:,}")
        print("Starting parameter sweep...\n")

    pbar = tqdm(total=total, desc="Parameter sweep", disable=not verbose)

    for N, m, alpha, M, case, method in product(
        N_values, m_values, alpha_values,
        M_values, cases, distance_methods
    ):
        center_prestige, centralized_neologism_creation = case_configs[case]

        try:
            result = analyze_concentric_for_parameters(
                N, m, alpha, M,
                center_prestige, centralized_neologism_creation,
                distance_method=method,
                use_symbolic=use_symbolic,
                verbose=False
            )

            F_mat = result['F_matrix']

            results.append({
                'N': N,
                'm': m,
                'alpha': alpha,
                'M': M,
                'case': case,
                'distance_method': method,
                'is_concentric': result['is_concentric'],
                'method_used': result['method_used'],
                # Store some F-matrix statistics
                'F_diag_mean': np.mean(np.diag(F_mat)),
                'F_offdiag_mean': np.mean(F_mat[~np.eye(M, dtype=bool)]),
                'F_min': np.min(F_mat),
                'F_max': np.max(F_mat),
            })

        except Exception as e:
            if verbose:
                print(f"\nError at N={N:.2e}, m={m:.2e}, α={alpha:.2e}, M={M}, {case}, {method}: {e}")

            results.append({
                'N': N,
                'm': m,
                'alpha': alpha,
                'M': M,
                'case': case,
                'distance_method': method,
                'is_concentric': np.nan,
                'method_used': 'error',
                'F_diag_mean': np.nan,
                'F_offdiag_mean': np.nan,
                'F_min': np.nan,
                'F_max': np.nan,
            })

        pbar.update(1)

    pbar.close()

    # Create DataFrame
    df = pd.DataFrame(results)

    # Add log-transformed parameter columns for easier plotting
    df['log2_N'] = np.log2(df['N'])
    df['log2_m'] = np.log2(df['m'])
    df['log2_alpha'] = np.log2(df['alpha'])

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'concentric_parameter_sweep.csv')
    df.to_csv(csv_path, index=False)

    if verbose:
        print(f"\nResults saved to: {csv_path}")
        print(f"Total results: {len(df)}")
        print(f"Concentric cases: {df['is_concentric'].sum()} ({100*df['is_concentric'].mean():.1f}%)")

    return df


def plot_concentric_heatmaps(df, output_dir='IBD_analysis/results/concentric_analysis'):
    """
    Plot heatmaps showing where concentric distribution occurs.

    Creates separate plots for each combination of:
    - M value
    - case (case1, case2, case3, case4)
    - distance_method

    Args:
        df: Results DataFrame
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get unique values
    M_values = sorted(df['M'].unique())
    cases = sorted(df['case'].unique())
    methods = sorted(df['distance_method'].unique())

    for M_val, case, method in product(M_values, cases, methods):
        # Filter data
        data = df[(df['M'] == M_val) &
                  (df['case'] == case) &
                  (df['distance_method'] == method)]

        if len(data) == 0:
            continue

        # Get unique N values
        N_vals = sorted(data['N'].unique())
        n_N = len(N_vals)

        if n_N == 0:
            continue

        # Create subplots
        fig, axes = plt.subplots(1, n_N, figsize=(6*n_N, 5))
        if n_N == 1:
            axes = [axes]

        for idx, N_val in enumerate(N_vals):
            ax = axes[idx]

            # Pivot table for heatmap (alpha vs m)
            subset = data[data['N'] == N_val]
            pivot = subset.pivot_table(
                values='is_concentric',
                index='log2_alpha',  # Y-axis
                columns='log2_m',    # X-axis
                aggfunc='first'
            )

            if pivot.empty:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                continue

            # Plot heatmap
            im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto',
                          origin='lower', vmin=0, vmax=1, interpolation='nearest')

            # Labels
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_yticks(range(len(pivot.index)))

            # Format tick labels as powers
            ax.set_xticklabels([f'2^{int(x)}' for x in pivot.columns], rotation=45, ha='right')
            ax.set_yticklabels([f'2^{int(y)}' for y in pivot.index])

            ax.set_xlabel('m (coupling strength)')
            ax.set_ylabel('α (innovation rate)')
            ax.set_title(f'N=2^{int(np.log2(N_val))}={N_val:.0f}')

            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('is_concentric')

        fig.suptitle(f'M={M_val}, {case}, {method} distance', fontsize=14, y=1.02)
        plt.tight_layout()

        # Save
        filename = f'concentric_heatmap_M{M_val}_{case}_{method}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {filename}")


def plot_summary_statistics(df, output_dir='IBD_analysis/results/concentric_analysis'):
    """
    Plot summary statistics of concentric analysis.

    Args:
        df: Results DataFrame
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Concentric rate by case and distance method
    fig, ax = plt.subplots(figsize=(10, 6))

    summary = df.groupby(['case', 'distance_method'])['is_concentric'].agg(['mean', 'count'])
    summary = summary.reset_index()
    summary['percentage'] = summary['mean'] * 100

    pivot = summary.pivot(index='case', columns='distance_method', values='percentage')
    pivot.plot(kind='bar', ax=ax)

    ax.set_ylabel('Concentric rate (%)')
    ax.set_xlabel('Case')
    ax.set_title('Concentric distribution rate by case and distance method')
    ax.legend(title='Distance method')
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'concentric_rate_summary.png'), dpi=150)
    plt.close()

    print("Saved: concentric_rate_summary.png")

    # 2. Concentric rate vs parameters
    for param in ['N', 'm', 'alpha']:
        fig, ax = plt.subplots(figsize=(10, 6))

        for case in df['case'].unique():
            subset = df[df['case'] == case]
            grouped = subset.groupby(param)['is_concentric'].mean()

            ax.plot(grouped.index, grouped.values, marker='o', label=case)

        ax.set_xlabel(param)
        ax.set_ylabel('Concentric rate')
        ax.set_xscale('log', base=2)
        ax.set_title(f'Concentric rate vs {param}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, f'concentric_vs_{param}.png'), dpi=150)
        plt.close()

        print(f"Saved: concentric_vs_{param}.png")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Analyze concentric distribution in parameter space'
    )

    # N parameters
    parser.add_argument('--N-base', type=float, default=2,
                       help='Base for N values (default: 2)')
    parser.add_argument('--N-powers', type=str, default='5:11',
                       help='Range for N powers as start:stop (default: 5:11 for 2^5 to 2^10)')

    # m parameters
    parser.add_argument('--m-base', type=float, default=2,
                       help='Base for m values (default: 2)')
    parser.add_argument('--m-powers', type=str, default='-10:0',
                       help='Range for m powers as start:stop (default: -10:0)')

    # alpha parameters
    parser.add_argument('--alpha-base', type=float, default=2,
                       help='Base for alpha values (default: 2)')
    parser.add_argument('--alpha-powers', type=str, default='-15:-5',
                       help='Range for alpha powers as start:stop (default: -15:-5)')

    # Model parameters
    parser.add_argument('--M', type=int, nargs='+', default=[3],
                       help='M values to test (default: 3)')
    parser.add_argument('--cases', type=str, nargs='+',
                       default=['case1', 'case2', 'case3', 'case4'],
                       help='Cases to analyze')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['nei', '1-F', '1/F'],
                       help='Distance methods')

    # Options
    parser.add_argument('--numerical', action='store_true',
                       help='Use numerical computation instead of symbolic')
    parser.add_argument('--output-dir', type=str,
                       default='IBD_analysis/results/concentric_analysis',
                       help='Output directory')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plotting')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Parse power ranges
    def parse_range(s):
        start, stop = map(int, s.split(':'))
        return range(start, stop)

    N_powers = parse_range(args.N_powers)
    m_powers = parse_range(args.m_powers)
    alpha_powers = parse_range(args.alpha_powers)

    # Run parameter sweep
    df = parameter_sweep_concentric(
        N_base=args.N_base,
        N_powers=N_powers,
        m_base=args.m_base,
        m_powers=m_powers,
        alpha_base=args.alpha_base,
        alpha_powers=alpha_powers,
        M_values=args.M,
        cases=args.cases,
        distance_methods=args.methods,
        use_symbolic=not args.numerical,
        verbose=args.verbose,
        output_dir=args.output_dir
    )

    # Plot results
    if not args.no_plots:
        print("\nGenerating plots...")
        plot_concentric_heatmaps(df, args.output_dir)
        plot_summary_statistics(df, args.output_dir)

    print("\nAnalysis complete!")
    print(f"Results directory: {args.output_dir}")


if __name__ == "__main__":
    main()
