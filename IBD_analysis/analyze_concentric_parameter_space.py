"""
Parameter space analysis for concentric distribution.

Sweep through parameter space using exponential scales to identify
regions where concentric distributions occur.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import sys
from itertools import product
from tqdm import tqdm
import argparse
import multiprocessing as mp
from functools import partial

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.concentric_analysis import analyze_concentric_for_parameters
from IBD_analysis.src.f_matrix_cache import FMatrixCache, f_matrix_to_columns, columns_to_f_matrix
from IBD_analysis.src.distance_metrics import compute_f_distance
from IBD_analysis.src.concentric_analysis import is_concentric_distribution


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


def process_single_parameter_combination(args_tuple, cached_f_matrix=None):
    """
    Wrapper function for parallel processing of a single parameter combination.

    Args:
        args_tuple: (N, m, alpha, M, case, method, case_configs)
        cached_f_matrix: Pre-loaded F-matrix from cache (if available)

    Returns:
        dict: Result dictionary with F-matrix and metrics, or None if failed
    """
    N, m, alpha, M, case, method, case_configs = args_tuple

    try:
        # Get F-matrix (from cache or compute)
        if cached_f_matrix is not None:
            F_mat = cached_f_matrix
        else:
            center_prestige, centralized_neologism_creation = case_configs[case]
            result = analyze_concentric_for_parameters(
                N, m, alpha, M,
                center_prestige, centralized_neologism_creation,
                distance_method=method,
                use_symbolic=True,  # Always use symbolic
                verbose=False
            )
            F_mat = result['F_matrix']

        # Compute distance and concentric judgment (always fresh)
        distance_matrix = compute_f_distance(F_mat, method=method)
        is_concentric = is_concentric_distribution(distance_matrix)

        # Build result with F-matrix
        result_dict = {
            'N': N,
            'm': m,
            'alpha': alpha,
            'M': M,
            'case': case,
            'distance_method': method,
            'is_concentric': is_concentric,
            # Store full F-matrix
            **f_matrix_to_columns(F_mat, M),
            # Store some F-matrix statistics
            'F_diag_mean': np.mean(np.diag(F_mat)),
            'F_offdiag_mean': np.mean(F_mat[~np.eye(M, dtype=bool)]),
            'F_min': np.min(F_mat),
            'F_max': np.max(F_mat),
        }

        return result_dict

    except Exception as e:
        print(f"Error processing N={N}, m={m}, alpha={alpha}, M={M}, {case}, {method}: {e}")
        return None


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
    distance_methods=['nei', '1-F'],

    # Computation settings
    precompute_symbolic=False,
    use_cache=False,
    cache_dir='IBD_analysis/results/f_matrix_cache',
    verbose=False,
    n_workers=None,

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
        precompute_symbolic: Compute and save symbolic solutions before sweep
        use_cache: Use F-matrix cache to avoid recomputation
        cache_dir: Directory for F-matrix cache
        verbose: Print detailed progress
        n_workers: Number of parallel workers (default: CPU count)
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
        ...     cases=['case1'],
        ...     use_cache=True
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

    # Precompute symbolic solutions if requested
    if precompute_symbolic:
        print("\n" + "="*60)
        print("Precomputing symbolic solutions")
        print("="*60)

        from src.f_matrix_symbolic import compute_f_matrix_stationary, save_results, get_case_name

        # Check which symbolic solutions need to be computed
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

        for M in M_values:
            for case in cases:
                result_file = os.path.join(results_dir, f"M{M}_{case}.pkl")

                if os.path.exists(result_file):
                    print(f"✓ M={M}, {case}: symbolic solution already exists")
                else:
                    print(f"⏳ M={M}, {case}: computing symbolic solution...")

                    # Determine case parameters
                    case_configs = {
                        'case1': (False, False),
                        'case2': (True, False),
                        'case3': (False, True),
                        'case4': (True, True),
                    }
                    center_prestige, centralized_neologism = case_configs[case]

                    # Compute symbolic solution
                    F_matrix, metadata = compute_f_matrix_stationary(
                        M=M,
                        center_prestige=center_prestige,
                        centralized_neologism_creation=centralized_neologism,
                        verbose=verbose
                    )

                    # Save result
                    save_results(
                        M=M,
                        case_name=case,
                        F_matrix=F_matrix,
                        metadata=metadata,
                        output_dir=results_dir
                    )

                    print(f"  ✓ Saved to {result_file}")

        print("="*60)
        print("Symbolic solution precomputation complete\n")

    # Case configurations
    case_configs = {
        'case1': (False, False),
        'case2': (True, False),
        'case3': (False, True),
        'case4': (True, True),
    }

    # Determine number of workers
    if n_workers is None:
        n_workers = mp.cpu_count()

    # All results
    results = []

    # Process each M value separately
    for M in M_values:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing M={M}")
            print('='*60)

        # Initialize cache for this M
        cache = FMatrixCache(cache_dir, M) if use_cache else None

        # Process each case
        for case in cases:
            if verbose:
                print(f"\nCase: {case}")

            # Generate parameter combinations for this case
            param_combinations = list(product(N_values, m_values, alpha_values))

            if use_cache:
                # Load cached F-matrices
                cached_df = cache.load_case(case)

                if verbose and len(cached_df) > 0:
                    print(f"  Loaded {len(cached_df)} cached F-matrices")

                # Identify missing parameters
                missing_params = cache.get_missing_params(case, param_combinations)

                if verbose:
                    print(f"  Parameters: {len(param_combinations)} total, {len(missing_params)} to compute")

                # Compute missing F-matrices
                if len(missing_params) > 0:
                    if verbose:
                        print(f"  Computing {len(missing_params)} new F-matrices...")

                    # Prepare args for missing computations (method-independent)
                    compute_args = [
                        (N, m, alpha, M, case, distance_methods[0], case_configs)  # Use any method (doesn't affect F-matrix)
                        for N, m, alpha in missing_params
                    ]

                    # Compute in parallel
                    new_results = []
                    with mp.Pool(processes=n_workers) as pool:
                        for result in tqdm(
                            pool.imap(process_single_parameter_combination, compute_args),
                            total=len(compute_args),
                            desc=f"  Computing F-matrices ({case})",
                            disable=not verbose
                        ):
                            if result is not None:
                                new_results.append(result)

                    # Save new F-matrices to cache
                    if len(new_results) > 0:
                        f_matrix_results = [
                            {
                                'N': r['N'],
                                'm': r['m'],
                                'alpha': r['alpha'],
                                'F_matrix': columns_to_f_matrix(r, M)
                            }
                            for r in new_results
                        ]
                        cache.append_results(case, f_matrix_results)

                        if verbose:
                            print(f"  Saved {len(f_matrix_results)} new F-matrices to cache")

                # Now reload cache and process all distance methods
                cached_df = cache.load_case(case)

                if verbose:
                    print(f"  Computing distances for {len(distance_methods)} methods...")

                # Process all parameters with all distance methods
                for method in distance_methods:
                    for idx, row in tqdm(
                        cached_df.iterrows(),
                        total=len(cached_df),
                        desc=f"  {case}, {method}",
                        disable=not verbose
                    ):
                        N, m, alpha = row['N'], row['m'], row['alpha']
                        F_mat = columns_to_f_matrix(row, M)

                        # Compute distance and concentric
                        distance_matrix = compute_f_distance(F_mat, method=method)
                        is_concentric = is_concentric_distribution(distance_matrix)

                        result = {
                            'N': N, 'm': m, 'alpha': alpha, 'M': M,
                            'case': case, 'distance_method': method,
                            'is_concentric': is_concentric,
                            **f_matrix_to_columns(F_mat, M),
                            'F_diag_mean': np.mean(np.diag(F_mat)),
                            'F_offdiag_mean': np.mean(F_mat[~np.eye(M, dtype=bool)]),
                            'F_min': np.min(F_mat),
                            'F_max': np.max(F_mat),
                        }
                        results.append(result)

            else:
                # No cache: compute everything from scratch
                process_args = [
                    (N, m, alpha, M, case, method, case_configs)
                    for N, m, alpha in param_combinations
                    for method in distance_methods
                ]

                if verbose:
                    print(f"  Computing {len(process_args)} combinations...")

                with mp.Pool(processes=n_workers) as pool:
                    for result in tqdm(
                        pool.imap(process_single_parameter_combination, process_args),
                        total=len(process_args),
                        desc=f"  {case}",
                        disable=not verbose
                    ):
                        if result is not None:
                            results.append(result)

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

    Each plot contains subplots for different m values (side by side).
    Each subplot has X-axis=N, Y-axis=alpha.

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

        # Get unique m values
        m_vals = sorted(data['m'].unique())
        n_m = len(m_vals)

        if n_m == 0:
            continue

        # Create subplots (one for each m value)
        fig, axes = plt.subplots(1, n_m, figsize=(6*n_m, 5))
        if n_m == 1:
            axes = [axes]

        for idx, m_val in enumerate(m_vals):
            ax = axes[idx]

            # Pivot table for heatmap (N vs alpha)
            subset = data[data['m'] == m_val]
            pivot = subset.pivot_table(
                values='is_concentric',
                index='log2_alpha',  # Y-axis
                columns='log2_N',    # X-axis
                aggfunc='first'
            )

            if pivot.empty:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                continue

            # Plot heatmap with discrete colormap
            # Use discrete colors: green for 0 (not concentric), red for 1 (concentric)
            cmap = ListedColormap(['#90EE90', '#FF6B6B'])  # Light green, light red
            im = ax.imshow(pivot.values, cmap=cmap, aspect='auto',
                          origin='lower', vmin=0, vmax=1, interpolation='nearest')

            # Labels
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_yticks(range(len(pivot.index)))

            # Format tick labels as powers
            ax.set_xticklabels([f'2^{int(x)}' for x in pivot.columns], rotation=45, ha='right')
            ax.set_yticklabels([f'2^{int(y)}' for y in pivot.index])

            ax.set_xlabel('N (population size)')
            ax.set_ylabel('α (innovation rate)')

            # Title with m value
            m_log2 = int(np.log2(m_val))
            ax.set_title(f'm=2^{m_log2}={m_val:.6f}')

            # Colorbar
            cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
            cbar.set_ticklabels(['No', 'Yes'])
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
    parser.add_argument('--N-powers', type=str, default='0:10',
                       help='Range for N powers as start:stop (default: 5:11 for 2^5 to 2^10)')

    # m parameters
    parser.add_argument('--m-base', type=float, default=2,
                       help='Base for m values (default: 2)')
    parser.add_argument('--m-powers', type=str, default='-10:0',
                       help='Range for m powers as start:stop (default: -10:0)')

    # alpha parameters
    parser.add_argument('--alpha-base', type=float, default=2,
                       help='Base for alpha values (default: 2)')
    parser.add_argument('--alpha-powers', type=str, default='-15:5',
                       help='Range for alpha powers as start:stop (default: -15:-5)')

    # Model parameters
    parser.add_argument('--M', type=int, nargs='+', default=[3],
                       help='M values to test (default: 3)')
    parser.add_argument('--cases', type=str, nargs='+',
                       default=['case1', 'case2', 'case3', 'case4'],
                       help='Cases to analyze')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['nei', '1-F'],
                       help='Distance methods')

    # Options
    parser.add_argument('--precompute-symbolic', action='store_true',
                       help='Precompute symbolic solutions for all M values before parameter sweep')
    parser.add_argument('--use-cache', action='store_true',
                       help='Use F-matrix cache to avoid recomputation')
    parser.add_argument('--cache-dir', type=str,
                       default='IBD_analysis/results/f_matrix_cache',
                       help='Directory for F-matrix cache')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear F-matrix cache and exit')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--output-dir', type=str,
                       default='IBD_analysis/results/concentric_analysis',
                       help='Output directory')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plotting')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Handle cache clearing
    if args.clear_cache:
        for M in args.M:
            cache = FMatrixCache(args.cache_dir, M)
            cache.clear_cache()
        print(f"Cache cleared for M={args.M}")
        return

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
        precompute_symbolic=args.precompute_symbolic,
        use_cache=args.use_cache,
        cache_dir=args.cache_dir,
        verbose=args.verbose,
        n_workers=args.workers,
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
