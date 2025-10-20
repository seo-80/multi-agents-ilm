"""
Analyze parameter sweep results and create integrated visualizations.

This script reads multiple simulation results, computes diversity metrics,
and creates comprehensive plots showing parameter effects.

Usage examples:
  # Analyze single parameter sweep
  python src/analyze_parameter_sweep.py --param strength --metric diversity

  # Create heatmap for two parameters
  python src/analyze_parameter_sweep.py --param strength,Ni --metric distance --plot_type heatmap

  # Analyze all parameters with all metrics
  python src/analyze_parameter_sweep.py --all
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import argparse
from tqdm import tqdm
import pickle
from scipy import stats
import re


# Parameter defaults (must match run_naive_simulations_parallel.sh)
DEFAULT_PARAMS = {
    'strength': [0.0025, 0.005, 0.01, 0.02, 0.04],
    'alpha': [0.00025, 0.0005, 0.001, 0.002, 0.004],
    'Ni': [25, 50, 100, 200, 400],
    'flow_type': ['outward', 'bidirectional'],
    'nonzero_alpha': ['center', 'evenly']
}

DEFAULT_FIXED = {
    'strength': 0.01,
    'alpha': 0.001,
    'Ni': 100,
    'agents_count': 7
}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze parameter sweep results')

    parser.add_argument('--param', '-p', type=str, default='strength',
                       help='Parameter(s) to analyze: strength, alpha, Ni, or comma-separated pair (e.g., strength,Ni)')
    parser.add_argument('--metric', '-m', type=str, default='all',
                       choices=['diversity', 'distance', 'similarity', 'age', 'all'],
                       help='Metric to compute and plot')
    parser.add_argument('--plot_type', type=str, default='auto',
                       choices=['line', 'heatmap', 'bar', 'auto'],
                       help='Type of plot (auto selects based on number of parameters)')
    parser.add_argument('--all', action='store_true',
                       help='Generate all combinations of parameters and metrics')
    parser.add_argument('--skip', type=int, default=0,
                       help='Number of initial snapshots to skip')
    parser.add_argument('--max_snapshots', type=int, default=None,
                       help='Maximum number of snapshots to analyze per simulation')
    parser.add_argument('--output_dir', type=str, default='data/naive_simulation/parameter_analysis',
                       help='Output directory for plots and data')
    parser.add_argument('--base_dir', type=str, default='results',
                       help='Base directory containing simulation results')
    parser.add_argument('--cache', action='store_true',
                       help='Use cached metrics if available')
    parser.add_argument('--flow_type', type=str, default='all',
                       choices=['outward', 'bidirectional', 'all'],
                       help='Filter by flow type')
    parser.add_argument('--nonzero_alpha', type=str, default='all',
                       choices=['center', 'evenly', 'all'],
                       help='Filter by nonzero_alpha distribution')

    # Fixed parameter values (used for directory naming)
    parser.add_argument('--fixed_strength', type=float, default=DEFAULT_FIXED['strength'],
                       help='Fixed strength value (for directory naming)')
    parser.add_argument('--fixed_alpha', type=float, default=DEFAULT_FIXED['alpha'],
                       help='Fixed alpha value (for directory naming)')
    parser.add_argument('--fixed_Ni', type=int, default=DEFAULT_FIXED['Ni'],
                       help='Fixed N_i value (for directory naming)')
    parser.add_argument('--fixed_agents_count', type=int, default=DEFAULT_FIXED['agents_count'],
                       help='Fixed agents_count value (for directory naming)')

    return parser.parse_args()


def get_subdir_name(flow_type, nonzero_alpha, strength, agents_count, N_i, alpha):
    """Generate subdirectory name matching simulation naming convention."""
    if flow_type == 'bidirectional':
        flow_prefix = 'bidirectional_flow-'
    else:
        flow_prefix = 'outward_flow-'

    return f"{flow_prefix}nonzero_alpha_{nonzero_alpha}_fr_{strength}_agents_{agents_count}_N_i_{N_i}_alpha_{alpha}"


def calculate_diversity_metrics(state):
    """
    Calculate diversity metrics from state array.

    Args:
        state: (agents_count, N_i, 3) array

    Returns:
        dict with diversity metrics
    """
    agents_count, N_i, _ = state.shape

    # Flatten all memes across all agents
    all_memes = [tuple(state[i, j]) for i in range(agents_count) for j in range(N_i)]
    unique_memes = set(all_memes)

    # Count frequencies
    from collections import Counter
    meme_counts = Counter(all_memes)
    total = sum(meme_counts.values())

    # Shannon entropy
    shannon_entropy = -sum((count/total) * np.log(count/total)
                          for count in meme_counts.values() if count > 0)

    # Simpson diversity (1 - Simpson index)
    simpson = 1 - sum((count/total)**2 for count in meme_counts.values())

    # Per-agent diversity
    agent_diversities = []
    for i in range(agents_count):
        agent_memes = [tuple(state[i, j]) for j in range(N_i)]
        agent_unique = len(set(agent_memes))
        agent_diversities.append(agent_unique / N_i)

    return {
        'unique_memes': len(unique_memes),
        'shannon_entropy': shannon_entropy,
        'simpson_diversity': simpson,
        'mean_agent_diversity': np.mean(agent_diversities),
        'std_agent_diversity': np.std(agent_diversities),
        'total_memes': total
    }


def calculate_distance_metrics(distance_matrix):
    """Calculate summary metrics from distance matrix."""
    n = distance_matrix.shape[0]

    # Extract upper triangle (exclude diagonal)
    triu_indices = np.triu_indices(n, k=1)
    distances = distance_matrix[triu_indices]

    return {
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'max_distance': np.max(distances),
        'min_distance': np.min(distances),
        'median_distance': np.median(distances)
    }


def load_and_compute_metrics(data_dir, skip=0, max_snapshots=None):
    """
    Load simulation data and compute all metrics.

    Args:
        data_dir: Directory containing raw simulation data
        skip: Number of initial files to skip
        max_snapshots: Maximum number of snapshots to analyze

    Returns:
        dict with time series of metrics
    """
    state_files = sorted(glob.glob(os.path.join(data_dir, "state_*.npy")))
    distance_files = sorted(glob.glob(os.path.join(data_dir, "distance_*.npy")))

    if not state_files:
        print(f"Warning: No state files found in {data_dir}")
        return None

    # Apply skip and max_snapshots
    state_files = state_files[skip:]
    distance_files = distance_files[skip:]

    if max_snapshots:
        state_files = state_files[:max_snapshots]
        distance_files = distance_files[:max_snapshots]

    # Load time map
    csv_path = os.path.join(data_dir, "save_idx_t_map.csv")
    if os.path.exists(csv_path):
        time_map = pd.read_csv(csv_path)
        id2t = {int(row['save_idx']): int(row['t']) for _, row in time_map.iterrows()}
    else:
        id2t = {}

    metrics = {
        'timesteps': [],
        'diversity': [],
        'distance': [],
        'file_ids': []
    }

    for state_file, dist_file in tqdm(zip(state_files, distance_files),
                                      total=len(state_files),
                                      desc=f"Processing {os.path.basename(data_dir)}"):
        # Extract file ID
        file_id = int(os.path.basename(state_file).split('_')[1].split('.')[0])
        t = id2t.get(file_id, 0)

        # Load data
        state = np.load(state_file)
        distance = np.load(dist_file)

        # Compute metrics
        div_metrics = calculate_diversity_metrics(state)
        dist_metrics = calculate_distance_metrics(distance)

        metrics['timesteps'].append(t)
        metrics['file_ids'].append(file_id)
        metrics['diversity'].append(div_metrics)
        metrics['distance'].append(dist_metrics)

    return metrics


def find_simulation_dirs(base_dir, param_name, fixed_params, flow_type='all', nonzero_alpha='all'):
    """
    Find all simulation directories for a given parameter sweep.

    Args:
        base_dir: Base results directory
        param_name: Parameter being swept (strength, alpha, Ni)
        fixed_params: Dict with fixed parameter values {'strength': 0.01, 'alpha': 0.001, 'Ni': 100}
        flow_type: Filter by flow type
        nonzero_alpha: Filter by nonzero_alpha

    Returns:
        list of (param_value, data_dir, flow_type, nonzero_alpha) tuples
    """
    raw_base = os.path.join("data/naive_simulation/raw")

    if not os.path.exists(raw_base):
        return []

    results = []
    all_dirs = os.listdir(raw_base)

    # Get all subdirectories
    for subdir in all_dirs:
        full_path = os.path.join(raw_base, subdir)
        if not os.path.isdir(full_path):
            continue

        # Parse directory name
        parts = subdir.split('_')
        try:
            # Extract parameters from directory name
            if 'bidirectional_flow-' in subdir:
                sim_flow = 'bidirectional'
            elif 'outward_flow-' in subdir:
                sim_flow = 'outward'
            else:
                continue

            # Filter by flow_type
            if flow_type != 'all' and sim_flow != flow_type:
                continue

            # Find nonzero_alpha
            if 'nonzero_alpha_evenly' in subdir:
                sim_nza = 'evenly'
            elif 'nonzero_alpha_center' in subdir:
                sim_nza = 'center'
            else:
                continue

            # Filter by nonzero_alpha
            if nonzero_alpha != 'all' and sim_nza != nonzero_alpha:
                continue

            # Extract all parameter values from directory using regex
            sim_params = {}

            # Match strength: _fr_<value>
            strength_match = re.search(r'_fr_([0-9.]+)', subdir)
            if strength_match:
                sim_params['strength'] = float(strength_match.group(1))

            # Match alpha: _alpha_<value> (at the end to avoid nonzero_alpha)
            alpha_match = re.search(r'_alpha_([0-9.]+)$', subdir)
            if alpha_match:
                sim_params['alpha'] = float(alpha_match.group(1))

            # Match N_i: _N_i_<value>
            ni_match = re.search(r'_N_i_([0-9]+)', subdir)
            if ni_match:
                sim_params['Ni'] = int(ni_match.group(1))

            # Match agents_count: _agents_<value>
            agents_match = re.search(r'_agents_([0-9]+)', subdir)
            if agents_match:
                sim_params['agents_count'] = int(agents_match.group(1))

            # Check if fixed parameters match
            match = True
            for param, value in fixed_params.items():
                if param == param_name:
                    # This is the varying parameter, skip
                    continue
                if param in sim_params:
                    # Allow small floating point tolerance
                    if isinstance(value, float):
                        if abs(sim_params[param] - value) > 1e-9:
                            match = False
                            break
                    else:
                        if sim_params[param] != value:
                            match = False
                            break

            if not match:
                continue

            # Extract the varying parameter value
            if param_name in sim_params:
                param_value = sim_params[param_name]
                results.append((param_value, full_path, sim_flow, sim_nza))

        except (ValueError, IndexError) as e:
            continue

    # Filter by DEFAULT_PARAMS values for the varying parameter
    if param_name in DEFAULT_PARAMS:
        allowed_values = set(DEFAULT_PARAMS[param_name])
        results = [(val, path, flow, nza) for val, path, flow, nza in results
                   if val in allowed_values]

    return sorted(results, key=lambda x: x[0])


def aggregate_metrics(metrics_list, stat_type='mean'):
    """
    Aggregate metrics across time for each simulation.

    Args:
        metrics_list: List of metric dicts (already extracted by type)
        stat_type: 'mean', 'std', 'final', or specific metric name

    Returns:
        Aggregated value (dict or scalar)
    """
    if not metrics_list or len(metrics_list) == 0:
        return {}

    # metrics_list is already a list of dicts with specific metrics
    # e.g., [{'mean_distance': 10, 'std_distance': 2, ...}, ...]

    if stat_type == 'final':
        return metrics_list[-1]
    elif stat_type == 'mean':
        # Average each metric over time
        keys = metrics_list[0].keys()
        return {k: np.mean([v[k] for v in metrics_list]) for k in keys}
    elif stat_type == 'std':
        # Standard deviation over time
        keys = metrics_list[0].keys()
        return {k: np.std([v[k] for v in metrics_list]) for k in keys}
    else:
        # Specific metric name
        if stat_type in metrics_list[0]:
            return np.mean([v[stat_type] for v in metrics_list])
        else:
            return np.nan


def plot_single_parameter(df, param_name, metric_name, output_dir):
    """
    Create line plot for single parameter vs metric.

    Args:
        df: DataFrame with columns [param_value, metric_value, flow_type, nonzero_alpha]
        param_name: Name of the parameter
        metric_name: Name of the metric
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create figure with subplots for each flow_type x nonzero_alpha combination
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{metric_name} vs {param_name}', fontsize=16)

    # Match stitch_figures.py layout: rows=flow_type, cols=nonzero_alpha
    combinations = [
        ('bidirectional', 'evenly'),  # top-left
        ('bidirectional', 'center'),  # top-right
        ('outward', 'evenly'),        # bottom-left
        ('outward', 'center')         # bottom-right
    ]

    for ax, (flow, nza) in zip(axes.flat, combinations):
        subset = df[(df['flow_type'] == flow) & (df['nonzero_alpha'] == nza)]

        if len(subset) > 0:
            ax.plot(subset['param_value'], subset['metric_value'], 'o-', linewidth=2, markersize=8)
            ax.set_xlabel(param_name, fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            ax.set_title(f'{flow} / {nza}', fontsize=11)
            ax.grid(True, alpha=0.3)

            # Log scale for parameters that vary over orders of magnitude
            if param_name in ['strength', 'alpha'] and subset['param_value'].max() / subset['param_value'].min() > 10:
                ax.set_xscale('log')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel(param_name, fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            ax.set_title(f'{flow} / {nza}', fontsize=11)

    plt.tight_layout()

    output_path = os.path.join(output_dir, f'{param_name}_vs_{metric_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_two_parameters_heatmap(df, param1_name, param2_name, metric_name, output_dir):
    """
    Create heatmap for two parameters vs metric.

    Args:
        df: DataFrame with columns [param1, param2, metric_value, flow_type, nonzero_alpha]
        param1_name: Name of first parameter (rows)
        param2_name: Name of second parameter (columns)
        metric_name: Name of the metric
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'{metric_name}: {param1_name} vs {param2_name}', fontsize=16)

    # Match stitch_figures.py layout: rows=flow_type, cols=nonzero_alpha
    combinations = [
        ('bidirectional', 'evenly'),  # top-left
        ('bidirectional', 'center'),  # top-right
        ('outward', 'evenly'),        # bottom-left
        ('outward', 'center')         # bottom-right
    ]

    for ax, (flow, nza) in zip(axes.flat, combinations):
        subset = df[(df['flow_type'] == flow) & (df['nonzero_alpha'] == nza)]

        if len(subset) > 0:
            # Pivot to create matrix
            pivot = subset.pivot(index=param1_name, columns=param2_name, values='metric_value')

            # Create heatmap
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis', ax=ax, cbar_kws={'label': metric_name})
            ax.set_title(f'{flow} / {nza}', fontsize=11)
            ax.set_xlabel(param2_name, fontsize=12)
            ax.set_ylabel(param1_name, fontsize=12)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{flow} / {nza}', fontsize=11)

    plt.tight_layout()

    output_path = os.path.join(output_dir, f'{param1_name}_{param2_name}_vs_{metric_name}_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def analyze_single_parameter(args, param_name, metric_name):
    """Analyze single parameter sweep."""
    print(f"\n{'='*60}")
    print(f"Analyzing {param_name} vs {metric_name}")
    print(f"{'='*60}")

    # Build fixed parameters dict for filtering
    fixed_param_values = {
        'strength': args.fixed_strength,
        'alpha': args.fixed_alpha,
        'Ni': args.fixed_Ni,
        'agents_count': args.fixed_agents_count
    }

    # Create output directory with fixed parameter values
    # Determine which parameters are fixed (not being swept)
    fixed_param_labels = []
    if param_name != 'Ni':
        fixed_param_labels.append(f"N_i_{args.fixed_Ni}")
    if param_name != 'strength':
        fixed_param_labels.append(f"strength_{args.fixed_strength}")
    if param_name != 'alpha':
        fixed_param_labels.append(f"alpha_{args.fixed_alpha}")

    # Create directory structure
    if fixed_param_labels:
        param_output_dir = os.path.join(args.output_dir, '_'.join(fixed_param_labels))
    else:
        param_output_dir = args.output_dir

    # Find simulation directories with fixed parameter filtering
    sim_dirs = find_simulation_dirs(args.base_dir, param_name, fixed_param_values,
                                     args.flow_type, args.nonzero_alpha)

    if not sim_dirs:
        print(f"No simulation directories found for parameter: {param_name}")
        return

    print(f"Found {len(sim_dirs)} simulation directories")
    print(f"Output directory: {param_output_dir}")

    # Collect data
    data_rows = []

    for param_value, data_dir, flow_type, nonzero_alpha in sim_dirs:
        print(f"\nProcessing: {param_name}={param_value}, flow={flow_type}, nza={nonzero_alpha}")

        # Check cache
        cache_file = os.path.join(data_dir, f"metrics_cache_skip{args.skip}.pkl")

        if args.cache and os.path.exists(cache_file):
            print(f"Loading cached metrics from {cache_file}")
            with open(cache_file, 'rb') as f:
                metrics = pickle.load(f)
        else:
            # Compute metrics
            metrics = load_and_compute_metrics(data_dir, args.skip, args.max_snapshots)

            if metrics is None:
                continue

            # Save cache
            with open(cache_file, 'wb') as f:
                pickle.dump(metrics, f)
            print(f"Saved metrics cache to {cache_file}")

        # Extract specific metric
        if metric_name == 'diversity':
            # Average diversity metrics over time
            agg_metrics = aggregate_metrics(metrics['diversity'], 'mean')
            for key, value in agg_metrics.items():
                data_rows.append({
                    'param_value': param_value,
                    'flow_type': flow_type,
                    'nonzero_alpha': nonzero_alpha,
                    'metric_name': key,
                    'metric_value': value
                })
        elif metric_name == 'distance':
            # Average distance metrics over time
            agg_metrics = aggregate_metrics(metrics['distance'], 'mean')
            for key, value in agg_metrics.items():
                data_rows.append({
                    'param_value': param_value,
                    'flow_type': flow_type,
                    'nonzero_alpha': nonzero_alpha,
                    'metric_name': key,
                    'metric_value': value
                })

    if not data_rows:
        print("No data collected")
        return

    # Create DataFrame
    df = pd.DataFrame(data_rows)

    # Save summary data
    summary_dir = os.path.join(param_output_dir, 'summary_data')
    os.makedirs(summary_dir, exist_ok=True)
    csv_path = os.path.join(summary_dir, f'{param_name}_vs_{metric_name}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved summary data to {csv_path}")

    # Create plots for each specific metric
    for specific_metric in df['metric_name'].unique():
        df_metric = df[df['metric_name'] == specific_metric].copy()
        plot_single_parameter(df_metric, param_name, specific_metric, param_output_dir)


def analyze_two_parameters(args, param1_name, param2_name, metric_name):
    """Analyze two-parameter sweep (creates heatmap)."""
    print(f"\n{'='*60}")
    print(f"Analyzing {param1_name} x {param2_name} vs {metric_name}")
    print(f"{'='*60}")

    # Build fixed parameters dict (excluding the two varying parameters)
    fixed_param_values = {
        'strength': args.fixed_strength,
        'alpha': args.fixed_alpha,
        'Ni': args.fixed_Ni,
        'agents_count': args.fixed_agents_count
    }

    # Remove the varying parameters from fixed params
    for param in [param1_name, param2_name]:
        if param in fixed_param_values:
            del fixed_param_values[param]

    # Create output directory
    fixed_param_labels = []
    for param, value in fixed_param_values.items():
        if param == 'Ni':
            fixed_param_labels.append(f"N_i_{value}")
        elif param == 'agents_count':
            fixed_param_labels.append(f"agents_{value}")
        else:
            fixed_param_labels.append(f"{param}_{value}")

    if fixed_param_labels:
        param_output_dir = os.path.join(args.output_dir, '_'.join(fixed_param_labels))
    else:
        param_output_dir = args.output_dir

    # Get all valid combinations from raw data
    raw_base = os.path.join("data/naive_simulation/raw")

    if not os.path.exists(raw_base):
        print("No simulation data found")
        return

    # Collect all simulations matching fixed parameters
    data_rows = []

    for subdir in os.listdir(raw_base):
        full_path = os.path.join(raw_base, subdir)
        if not os.path.isdir(full_path):
            continue

        # Parse directory
        try:
            # Extract flow type
            if 'bidirectional_flow-' in subdir:
                sim_flow = 'bidirectional'
            elif 'outward_flow-' in subdir:
                sim_flow = 'outward'
            else:
                continue

            if args.flow_type != 'all' and sim_flow != args.flow_type:
                continue

            # Extract nonzero_alpha
            if 'nonzero_alpha_evenly' in subdir:
                sim_nza = 'evenly'
            elif 'nonzero_alpha_center' in subdir:
                sim_nza = 'center'
            else:
                continue

            if args.nonzero_alpha != 'all' and sim_nza != args.nonzero_alpha:
                continue

            # Extract parameters using regex
            sim_params = {}

            strength_match = re.search(r'_fr_([0-9.]+)', subdir)
            if strength_match:
                sim_params['strength'] = float(strength_match.group(1))

            alpha_match = re.search(r'_alpha_([0-9.]+)$', subdir)
            if alpha_match:
                sim_params['alpha'] = float(alpha_match.group(1))

            ni_match = re.search(r'_N_i_([0-9]+)', subdir)
            if ni_match:
                sim_params['Ni'] = int(ni_match.group(1))

            agents_match = re.search(r'_agents_([0-9]+)', subdir)
            if agents_match:
                sim_params['agents_count'] = int(agents_match.group(1))

            # Check if fixed parameters match
            match = True
            for param, value in fixed_param_values.items():
                if param in sim_params:
                    if isinstance(value, float):
                        if abs(sim_params[param] - value) > 1e-9:
                            match = False
                            break
                    else:
                        if sim_params[param] != value:
                            match = False
                            break

            if not match:
                continue

            # Check if both varying parameters are present
            if param1_name not in sim_params or param2_name not in sim_params:
                continue

            param1_val = sim_params[param1_name]
            param2_val = sim_params[param2_name]

            # Filter by DEFAULT_PARAMS
            if param1_name in DEFAULT_PARAMS and param1_val not in DEFAULT_PARAMS[param1_name]:
                continue
            if param2_name in DEFAULT_PARAMS and param2_val not in DEFAULT_PARAMS[param2_name]:
                continue

            # Load and compute metrics
            cache_file = os.path.join(full_path, f"metrics_cache_skip{args.skip}.pkl")

            if args.cache and os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    metrics = pickle.load(f)
            else:
                metrics = load_and_compute_metrics(full_path, args.skip, args.max_snapshots)
                if metrics is None:
                    continue
                with open(cache_file, 'wb') as f:
                    pickle.dump(metrics, f)

            # Extract metrics
            if metric_name == 'diversity':
                agg_metrics = aggregate_metrics(metrics['diversity'], 'mean')
                for key, value in agg_metrics.items():
                    data_rows.append({
                        param1_name: param1_val,
                        param2_name: param2_val,
                        'flow_type': sim_flow,
                        'nonzero_alpha': sim_nza,
                        'metric_name': key,
                        'metric_value': value
                    })
            elif metric_name == 'distance':
                agg_metrics = aggregate_metrics(metrics['distance'], 'mean')
                for key, value in agg_metrics.items():
                    data_rows.append({
                        param1_name: param1_val,
                        param2_name: param2_val,
                        'flow_type': sim_flow,
                        'nonzero_alpha': sim_nza,
                        'metric_name': key,
                        'metric_value': value
                    })

        except Exception as e:
            continue

    if not data_rows:
        print("No data collected for two-parameter analysis")
        return

    # Create DataFrame
    df = pd.DataFrame(data_rows)

    # Save summary data
    summary_dir = os.path.join(param_output_dir, 'summary_data')
    os.makedirs(summary_dir, exist_ok=True)
    csv_path = os.path.join(summary_dir, f'{param1_name}_{param2_name}_vs_{metric_name}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved summary data to {csv_path}")

    # Create heatmaps for each specific metric
    for specific_metric in df['metric_name'].unique():
        df_metric = df[df['metric_name'] == specific_metric].copy()
        plot_two_parameters_heatmap(df_metric, param1_name, param2_name, specific_metric, param_output_dir)


def main():
    """Main analysis function."""
    args = parse_arguments()

    # Parse parameters
    params = args.param.split(',')

    if args.all:
        # Generate all combinations
        for param in ['strength', 'alpha', 'Ni']:
            for metric in ['diversity', 'distance']:
                analyze_single_parameter(args, param, metric)
    elif len(params) == 1:
        # Single parameter analysis
        metrics = [args.metric] if args.metric != 'all' else ['diversity', 'distance']
        for metric in metrics:
            analyze_single_parameter(args, params[0], metric)
    elif len(params) == 2:
        # Two parameter analysis (heatmap)
        metrics = [args.metric] if args.metric != 'all' else ['diversity', 'distance']
        for metric in metrics:
            analyze_two_parameters(args, params[0], params[1], metric)
    else:
        print("Error: Too many parameters specified")
        return

    # Determine output directory based on parameters
    fixed_params = []
    params_list = args.param.split(',')
    if 'Ni' not in params_list:
        fixed_params.append(f"N_i_{args.fixed_Ni}")
    if 'strength' not in params_list:
        fixed_params.append(f"strength_{args.fixed_strength}")
    if 'alpha' not in params_list:
        fixed_params.append(f"alpha_{args.fixed_alpha}")

    if fixed_params:
        final_output_dir = os.path.join(args.output_dir, '_'.join(fixed_params))
    else:
        final_output_dir = args.output_dir

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"Results saved to: {final_output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
