import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import argparse
import scipy.stats as stats
from tqdm import tqdm
from sklearn.metrics import classification_report
import pandas as pd
import statsmodels.api as sm
import matplotlib.colors as colors
from matplotlib.colors import BoundaryNorm, ListedColormap
import itertools
import pickle
import collections

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Language evolution analysis script')

    # Basic parameters
    parser.add_argument('--nonzero_alpha', '-a', type=str, default='all',
                       choices=['evenly', 'center', 'all'],
                       help='nonzero_alpha: "evenly", "center", or "all"')
    parser.add_argument('--flow_type', '-f', type=str, default='all',
                       choices=['bidirectional', 'outward', 'all'],
                       help='flow_type: "bidirectional", "outward", or "all"')
    parser.add_argument('--skip', '-s', type=int, default=0,
                       help='Number of files to skip (default: 0)')
    parser.add_argument('--agent_id', '-i', type=int, default=0,
                       help='Agent ID for bar plot display (default: 0)')
    parser.add_argument('--agents_count', '-m', type=int, default=15,
                       help='Number of agents (default: 15)')
    parser.add_argument('--N_i', '-n', type=int, default=100,
                       help='Number of data per subpopulation (default: 100)')
    parser.add_argument('--coupling_strength', '-c', type=float, default=0.01,
                       help='Coupling strength (default: 0.01)')
    parser.add_argument('--alpha_per_data', type=float, default=0.001, help='New word generation bias')
    parser.add_argument('--recompute_mean_similarity', action='store_true',
                       help='If set, recompute mean similarity even if saved file exists')
    parser.add_argument('--clear_figure_dir', action='store_true',
                       help='If set, clear figure directory')

    # Analysis options
    parser.add_argument('--check_concentric', action='store_true',
                       help='If set, perform concentric distribution analysis')
    parser.add_argument('--plot_distance', action='store_true',
                       help='If set, plot distance analysis')
    parser.add_argument('--plot_age', action='store_true',
                       help='If set, plot age of words')
    parser.add_argument('--plot_similarity', action='store_true',
                       help='If set, plot similarity analysis')
    parser.add_argument('--make_age_files', action='store_true',
                       help='If set, make age files from state files')
    parser.add_argument('--logistic_regression', action='store_true',
                       help='If set, perform logistic regression analysis')
    parser.add_argument('--pairwise_regression', action='store_true',
                       help='If set, perform pairwise logistic regression and plot heatmaps')
    parser.add_argument('--center_agent', type=int, default=None,
                       help='Index of the center (hub) agent (default: agents_count//2)')
    parser.add_argument('--opposite_agent', type=int, default=None,
                       help='Index of the opposite-side agent (default: agents_count//2 + (agents_count//2)//2)')
    parser.add_argument('--plot_discrete_colorbar', action='store_true',
                       help='If set, add discrete colorbar to distance rank matrix heatmap')

    return parser.parse_known_args()

def get_combinations(nonzero_alpha, flow_type):
    """Get all combinations of nonzero_alpha and flow_type."""
    nonzero_alpha_options = ['evenly', 'center'] if nonzero_alpha == 'all' else [nonzero_alpha]
    flow_type_options = ['bidirectional', 'outward'] if flow_type == 'all' else [flow_type]
    return [(na, ft) for na in nonzero_alpha_options for ft in flow_type_options]

def get_directory_paths(flow_type, nonzero_alpha, coupling_strength, agents_count, N_i, alpha):
    """Get load and save directory paths."""
    if flow_type == 'bidirectional':
        flow_str = 'bidirectional_flow-'
    elif flow_type == 'outward':
        flow_str = 'outward_flow-'
    else:
        raise ValueError(f"Unknown flow_type: {flow_type}")

    subdir = f"{flow_str}nonzero_alpha_{nonzero_alpha}_fr_{coupling_strength}_agents_{agents_count}_N_i_{N_i}_alpha_{alpha}"
    load_dir = f"data/naive_simulation/raw/{subdir}"
    save_dir = f"data/naive_simulation/fig/{subdir}"

    return load_dir, save_dir

def is_concentric_distribution(distance_matrix):
    """Check if the distance matrix shows a concentric distribution."""
    center = len(distance_matrix) // 2

    for base in range(len(distance_matrix)):
        if base == center:
            continue

        for reference in range(len(distance_matrix)):
            is_opposite_side = (base - center) * (reference - center) < 0
            if is_opposite_side and distance_matrix[base][reference] < distance_matrix[base][center]:
                return True
    return False

def create_age_files(load_dir, save_dir, state_files):
    """Create age statistics files from state files."""
    print("Generating age files...")
    idx_t_map = np.loadtxt(os.path.join(load_dir, "save_idx_t_map.csv"),
                          delimiter=',', dtype=int, skiprows=1)
    id2t = {id_: t for id_, t in idx_t_map}

    if not state_files:
        print(f"No state files found in {load_dir}. Skipping...")
        return

    # Per-agent age statistics
    age_means_per_agent = []
    age_vars_per_agent = []
    file_ids_per_agent = []
    agent_ids_per_agent = []

    for state_file in tqdm(state_files):
        basename = os.path.basename(state_file)
        file_id = int(basename.split('_')[1].split('.')[0])
        file_t = id2t[file_id]
        state = np.load(state_file)
        word_ts = state[..., 0]
        ages = file_t - word_ts

        for agent in range(ages.shape[0]):
            age_means_per_agent.append(np.mean(ages[agent]))
            age_vars_per_agent.append(np.var(ages[agent]))
            file_ids_per_agent.append(file_id)
            agent_ids_per_agent.append(agent)

    # Save per-agent statistics
    df_age_mean = pd.DataFrame({
        'file_id': file_ids_per_agent,
        'agent_id': agent_ids_per_agent,
        'age_mean': age_means_per_agent
    })
    df_age_mean.to_csv(os.path.join(save_dir, 'word_age_mean_per_agent.csv'), index=False)

    df_age_var = pd.DataFrame({
        'file_id': file_ids_per_agent,
        'agent_id': agent_ids_per_agent,
        'age_var': age_vars_per_agent
    })
    df_age_var.to_csv(os.path.join(save_dir, 'word_age_var_per_agent.csv'), index=False)

    # Overall age statistics
    age_means = []
    age_vars = []
    file_ids = []

    for state_file in tqdm(state_files):
        basename = os.path.basename(state_file)
        file_id = int(basename.split('_')[1].split('.')[0])
        file_ids.append(file_id)
        file_t = id2t[file_id]
        state = np.load(state_file)
        word_ts = state[..., 0]
        ages = file_t - word_ts
        ages_flat = ages.flatten()
        age_means.append(np.mean(ages_flat))
        age_vars.append(np.var(ages_flat))

    df_age = pd.DataFrame({
        'file_id': file_ids,
        'age_mean': age_means,
        'age_var': age_vars
    })
    df_age.to_csv(os.path.join(save_dir, 'word_age_stats.csv'), index=False)
    print(f"Saved age statistics to {save_dir}")

def calculate_genetic_similarity(state):
    """Calculate genetic similarity between agents."""
    agents_count, N_i, _ = state.shape

    # Calculate meme frequencies for each agent
    freq_list = []
    for i in range(agents_count):
        counts = collections.Counter(tuple(row) for row in state[i])
        freq_dict = {meme: count / N_i for meme, count in counts.items()}
        freq_list.append(freq_dict)

    # Initialize similarity matrices
    dot_product_matrix = np.zeros((agents_count, agents_count))
    cosine_similarity_matrix = np.zeros((agents_count, agents_count))

    # Calculate similarities for all agent pairs
    for i in range(agents_count):
        for j in range(i, agents_count):
            freq_i = freq_list[i]
            freq_j = freq_list[j]

            all_memes = set(freq_i.keys()) | set(freq_j.keys())

            # Dot product calculation
            dot_product = sum(freq_i.get(meme, 0) * freq_j.get(meme, 0) for meme in all_memes)
            dot_product_matrix[i, j] = dot_product_matrix[j, i] = dot_product

            # Cosine similarity calculation
            norm_i = np.sqrt(sum(p**2 for p in freq_i.values()))
            norm_j = np.sqrt(sum(p**2 for p in freq_j.values()))

            if norm_i > 0 and norm_j > 0:
                cosine_sim = dot_product / (norm_i * norm_j)
            else:
                cosine_sim = 0.0

            cosine_similarity_matrix[i, j] = cosine_similarity_matrix[j, i] = cosine_sim

    return dot_product_matrix, cosine_similarity_matrix

def load_similarity_data(load_dir, force_recompute=False):
    """Load dot product and cosine similarity data, with caching."""
    mean_dot_path = os.path.join(load_dir, "mean_similarity_dot.npy")
    mean_cosine_path = os.path.join(load_dir, "mean_similarity_cosine.npy")

    if not force_recompute and os.path.exists(mean_dot_path) and os.path.exists(mean_cosine_path):
        mean_similarity_dot = np.load(mean_dot_path)
        mean_similarity_cosine = np.load(mean_cosine_path)
        print("Loaded mean similarities from file.")
        return (mean_similarity_dot, mean_similarity_cosine, None, None)

    # Compute from state files
    state_files = sorted(glob.glob(os.path.join(load_dir, "state_*.npy")))
    if not state_files:
        return (None, None, None, None)

    print(f"Computing similarities from {len(state_files)} state files...")

    dot_similarities = []
    cosine_similarities = []

    for state_file in tqdm(state_files):
        state = np.load(state_file)
        dot_sim, cosine_sim = calculate_genetic_similarity(state)
        dot_similarities.append(dot_sim)
        cosine_similarities.append(cosine_sim)

    dot_similarities = np.array(dot_similarities)
    cosine_similarities = np.array(cosine_similarities)

    mean_similarity_dot = np.mean(dot_similarities, axis=0)
    mean_similarity_cosine = np.mean(cosine_similarities, axis=0)

    # Save computed means
    np.save(mean_dot_path, mean_similarity_dot)
    np.save(mean_cosine_path, mean_similarity_cosine)
    print(f"Saved mean similarities to {load_dir}")

    return (mean_similarity_dot, mean_similarity_cosine, dot_similarities, cosine_similarities)

def plot_histogram_comparison(data1, data2, labels, colors, save_path, title_suffix="",
                            log_scale=False, shift_amount=None):
    """Plot histogram comparison between two datasets."""
    plt.figure(figsize=(5, 5))
    all_data = np.concatenate([data1, data2])
    bin_edges = np.linspace(np.min(all_data), np.max(all_data), 202)
    bin_width = bin_edges[1] - bin_edges[0]
    if shift_amount is None:
        shift_amount = bin_width / 2

    for i, (data, label, color) in enumerate(zip([data1, data2], labels, colors)):
        mean_val = np.mean(data)
        shift = -shift_amount if i == 0 else shift_amount
        plt.hist(data + shift, bins=bin_edges + shift, label=label,
                color=color, density=True)
        plt.axvline(mean_val + shift, color=color, linestyle='--',
                   label=f'Mean {label}: {mean_val:.3f} (shifted)')

    if log_scale:
        plt.yscale('log')
        plt.grid(True, alpha=0.3, which='both')
    else:
        plt.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_mean_distance_analysis(mean_distance, save_dir, agent_id, N_i, center_agent, plot_discrete_colorbar=False):
    """Plot mean distance heatmap analysis."""
    # Normalize
    mean_distance = mean_distance / (2 * N_i)
    extent = [0, mean_distance.shape[0], 0, mean_distance.shape[1]]

    # Linear scale heatmap without colorbar
    plt.figure(figsize=(5, 5))
    im = plt.imshow(mean_distance, extent=extent,
                    cmap='Blues', aspect='equal', vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(save_dir, "mean_distance_heatmap_Blues_no_colorbar.png"), dpi=300)
    plt.close()

    # Various colormaps with colorbar
    for cmap in ['Blues', 'Reds', 'Greens', 'Blues_Reds']:
        plt.figure(figsize=(5, 5))
        if cmap == 'Blues_Reds':
            custom_cmap = colors.LinearSegmentedColormap.from_list('Blues_Reds', ['blue','red'])
            im = plt.imshow(mean_distance, extent=extent,
                            cmap=custom_cmap, aspect='equal', vmin=0, vmax=1)
            filename = "mean_distance_heatmap_Blues_Reds.png"
        else:
            im = plt.imshow(mean_distance, extent=extent,
                            cmap=cmap, aspect='equal', vmin=0, vmax=1)
            filename = f"mean_distance_heatmap_{cmap}.png"
        plt.colorbar(im)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(save_dir, filename), dpi=300)
        plt.close()

    # Agent-specific distance plot
    plt.figure(figsize=(5, 5))
    plt.plot(np.arange(mean_distance.shape[0]), mean_distance[agent_id], marker='o')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(save_dir, f"mean_distance_from_agent{agent_id}.png"), dpi=300)
    plt.close()

    # Distance rank matrix
    rank_matrix = mean_distance.argsort(axis=1).argsort(axis=1) + 1
    is_concentric = is_concentric_distribution(mean_distance)

    for cmap in ['Blues', 'Reds', 'Greens', 'bwr', 'Blues_Reds', 'Greys', 'White_Blue']:
        filename = f'distance_rank_matrix_heatmap_{cmap}_with_border.png'

        fig, ax = plt.subplots(figsize=(5, 5))

        if cmap == 'Blues_Reds':
            custom_cmap = colors.LinearSegmentedColormap.from_list('Blues_Reds', ['blue', 'red'])
            im = ax.imshow(rank_matrix, cmap=custom_cmap, aspect='equal')
        elif cmap == 'White_Blue':
            custom_cmap = colors.LinearSegmentedColormap.from_list('White_Blue', ['white', 'blue'])
            im = ax.imshow(rank_matrix, cmap=custom_cmap, aspect='equal')
        else:
            im = ax.imshow(rank_matrix, cmap=cmap, aspect='equal')

        ax.set_xticks([])
        ax.set_yticks([])

        # Add borders for center agent and concentric violations
        rows, cols = rank_matrix.shape
        center_edgecolor = "black"
        concentric_cell_edgecolor = "red"
        if cols > center_agent:
            for i in range(rows):
                for j in range(cols):
                    if j == center_agent and i != center_agent:
                        rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            fill=False,
                                            edgecolor=center_edgecolor,
                                            linewidth=2)
                        ax.add_patch(rect)

                    if rank_matrix[i, center_agent] > rank_matrix[i, j] and (i - center_agent) * (j - center_agent) < 0:
                        rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            fill=False,
                                            edgecolor=concentric_cell_edgecolor,
                                            linewidth=2)
                        ax.add_patch(rect)

        plt.savefig(os.path.join(save_dir, filename), dpi=300)
        plt.close(fig)

def plot_distance_analysis(distances, save_dir, N_i, center_agent, opposite_agent):
    """Plot comprehensive distance analysis."""
    # Normalize
    norm_factor = 2 * N_i
    distances_0_center = distances[:, 0, center_agent] / norm_factor
    distances_0_opposite = distances[:, 0, opposite_agent] / norm_factor

    # Save distance data
    df_distances = pd.DataFrame({
        f'distance_0_{center_agent}': distances_0_center,
        f'distance_0_{opposite_agent}': distances_0_opposite
    })
    df_distances.to_csv(os.path.join(save_dir, f'distances_0_{center_agent}_0_{opposite_agent}.csv'), index=False)

    # Histogram comparison
    plot_histogram_comparison(
        distances_0_center, distances_0_opposite,
        [f'Agent 0-{center_agent}', f'Agent 0-{opposite_agent}'],
        ['blue', 'red'],
        os.path.join(save_dir, 'agent_pair_distances_histogram.png')
    )

    # Log scale histogram
    plot_histogram_comparison(
        distances_0_center, distances_0_opposite,
        [f'Agent 0-{center_agent}', f'Agent 0-{opposite_agent}'],
        ['blue', 'red'],
        os.path.join(save_dir, 'agent_pair_distances_histogram_log.png'),
        log_scale=True
    )

def plot_age_analysis(save_dir):
    """Plot age analysis."""
    age_mean_file = os.path.join(save_dir, 'word_age_mean_per_agent.csv')
    if not os.path.exists(age_mean_file):
        print(f"Age file {age_mean_file} not found. Skipping age analysis.")
        return

    df = pd.read_csv(age_mean_file)
    mean_by_agent = df.groupby('agent_id')['age_mean'].mean()
    std_by_agent = df.groupby('agent_id')['age_mean'].std()

    # Mean only plot
    plt.figure(figsize=(5, 5))
    plt.plot(mean_by_agent.index, mean_by_agent.values, marker='o')
    plt.xticks([])
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'age_mean_timeavg_per_agent.png'), dpi=300)
    plt.close()

    # Mean with standard deviation
    plt.figure(figsize=(5, 5))
    plt.plot(mean_by_agent.index, mean_by_agent.values, marker='o', label='Mean')
    plt.fill_between(mean_by_agent.index,
                     mean_by_agent.values - std_by_agent.values,
                     mean_by_agent.values + std_by_agent.values,
                     color='blue', alpha=0.2, label='Std')
    plt.xticks([])
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'age_mean_timeavg_per_agent_with_std.png'), dpi=300)
    plt.close()

def plot_similarity_analysis(similarities, mean_similarity, save_dir, similarity_type, center_agent, opposite_agent):
    """Plot similarity analysis."""
    if similarities is None or mean_similarity is None:
        return None, None

    # Heatmap (always available with mean data)
    plt.figure(figsize=(5, 5))
    vmax = 1 if similarity_type == 'cosine' else None
    im = plt.imshow(mean_similarity, vmin=0, vmax=vmax, aspect="equal", cmap='viridis')
    plt.colorbar(im, label=f"Mean {similarity_type.title()} Similarity")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(save_dir, f"mean_{similarity_type}_similarity_heatmap.png"), dpi=300)
    plt.close()

    # Agent 0 similarity plot
    plt.figure(figsize=(5, 5))
    plt.plot(np.arange(mean_similarity.shape[0]), mean_similarity[0], marker='o')
    plt.xticks([])
    plt.savefig(os.path.join(save_dir, f"mean_{similarity_type}_similarity_from_agent0.png"), dpi=300)
    plt.close()

    return None, None

def perform_binomial_tests(distances_0_center, distances_0_opposite, similarities_dot_0_center, similarities_dot_0_opposite,
                          similarities_cosine_0_center, similarities_cosine_0_opposite, mean_distance,
                          mean_similarity_dot, mean_similarity_cosine, center_agent, opposite_agent):
    """Perform binomial tests for distance and similarity comparisons."""
    print("=== Distance Binomial test results ===")

    # Distance tests
    over_count = np.sum(distances_0_center > distances_0_opposite)
    under_count = np.sum(distances_0_center < distances_0_opposite)
    equal_count = np.sum(distances_0_center == distances_0_opposite)
    total = len(distances_0_center)

    p_value = stats.binomtest(over_count, n=total, p=0.5, alternative='two-sided').pvalue

    print(f"Number of times 0-{center_agent} > 0-{opposite_agent}: {over_count}/{total}")
    print(f"Number of times 0-{center_agent} < 0-{opposite_agent}: {under_count}/{total}")
    print(f"Number of times 0-{center_agent} = 0-{opposite_agent}: {equal_count}/{total}")
    print(f"p-value: {p_value:.3e}")
    print(f"mean distance is concentric: {is_concentric_distribution(mean_distance)}")

def main():
    """Main function to orchestrate the analysis."""
    args, unknown = parse_arguments()

    combinations = get_combinations(args.nonzero_alpha, args.flow_type)

    # Compute defaults for center and opposite agents if not provided
    if args.center_agent is None:
        args.center_agent = args.agents_count // 2
    if args.opposite_agent is None:
        half = args.agents_count // 2
        args.opposite_agent = half + (half // 2)

    for na, ft in combinations:
        print(f"\n{'='*50}")
        print(f"Processing: {na} + {ft}")
        print(f"{'='*50}")

        load_dir, save_dir = get_directory_paths(
            ft, na, args.coupling_strength, args.agents_count, args.N_i, args.alpha_per_data
        )
        os.makedirs(save_dir, exist_ok=True)

        distance_files = sorted(glob.glob(os.path.join(load_dir, "distance_*.npy")))
        print(f"Number of distance files found: {len(distance_files)}")

        if not distance_files:
            print(f"No distance files found in {load_dir}. Skipping...")
            continue

        distance_files = distance_files[args.skip:]

        # Create age files proactively if requested or required by plotting
        if args.make_age_files:
            state_files = sorted(glob.glob(os.path.join(load_dir, "state_*.npy")))
            create_age_files(load_dir, save_dir, state_files)
        elif args.plot_age:
            age_mean_file = os.path.join(save_dir, 'word_age_mean_per_agent.csv')
            if not os.path.exists(age_mean_file):
                state_files = sorted(glob.glob(os.path.join(load_dir, "state_*.npy")))
                if state_files:
                    print("Age files not found; generating them for plot_age...")
                    create_age_files(load_dir, save_dir, state_files)
                else:
                    print("No state_*.npy files found; cannot generate age files for plot_age.")

        # Load similarity data if needed
        mean_similarity_dot = None
        mean_similarity_cosine = None
        dot_similarities = None
        cosine_similarities = None

        if args.plot_similarity or args.check_concentric:
            (mean_similarity_dot, mean_similarity_cosine,
             dot_similarities, cosine_similarities) = load_similarity_data(
                load_dir, force_recompute=args.recompute_mean_similarity
            )

        mean_distance = None
        distances = None
        if args.plot_distance:
            print(f"Loading distance data from {len(distance_files)} files...")
            distances = np.array([np.load(f) for f in distance_files])
            mean_distance = np.mean(distances, axis=0)
            print("Distance data loaded successfully.")

        if args.plot_distance and mean_distance is not None:
            # Mean distance analysis
            plot_mean_distance_analysis(mean_distance, save_dir, args.agent_id, args.N_i, args.center_agent, args.plot_discrete_colorbar)
            # Detailed distance analysis
            if distances is not None:
                plot_distance_analysis(distances, save_dir, args.N_i, args.center_agent, args.opposite_agent)

        if args.plot_age:
            plot_age_analysis(save_dir)

        if args.plot_similarity:
            plot_similarity_analysis(dot_similarities, mean_similarity_dot, save_dir, 'dot', args.center_agent, args.opposite_agent)
            plot_similarity_analysis(cosine_similarities, mean_similarity_cosine, save_dir, 'cosine', args.center_agent, args.opposite_agent)

        if args.check_concentric:
            print("Checking concentric distribution patterns...")
            # Memory efficient concentric check
            concentric_results = []
            for f in distance_files:
                d = np.load(f)
                is_concentric = is_concentric_distribution(d)
                concentric_results.append(is_concentric)
            concentric_rate = np.mean(concentric_results)
            print(f"Concentric distribution rate: {concentric_rate:.3f}")

            # Binomial tests for distance comparison
            if len(distance_files) > 0:
                distances_0_center = []
                distances_0_opposite = []

                for f in distance_files:
                    snapshot_d = np.load(f)
                    distances_0_center.append(snapshot_d[0, args.center_agent] / (2 * args.N_i))
                    distances_0_opposite.append(snapshot_d[0, args.opposite_agent] / (2 * args.N_i))

                distances_0_center = np.array(distances_0_center)
                distances_0_opposite = np.array(distances_0_opposite)

                # Perform binomial test
                perform_binomial_tests(
                    distances_0_center, distances_0_opposite,
                    None, None, None, None,  # Similarity data not computed here
                    mean_distance, mean_similarity_dot, mean_similarity_cosine,
                    args.center_agent, args.opposite_agent
                )

if __name__ == "__main__":
    main()