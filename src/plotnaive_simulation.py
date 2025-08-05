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
    
    return parser.parse_known_args()


def get_combinations(nonzero_alpha, flow_type):
    """Get all combinations of nonzero_alpha and flow_type."""
    nonzero_alpha_options = ['evenly', 'center'] if nonzero_alpha == 'all' else [nonzero_alpha]
    flow_type_options = ['bidirectional', 'outward'] if flow_type == 'all' else [flow_type]
    return [(na, ft) for na in nonzero_alpha_options for ft in flow_type_options]


def get_directory_paths(flow_type, nonzero_alpha, coupling_strength, agents_count, N_i):
    """Get load and save directory paths."""
    if flow_type == 'bidirectional':
        flow_str = 'bidirectional_flow-'
    elif flow_type == 'outward':
        flow_str = 'outward_flow-'
    else:
        raise ValueError(f"Unknown flow_type: {flow_type}")
    
    subdir = f"{flow_str}nonzero_alpha_{nonzero_alpha}_fr_{coupling_strength}_agents_{agents_count}_N_i_{N_i}"
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


def load_similarity_data(load_dir, force_recompute=False):
    """Load dot product and cosine similarity data, with caching."""
    mean_dot_path = os.path.join(load_dir, "mean_similarity_dot.npy")
    mean_cosine_path = os.path.join(load_dir, "mean_similarity_cosine.npy")

    mean_similarity_dot = None
    mean_similarity_cosine = None
    similarities_dot = None
    similarities_cosine = None

    # 既存ファイルがあればロード
    if not force_recompute and os.path.exists(mean_dot_path) and os.path.exists(mean_cosine_path):
        mean_similarity_dot = np.load(mean_dot_path)
        mean_similarity_cosine = np.load(mean_cosine_path)
        print("Loaded mean similarities from file.")
        return (mean_similarity_dot, mean_similarity_cosine, None, None)

    # なければ計算
    dot_sim_files = sorted(glob.glob(os.path.join(load_dir, "similarity_dot_*.npy")))
    cosine_sim_files = sorted(glob.glob(os.path.join(load_dir, "similarity_cosine_*.npy")))

    if dot_sim_files:
        print(f"Found {len(dot_sim_files)} dot similarity files. Averaging...")
        similarities_dot = np.stack([np.load(f) for f in dot_sim_files], axis=0)
        mean_similarity_dot = similarities_dot.mean(axis=0)
        np.save(mean_dot_path, mean_similarity_dot)
    if cosine_sim_files:
        print(f"Found {len(cosine_sim_files)} cosine similarity files. Averaging...")
        similarities_cosine = np.stack([np.load(f) for f in cosine_sim_files], axis=0)
        mean_similarity_cosine = similarities_cosine.mean(axis=0)
        np.save(mean_cosine_path, mean_similarity_cosine)

    return (mean_similarity_dot, mean_similarity_cosine, similarities_dot, similarities_cosine)


def plot_histogram_comparison(data1, data2, labels, colors, save_path, title_suffix="", 
                            log_scale=False, shift_amount=None):
    """Plot histogram comparison between two datasets."""
    plt.figure(figsize=(5, 5))
    # bin幅を計算
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
    plt.show()


def plot_distance_analysis(distances, save_dir, N_i):
    """Plot comprehensive distance analysis."""
    # 正規化
    norm_factor = 2 * N_i
    distances_0_7 = distances[:, 0, 7] / norm_factor
    distances_0_10 = distances[:, 0, 10] / norm_factor
    
    # Save distance data
    df_distances = pd.DataFrame({
        'distance_0_7': distances_0_7,
        'distance_0_10': distances_0_10
    })
    df_distances.to_csv(os.path.join(save_dir, 'distances_0_7_0_10.csv'), index=False)
    
    # Histogram comparison
    plot_histogram_comparison(
        distances_0_7, distances_0_10, 
        ['Agent 0-7', 'Agent 0-10'], 
        ['blue', 'red'],
        os.path.join(save_dir, 'agent_pair_distances_histogram.png')
    )
    
    # Log scale histogram
    plot_histogram_comparison(
        distances_0_7, distances_0_10, 
        ['Agent 0-7', 'Agent 0-10'], 
        ['blue', 'red'],
        os.path.join(save_dir, 'agent_pair_distances_histogram_log.png'),
        log_scale=True
    )
    
    # Distance difference histogram
    diff_distances = distances_0_7 - distances_0_10
    plt.figure(figsize=(5, 5))
    bin_edges = np.linspace(np.min(diff_distances), np.max(diff_distances), 402)
    mean_diff = np.mean(diff_distances)
    
    plt.hist(diff_distances, bins=bin_edges, alpha=0.75, 
            label='d(0, 7) - d(0, 10)', color='green', density=True)
    plt.axvline(mean_diff, color='blue', linestyle='--', linewidth=0.5, 
               label=f'Mean Difference: {mean_diff:.2f}')
    # plt.xlabel('Distance Difference (d(0,7) - d(0,10))')
    # plt.ylabel('Density')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'distance_difference_histogram.png'), dpi=300)
    plt.show()
    
    # Scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(distances_0_7, distances_0_10, alpha=0.2, s=15, edgecolors='none')
    max_val = max(np.max(distances_0_7), np.max(distances_0_10))
    min_val = min(np.min(distances_0_7), np.min(distances_0_10))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='d(0,7) = d(0,10)')
    # plt.xlabel('Distance d(0, 7)')
    # plt.ylabel('Distance d(0, 10)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.savefig(os.path.join(save_dir, 'distance_scatter_plot.png'), dpi=300)
    plt.show()
    
    # Bubble charts and heatmaps
    plot_bubble_charts(distances_0_7, distances_0_10, save_dir)
    plot_2d_heatmaps(distances_0_7, distances_0_10, save_dir)

    # --- Rank Matrix Heatmap ---
    # 距離行列の平均を計算
    mean_distance_matrix = distances.mean(axis=0)
    # 各行ごとに値が大きいほど順位が大きくなるように順位を計算（1始まり）
    rank_matrix = mean_distance_matrix.argsort(axis=1).argsort(axis=1) + 1
    plt.figure(figsize=(5, 5))
    im = plt.imshow(rank_matrix, cmap='Blues', aspect='equal')
    plt.colorbar(im)
    plt.xticks([])
    plt.yticks([])
    plt.gca()
    # plt.title('Rank Matrix: iから見たjの距離順位（大きいほど順位大）')
    # plt.xlabel('Agent j')
    # plt.ylabel('Agent i')
    plt.savefig(os.path.join(save_dir, 'distance_rank_matrix_heatmap.png'), dpi=300)
    plt.show()


def plot_bubble_charts(distances_0_7, distances_0_10, save_dir):
    """Plot bubble charts for distance data."""
    df_scatter = pd.DataFrame({
        'd_0_7': distances_0_7,
        'd_0_10': distances_0_10
    })
    bubble_data = df_scatter.groupby(['d_0_7', 'd_0_10']).size().reset_index(name='count')
    
    # Linear bubble chart
    plt.figure(figsize=(7, 6))
    scale_factor = 5
    bubble_sizes = bubble_data['count'] * scale_factor
    plt.scatter(bubble_data['d_0_7'], bubble_data['d_0_10'], s=bubble_sizes, 
               alpha=0.1, edgecolors="w", linewidth=0.5)
    # plt.xlabel('Distance d(0, 7)')
    # plt.ylabel('Distance d(0, 10)')
    plt.grid(True, alpha=0.3, zorder=-1)
    plt.axis('equal')
    plt.savefig(os.path.join(save_dir, 'distance_bubble_exact_counts_plot.png'), dpi=300)
    plt.show()
    
    # Log bubble chart
    plt.figure(figsize=(7, 6))
    bubble_sizes_log = np.log(1 + bubble_data['count'] * scale_factor)
    plt.scatter(bubble_data['d_0_7'], bubble_data['d_0_10'], s=bubble_sizes_log, 
               alpha=0.8, edgecolors="w", linewidth=0.5)
    # plt.xlabel('Distance d(0, 7)')
    # plt.ylabel('Distance d(0, 10)')
    plt.grid(True, alpha=0.3, zorder=-1)
    plt.axis('equal')
    plt.savefig(os.path.join(save_dir, 'distance_log_bubble_exact_counts_plot.png'), dpi=300)
    plt.show()


def plot_2d_heatmaps(distances_0_7, distances_0_10, save_dir):
    """Plot 2D heatmaps for distance data."""
    bins = 50
    counts, xedges, yedges = np.histogram2d(distances_0_7, distances_0_10, bins=bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    # Linear scale heatmap
    plt.figure(figsize=(5, 5))
    im = plt.imshow(counts.T, extent=extent, 
                   cmap='viridis', aspect='equal')
    plt.colorbar(im)
    # plt.xlabel('Distance d(0, 7)')
    # plt.ylabel('Distance d(0, 10)')
    plt.savefig(os.path.join(save_dir, 'distance_heatmap_linear.png'), dpi=300)
    plt.show()
    
    # Log scale heatmaps with different colormaps
    for cmap in ['Blues', 'Reds', 'Greens']:
        plt.figure(figsize=(5, 5))
        counts_masked = np.ma.masked_where(counts == 0, counts)
        im_log = plt.imshow(counts_masked.T, extent=extent, 
                           cmap=cmap, aspect='equal', norm=colors.LogNorm())
        plt.colorbar(im_log)
        # plt.xlabel('Distance d(0, 7)')
        # plt.ylabel('Distance d(0, 10)')
        plt.savefig(os.path.join(save_dir, f'distance_heatmap_log_{cmap}.png'), dpi=300)
        plt.show()


def plot_mean_distance_analysis(mean_distance, save_dir, agent_id, N_i):
    """Plot mean distance heatmap in the same format as plot_2d_heatmaps."""
    # 正規化
    mean_distance = mean_distance / (2 * N_i)
    extent = [0, mean_distance.shape[0], 0, mean_distance.shape[1]]

    # Linear scale heatmap
    plt.figure(figsize=(5, 5))
    im = plt.imshow(mean_distance, extent=extent, 
                    cmap='Blues', aspect='equal')
    plt.colorbar(im)
    # plt.xlabel('Agent i')
    # plt.ylabel('Agent j')
    plt.savefig(os.path.join(save_dir, "mean_distance_heatmap_Blues.png"), dpi=300)
    plt.show()

    for cmap in ['Blues', 'Reds', 'Greens', 'Blues_Reds']:
        plt.figure(figsize=(5, 5))
        if cmap == 'Blues_Reds':
            custom_cmap = colors.LinearSegmentedColormap.from_list('Blues_Reds', ['blue','red'])
            im = plt.imshow(mean_distance, extent=extent, 
                            cmap=custom_cmap, aspect='equal')
            filename = "mean_distance_heatmap_Blues_Reds.png"
        else:
            im = plt.imshow(mean_distance, extent=extent, 
                            cmap=cmap, aspect='equal')
            filename = f"mean_distance_heatmap_{cmap}.png"
        plt.colorbar(im)
        # plt.xlabel('Agent i')
        # plt.ylabel('Agent j')
        plt.savefig(os.path.join(save_dir, filename), dpi=300)
        plt.show()
    
    # Agent-specific distance plot
    plt.figure(figsize=(5, 5))
    plt.plot(np.arange(mean_distance.shape[0]), mean_distance[agent_id], marker='o')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(save_dir, f"mean_distance_from_agent{agent_id}.png"), dpi=300)
    plt.show()


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
    plt.show()
    
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
    plt.show()


def plot_similarity_analysis(similarities, mean_similarity, save_dir, similarity_type):
    """Plot similarity analysis (dot product or cosine)."""
    if similarities is None or mean_similarity is None:
        return None, None
    
    similarities_0_7 = similarities[:, 0, 7]
    similarities_0_10 = similarities[:, 0, 10]
    
    # Save similarity data
    df_similarities = pd.DataFrame({
        f'{similarity_type}_similarity_0_7': similarities_0_7,
        f'{similarity_type}_similarity_0_10': similarities_0_10
    })
    csv_path = os.path.join(save_dir, f'{similarity_type}_similarities_0_7_0_10.csv')
    df_similarities.to_csv(csv_path, index=False)
    
    # Histogram comparison
    shift_amount = 0.005 if similarity_type == 'dot' else 0.001
    plot_histogram_comparison(
        similarities_0_7, similarities_0_10,
        ['Agent 0-7', 'Agent 0-10'],
        ['blue', 'red'],
        os.path.join(save_dir, f'agent_pair_{similarity_type}_similarities_histogram.png'),
        shift_amount=shift_amount
    )
    
    # Heatmap
    plt.figure(figsize=(5, 5))
    vmax = 1 if similarity_type == 'cosine' else None
    im = plt.imshow(mean_similarity, vmin=0, vmax=vmax, aspect="equal", cmap='viridis')
    plt.colorbar(im, label=f"Mean {similarity_type.title()} Similarity")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(save_dir, f"mean_{similarity_type}_similarity_heatmap.png"), dpi=300)
    plt.show()
    
    # Agent 0 similarity plot
    plt.figure(figsize=(5, 5))
    plt.plot(np.arange(mean_similarity.shape[0]), mean_similarity[0], marker='o')
    plt.xticks([])
    plt.savefig(os.path.join(save_dir, f"mean_{similarity_type}_similarity_from_agent0.png"), dpi=300)
    plt.show()
    
    return similarities_0_7, similarities_0_10


def perform_binomial_tests(distances_0_7, distances_0_10, similarities_dot_0_7, similarities_dot_0_10,
                          similarities_cosine_0_7, similarities_cosine_0_10, mean_distance,
                          mean_similarity_dot, mean_similarity_cosine):
    """Perform binomial tests for distance and similarity comparisons."""
    print("=== Distance Binomial test results ===")
    
    # Distance tests
    over_count = np.sum(distances_0_7 > distances_0_10)
    under_count = np.sum(distances_0_7 < distances_0_10)
    equal_count = np.sum(distances_0_7 == distances_0_10)
    total = len(distances_0_7)
    
    p_value = stats.binomtest(over_count, n=total, p=0.5, alternative='two-sided').pvalue
    
    print(f"Number of times 0-7 > 0-10: {over_count}/{total}")
    print(f"Number of times 0-7 < 0-10: {under_count}/{total}")
    print(f"Number of times 0-7 = 0-10: {equal_count}/{total}")
    print(f"p-value: {p_value:.3e}")
    print(f"mean distance is concentric: {is_concentric_distribution(mean_distance)}")
    
    # Similarity tests
    if similarities_dot_0_7 is not None:
        print("\n=== Dot Product Similarity Binomial test results ===")
        over_count_dot = np.sum(similarities_dot_0_7 > similarities_dot_0_10)
        under_count_dot = np.sum(similarities_dot_0_7 < similarities_dot_0_10)
        total_dot = len(similarities_dot_0_7)
        p_value_dot = stats.binomtest(over_count_dot, n=total_dot, p=0.5, alternative='two-sided').pvalue
        
        print(f"Number of times 0-7 > 0-10: {over_count_dot}/{total_dot}")
        print(f"p-value: {p_value_dot:.3e}")
        print(f"mean dot similarity is concentric: {is_concentric_distribution(-mean_similarity_dot)}")
    
    if similarities_cosine_0_7 is not None:
        print("\n=== Cosine Similarity Binomial test results ===")
        over_count_cosine = np.sum(similarities_cosine_0_7 > similarities_cosine_0_10)
        under_count_cosine = np.sum(similarities_cosine_0_7 < similarities_cosine_0_10)
        total_cosine = len(similarities_cosine_0_7)
        p_value_cosine = stats.binomtest(over_count_cosine, n=total_cosine, p=0.5, alternative='two-sided').pvalue
        
        print(f"Number of times 0-7 > 0-10: {over_count_cosine}/{total_cosine}")
        print(f"p-value: {p_value_cosine:.3e}")
        print(f"mean cosine similarity is concentric: {is_concentric_distribution(-mean_similarity_cosine)}")


def collect_logistic_regression_data(distance_files, na, ft, distances, 
                                   similarities_dot, similarities_cosine):
    """Collect data for logistic regression analysis."""
    x1 = 1 if na == 'center' else 0
    x2 = 1 if ft == 'outward' else 0
    
    logistic_data_concentric = []
    logistic_data_distance = []
    logistic_data_dot_sim = []
    logistic_data_cosine_sim = []
    
    # Concentric distribution data
    for f in distance_files:
        d = np.load(f)
        is_concentric = is_concentric_distribution(d)
        
        logistic_data_concentric.append({
            'y': int(is_concentric),
            'x1': x1,
            'x2': x2,
            'x1_x2': x1 * x2,
            'nonzero_alpha': na,
            'flow_type': ft,
            'file': os.path.basename(f)
        })
    
    # Distance comparison data
    distances_0_7 = distances[:, 0, 7]
    distances_0_10 = distances[:, 0, 10]
    
    for i, (d_0_7, d_0_10) in enumerate(zip(distances_0_7, distances_0_10)):
        logistic_data_distance.append({
            'y': int(d_0_7 > d_0_10),
            'x1': x1,
            'x2': x2,
            'x1_x2': x1 * x2,
            'nonzero_alpha': na,
            'flow_type': ft,
            'distance_0_7': d_0_7,
            'distance_0_10': d_0_10,
            'snapshot': i
        })
    
    # Similarity data
    if similarities_dot is not None:
        similarities_dot_0_7 = similarities_dot[:, 0, 7]
        similarities_dot_0_10 = similarities_dot[:, 0, 10]
        
        for i, (s_0_7, s_0_10) in enumerate(zip(similarities_dot_0_7, similarities_dot_0_10)):
            logistic_data_dot_sim.append({
                'y': int(s_0_7 > s_0_10),
                'x1': x1,
                'x2': x2,
                'x1_x2': x1 * x2,
                'nonzero_alpha': na,
                'flow_type': ft,
                'similarity_0_7': s_0_7,
                'similarity_0_10': s_0_10,
                'snapshot': i
            })
    
    if similarities_cosine is not None:
        similarities_cosine_0_7 = similarities_cosine[:, 0, 7]
        similarities_cosine_0_10 = similarities_cosine[:, 0, 10]
        
        for i, (s_0_7, s_0_10) in enumerate(zip(similarities_cosine_0_7, similarities_cosine_0_10)):
            logistic_data_cosine_sim.append({
                'y': int(s_0_7 > s_0_10),
                'x1': x1,
                'x2': x2,
                'x1_x2': x1 * x2,
                'nonzero_alpha': na,
                'flow_type': ft,
                'similarity_0_7': s_0_7,
                'similarity_0_10': s_0_10,
                'snapshot': i
            })
    
    return (logistic_data_concentric, logistic_data_distance, 
            logistic_data_dot_sim, logistic_data_cosine_sim)


def perform_logistic_regression(logistic_data, analysis_name):
    """Perform logistic regression analysis."""
    if len(logistic_data) == 0:
        return
    
    print(f"\n{'='*60}")
    print(f"LOGISTIC REGRESSION ANALYSIS ({analysis_name})")
    print(f"{'='*60}")
    
    df_logistic = pd.DataFrame(logistic_data)
    print(f"Total observations: {len(df_logistic)}")
    print(f"Positive cases: {df_logistic['y'].sum()} ({df_logistic['y'].mean()*100:.1f}%)")
    
    condition_stats = df_logistic.groupby(['nonzero_alpha', 'flow_type']).agg(
        {'y': ['count', 'sum', 'mean']}
    ).round(3)
    condition_stats.columns = ['count', 'positive_count', 'positive_rate']
    print("\nRates by condition:")
    print(condition_stats)
    print()
    
    unique_conditions = df_logistic[['x1', 'x2']].drop_duplicates()
    if len(unique_conditions) > 1:
        X = df_logistic[['x1', 'x2', 'x1_x2']].values
        y = df_logistic['y'].values
        
        print("\n--- Logistic Regression Results (statsmodels) ---")
        X_sm = sm.add_constant(X, prepend=True)
        
        try:
            logit_model = sm.Logit(y, X_sm).fit(disp=0)
            print(logit_model.summary(
                xname=['Intercept', 'x1_center', 'x2_outward', 'x1_x2_interaction']
            ))
            
            # Model performance
            print("\n--- Model Performance ---")
            y_pred_proba = logit_model.predict(X_sm)
            y_pred = (y_pred_proba > 0.5).astype(int)
            accuracy = (y_pred == y).mean()
            print(f"Accuracy: {accuracy:.3f}")
            print("\nClassification Report:")
            print(classification_report(y, y_pred, zero_division=0))
            
            # Predicted probabilities for each condition
            print("\n--- Predicted probabilities for each condition ---")
            conditions = [
                [0, 0, 0],  # evenly, bidirectional
                [1, 0, 0],  # center, bidirectional
                [0, 1, 0],  # evenly, outward
                [1, 1, 1]   # center, outward
            ]
            condition_names = [
                'evenly + bidirectional',
                'center + bidirectional',
                'evenly + outward',
                'center + outward'
            ]
            conditions_sm = sm.add_constant(np.array(conditions), prepend=True)
            predicted_probs = logit_model.predict(conditions_sm)
            
            for name, prob in zip(condition_names, predicted_probs):
                print(f"{name}: {prob:.3f}")
                
        except Exception as e:
            print(f"Could not fit statsmodels Logit model. Error: {e}")
            print("This might be due to perfect separation in the data.")
    else:
        print("All observations have the same condition. Cannot perform logistic regression.")


def main():
    """Main function to orchestrate the analysis."""
    args, unknown = parse_arguments()
    
    # Get parameter combinations
    combinations = get_combinations(args.nonzero_alpha, args.flow_type)
    
    # Initialize logistic regression data containers
    all_logistic_data_concentric = []
    all_logistic_data_distance = []
    all_logistic_data_dot_sim = []
    all_logistic_data_cosine_sim = []
    
    for na, ft in combinations:
        print(f"\n{'='*50}")
        print(f"Processing: {na} + {ft}")
        print(f"{'='*50}")
        
        # Get directory paths
        load_dir, save_dir = get_directory_paths(
            ft, na, args.coupling_strength, args.agents_count, args.N_i
        )
        os.makedirs(save_dir, exist_ok=True)
        
        # Load distance files
        distance_files = sorted(glob.glob(os.path.join(load_dir, "distance_*.npy")))
        print(f"Number of distance files found: {len(distance_files)}")
        
        if not distance_files:
            print(f"No distance files found in {load_dir}. Skipping...")
            continue
            
        distance_files = distance_files[args.skip:]
        
        # Create age files if requested
        if args.make_age_files:
            state_files = sorted(glob.glob(os.path.join(load_dir, "state_*.npy")))
            create_age_files(load_dir, save_dir, state_files)
        
        # Load similarity data
        (mean_similarity_dot, mean_similarity_cosine, 
         similarities_dot, similarities_cosine) = load_similarity_data(
            load_dir, force_recompute=args.recompute_mean_similarity or args.check_concentric or args.logistic_regression
        )
        
        # Load and process distance data
        distances = []
        for f in distance_files:
            distances.append(np.load(f))
        distances = np.stack(distances, axis=0)
        
        mean_distance = distances.mean(axis=0)
        
        # Distance analysis
        if args.plot_distance:
            plot_distance_analysis(distances, save_dir, args.N_i)
            plot_mean_distance_analysis(mean_distance, save_dir, args.agent_id, args.N_i)
        
        # Age analysis
        if args.plot_age:
            plot_age_analysis(save_dir)
        
        # Similarity analysis
        similarities_dot_0_7 = None
        similarities_dot_0_10 = None
        similarities_cosine_0_7 = None
        similarities_cosine_0_10 = None
        
        if args.plot_similarity:
            if similarities_dot is not None:
                similarities_dot_0_7, similarities_dot_0_10 = plot_similarity_analysis(
                    similarities_dot, mean_similarity_dot, save_dir, 'dot'
                )
            
            if similarities_cosine is not None:
                similarities_cosine_0_7, similarities_cosine_0_10 = plot_similarity_analysis(
                    similarities_cosine, mean_similarity_cosine, save_dir, 'cosine'
                )
        
        # Concentric distribution check
        if args.check_concentric:
            distances_0_7 = distances[:, 0, 7]
            distances_0_10 = distances[:, 0, 10]
            
            perform_binomial_tests(
                distances_0_7, distances_0_10,
                similarities_dot_0_7, similarities_dot_0_10,
                similarities_cosine_0_7, similarities_cosine_0_10,
                mean_distance, mean_similarity_dot, mean_similarity_cosine
            )
        
        # Collect logistic regression data
        if args.logistic_regression:
            (logistic_data_concentric, logistic_data_distance,
             logistic_data_dot_sim, logistic_data_cosine_sim) = collect_logistic_regression_data(
                distance_files, na, ft, distances, similarities_dot, similarities_cosine
            )
            
            all_logistic_data_concentric.extend(logistic_data_concentric)
            all_logistic_data_distance.extend(logistic_data_distance)
            all_logistic_data_dot_sim.extend(logistic_data_dot_sim)
            all_logistic_data_cosine_sim.extend(logistic_data_cosine_sim)
    
    # Perform logistic regression analyses
    if args.logistic_regression:
        perform_logistic_regression(all_logistic_data_concentric, "Concentric Distribution")
        perform_logistic_regression(all_logistic_data_distance, "Distance Comparison (d(0,7) > d(0,10))")
        
        if all_logistic_data_dot_sim:
            perform_logistic_regression(all_logistic_data_dot_sim, "Dot Product Similarity Comparison")
        
        if all_logistic_data_cosine_sim:
            perform_logistic_regression(all_logistic_data_cosine_sim, "Cosine Similarity Comparison")


if __name__ == "__main__":
    main()