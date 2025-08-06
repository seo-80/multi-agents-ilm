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
import itertools
import pickle


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
    parser.add_argument('--pairwise_regression', action='store_true', 
                       help='If set, perform pairwise logistic regression and plot heatmaps')
    
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




def _compute_mean_iteratively(file_list, save_path):
    """
    ファイルを1つずつ読み込み、メモリ効率良く平均を計算して保存するヘルパー関数。
    """
    if not file_list:
        return None

    print(f"Found {len(file_list)} files. Averaging iteratively...")

    # 1. 最初のファイルを読み込み、合計用配列を初期化
    sum_array = np.load(file_list[0]).astype(np.float64) # 高精度な計算のためfloat64に変換
    
    # 2. 2つ目以降のファイルをループで処理し、合計に加算
    for f in file_list[1:]:
        sum_array += np.load(f)
        
    # 3. 合計をファイル数で割り、平均を計算
    mean_array = sum_array / len(file_list)
    
    # 4. 結果を保存
    np.save(save_path, mean_array)
    print(f"Saved mean to {save_path}")
    
    return mean_array


def load_similarity_data(load_dir, force_recompute=False):
    """Load dot product and cosine similarity data, with caching."""
    mean_dot_path = os.path.join(load_dir, "mean_similarity_dot.npy")
    mean_cosine_path = os.path.join(load_dir, "mean_similarity_cosine.npy")

    mean_similarity_dot = None
    mean_similarity_cosine = None

    # 既存ファイルがあればロード (この部分は変更なし)
    if not force_recompute and os.path.exists(mean_dot_path) and os.path.exists(mean_cosine_path):
        mean_similarity_dot = np.load(mean_dot_path)
        mean_similarity_cosine = np.load(mean_cosine_path)
        print("Loaded mean similarities from file.")
        # 元のコードではNoneを返していたため、それに合わせる
        return (mean_similarity_dot, mean_similarity_cosine, None, None)

    # なければ計算
    dot_sim_files = sorted(glob.glob(os.path.join(load_dir, "similarity_dot_*.npy")))
    cosine_sim_files = sorted(glob.glob(os.path.join(load_dir, "similarity_cosine_*.npy")))
    
    # 逐次処理で平均を計算するヘルパー関数を呼び出す
    if dot_sim_files:
        mean_similarity_dot = _compute_mean_iteratively(dot_sim_files, mean_dot_path)
        
    if cosine_sim_files:
        mean_similarity_cosine = _compute_mean_iteratively(cosine_sim_files, mean_cosine_path)

    # 元の関数の返り値の形式に合わせる
    return (mean_similarity_dot, mean_similarity_cosine, None, None)


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
    plt.close()


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
    plt.close()
    
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
    plt.close()
    
    # Bubble charts and heatmaps
    plot_bubble_charts(distances_0_7, distances_0_10, save_dir)
    plot_2d_heatmaps(distances_0_7, distances_0_10, save_dir)

    # --- Rank Matrix Heatmap ---
    # 距離行列の平均を計算
    mean_distance_matrix = distances.mean(axis=0)
    # 各行ごとに値が大きいほど順位が大きくなるように順位を計算（1始まり）
    rank_matrix = mean_distance_matrix.argsort(axis=1).argsort(axis=1) + 1
    for cmap in ['Blues', 'Reds', 'Greens', 'bwr', 'Blues_Reds']:
        filename = f'distance_rank_matrix_heatmap_{cmap}.png'
        plt.figure(figsize=(5, 5))
        if cmap == 'Blues_Reds':
            custom_cmap = colors.LinearSegmentedColormap.from_list('Blues_Reds', ['blue','red'])
            im = plt.imshow(rank_matrix, cmap=custom_cmap, aspect='equal')
        else:
            im = plt.imshow(rank_matrix, cmap=cmap, aspect='equal')
        plt.colorbar(im)
        plt.xticks([])
        plt.yticks([])
        plt.gca()
        # plt.title('Rank Matrix: iから見たjの距離順位（大きいほど順位大）')
        # plt.xlabel('Agent j')
        # plt.ylabel('Agent i')
        plt.savefig(os.path.join(save_dir, filename), dpi=300)
        plt.close()


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
    plt.close()
    
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
    plt.close()


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
    plt.close()
    
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
        plt.close()


def plot_mean_distance_analysis(mean_distance, save_dir, agent_id, N_i):
    """Plot mean distance heatmap in the same format as plot_2d_heatmaps."""
    # 正規化
    mean_distance = mean_distance / (2 * N_i)
    extent = [0, mean_distance.shape[0], 0, mean_distance.shape[1]]

    # Linear scale heatmap
    plt.figure(figsize=(5, 5))
    im = plt.imshow(mean_distance, extent=extent, 
                    cmap='Blues', aspect='equal', vmin=0, vmax=1)
    # plt.colorbar(im)
    plt.xticks([])
    plt.yticks([])
    # plt.xlabel('Agent i')
    # plt.ylabel('Agent j')
    plt.savefig(os.path.join(save_dir, "mean_distance_heatmap_Blues_no_colorbar.png"), dpi=300)
    plt.close()

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
        # plt.xlabel('Agent i')
        # plt.ylabel('Agent j')
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
    plt.close()
    
    # Agent 0 similarity plot
    plt.figure(figsize=(5, 5))
    plt.plot(np.arange(mean_similarity.shape[0]), mean_similarity[0], marker='o')
    plt.xticks([])
    plt.savefig(os.path.join(save_dir, f"mean_{similarity_type}_similarity_from_agent0.png"), dpi=300)
    plt.close()
    
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


def generate_concentric_data(distance_files, na, ft):
    """'Concentric'判定データを1つずつ生成するジェネレータ"""
    x1 = 1 if na == 'center' else 0
    x2 = 1 if ft == 'outward' else 0
    for f in distance_files:
        d = np.load(f)
        is_concentric = is_concentric_distribution(d)
        yield {
            'y': int(is_concentric), 'x1': x1, 'x2': x2, 'x1_x2': x1 * x2,
            'nonzero_alpha': na, 'flow_type': ft, 'file': os.path.basename(f)
        }

def generate_hub_comparison_data(data_array, na, ft, agents_count, hub_agent=7):
    """
    エージェント'hub_agent'を基準とした距離比較データを生成するジェネレータ。
    (i-hub)*(j-hub) < 0 の条件を満たすペア(i,j)のみを対象とする。
    """
    x1 = 1 if na == 'center' else 0
    x2 = 1 if ft == 'outward' else 0
    
    if data_array is None:
        return

    # ループの範囲をエージェント数に設定
    for i in range(agents_count):
        # HUB自身は基準エージェントになれない
        if i == hub_agent:
            continue
            
        for j in range(agents_count):
            # HUBやi自身は比較対象になれない
            if j == hub_agent or j == i:
                continue

            # エージェントiとjがHUBを挟んで反対側にいる場合のみ処理
            if (i - hub_agent) * (j - hub_agent) < 0:
                
                data_i_hub = data_array[:, i, hub_agent]
                data_i_j = data_array[:, i, j]
                
                for snapshot_idx, (val_i_hub, val_i_j) in enumerate(zip(data_i_hub, data_i_j)):
                    # y=1 となる条件は d(i, hub) > d(i, j)
                    is_positive = int(val_i_hub > val_i_j)
                    
                    yield {
                        'y': is_positive, 'x1': x1, 'x2': x2, 'x1_x2': x1 * x2,
                        'base_agent': i, 'compared_agent': j, 'hub_agent': hub_agent,
                        'nonzero_alpha': na, 'flow_type': ft, 'snapshot': snapshot_idx
                    }


def generate_hub_comparison_data_from_files(distance_files, na, ft, agents_count, hub_agent=7):
    """
    ファイルパスのリストを直接受け取り、ファイルを1つずつ読み込んで処理するジェネレータ。
    これにより、巨大な連結配列をメモリに保持する必要がなくなる。
    """
    x1 = 1 if na == 'center' else 0
    x2 = 1 if ft == 'outward' else 0
    
    # ループの主役を「ファイル」にする
    for snapshot_idx, file_path in enumerate(distance_files):
        # ファイルを1つだけメモリに読み込む
        snapshot_d = np.load(file_path)
        
        # 内部のループは以前と同じ
        for i in range(agents_count):
            if i == hub_agent:
                continue
            for j in range(agents_count):
                if j == hub_agent or j == i:
                    continue

                if (i - hub_agent) * (j - hub_agent) < 0:
                    val_i_hub = snapshot_d[i, hub_agent]
                    val_i_j = snapshot_d[i, j]
                    
                    is_positive = int(val_i_hub > val_i_j)
                    
                    yield {
                        'y': is_positive, 'x1': x1, 'x2': x2, 'x1_x2': x1 * x2,
                        'base_agent': i, 'compared_agent': j, 'hub_agent': hub_agent,
                        'nonzero_alpha': na, 'flow_type': ft, 'snapshot': snapshot_idx
                    }
        # このブロックの終わりで、`snapshot_d` は不要になりメモリから解放される


def generate_comparison_data(data_array, na, ft, agents_count, comparison_type):
    """
    距離や類似度の比較データを1つずつ生成する汎用ジェネレータ
    comparison_type: 'distance' または 'similarity'
    """
    x1 = 1 if na == 'center' else 0
    x2 = 1 if ft == 'outward' else 0
    
    # data_arrayがNoneの場合は何もせず終了
    if data_array is None:
        return

    for i in range(agents_count):
        other_agents = [agent for agent in range(agents_count) if agent != i]
        for j, k in itertools.combinations(other_agents, 2):
            data_i_j = data_array[:, i, j]
            data_i_k = data_array[:, i, k]
            
            for snapshot_idx, (val_ij, val_ik) in enumerate(zip(data_i_j, data_i_k)):
                # 距離の場合は d(i,j) > d(i,k)、類似度の場合は sim(i,j) < sim(i,k) で y=1
                is_positive = (val_ij > val_ik) if comparison_type == 'distance' else (val_ij < val_ik)
                
                yield {
                    'y': int(is_positive), 'x1': x1, 'x2': x2, 'x1_x2': x1 * x2,
                    'base_agent': i, 'agent_1': j, 'agent_2': k,
                    'nonzero_alpha': na, 'flow_type': ft, 'snapshot': snapshot_idx
                }
def perform_logistic_regression(data_generator, analysis_name):
    """
    ジェネレータから全データをDataFrameに読み込み、statsmodelsで詳細な回帰分析を行う。
    データ量が少ない場合に適している。
    """
    print(f"\n{'='*60}")
    print(f"LOGISTIC REGRESSION ANALYSIS ({analysis_name})")
    print(f"{'='*60}")
    
    # ジェネレータから全データを読み込む (データ量が少ないので問題ない)
    df_logistic = pd.DataFrame(list(data_generator))

    if df_logistic.empty:
        print("No data to analyze.")
        return

    print(f"Total observations: {len(df_logistic)}")
    print(f"Positive cases (y=1): {df_logistic['y'].sum()} ({df_logistic['y'].mean()*100:.1f}%)")
    
    # 条件別の統計情報を表示
    if 'nonzero_alpha' in df_logistic.columns:
        condition_stats = df_logistic.groupby(['nonzero_alpha', 'flow_type']).agg(
            {'y': ['count', 'sum', 'mean']}
        ).round(3)
        condition_stats.columns = ['count', 'positive_count', 'positive_rate']
        print("\nRates by condition:")
        print(condition_stats)
    print()

    # ロジスティック回帰の実行
    X = df_logistic[['x1', 'x2', 'x1_x2']].values
    y = df_logistic['y'].values
    
    X_sm = sm.add_constant(X, prepend=True)
    try:
        logit_model = sm.Logit(y, X_sm).fit(disp=0)
        print("\n--- Logistic Regression Results (statsmodels) ---")
        print(logit_model.summary(
            xname=['Intercept', 'x1_center', 'x2_outward', 'x1_x2_interaction']
        ))
    except Exception as e:
        print(f"Could not fit statsmodels Logit model. Error: {e}")


def run_single_pair_regression(df_pair):
    """特定の(i, j)ペアのデータフレームを受け取り、回帰分析を実行して係数を返す"""
    # データが少なすぎる、またはyの値が全て0か1の場合は分析できない
    if len(df_pair) < 4 or df_pair['y'].nunique() < 2:
        return [np.nan] * 4 # 4つの係数すべてをNaNで返す

    try:
        X = df_pair[['x1', 'x2', 'x1_x2']]
        y = df_pair['y']
        X_sm = sm.add_constant(X, prepend=True)
        model = sm.Logit(y, X_sm).fit(disp=0)
        return model.params.values
    except Exception as e:
        return [np.nan] * 4


def analyze_and_plot_by_pair(args):
    """
    各(i, j)ペアに対してロジスティック回帰を実行し、結果をヒートマップとして保存する。
    結果は保存され、既に結果があればそれをロードする。
    """

    # 結果ファイルのパスを設定
    results_file = os.path.join(args.save_dir, 'pair_analysis_results.pkl')
    
    # 既に結果ファイルが存在する場合はロード
    if os.path.exists(results_file):
        print(f"Loading existing results from {results_file}")
        with open(results_file, 'rb') as f:
            df_results = pickle.load(f)
        print("Results loaded successfully.")
    else:
        print("Starting analysis for each (i, j) pair. This may take a very long time...")
        combinations = get_combinations(args.nonzero_alpha, args.flow_type)
        hub_agent = 7
        agents_count = args.agents_count
        results = []
        valid_pairs = []
        for i in range(agents_count):
            for j in range(agents_count):
                if i == hub_agent or j == hub_agent or i == j:
                    continue
                if (i - hub_agent) * (j - hub_agent) < 0:
                    valid_pairs.append((i, j))
        for idx, (i, j) in enumerate(valid_pairs):
            if (idx + 1) % 100 == 0:
                print(f"Processing pair {idx + 1}/{len(valid_pairs)}: ({i}, {j})")
            pair_data = []
            for na, ft in combinations:
                x1 = 1 if na == 'center' else 0
                x2 = 1 if ft == 'outward' else 0
                load_dir, _ = get_directory_paths(ft, na, args.coupling_strength, args.agents_count, args.N_i)
                distance_files = sorted(glob.glob(os.path.join(load_dir, "distance_*.npy")))
                for file_path in distance_files[args.skip:]:
                    snapshot_d = np.load(file_path)
                    val_i_hub = snapshot_d[i, hub_agent]
                    val_i_j = snapshot_d[i, j]
                    y_val = int(val_i_hub > val_i_j)
                    pair_data.append({'y': y_val, 'x1': x1, 'x2': x2, 'x1_x2': x1 * x2})
            df_pair = pd.DataFrame(pair_data)
            coeffs = run_single_pair_regression(df_pair)
            results.append({
                'i': i, 'j': j,
                'Intercept': coeffs[0],
                'x1_center': coeffs[1],
                'x2_outward': coeffs[2],
                'x1_x2_interaction': coeffs[3]
            })
        
        # 結果をDataFrameに変換
        df_results = pd.DataFrame(results)
        
        # 結果を保存
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"Saving results to {results_file}")
        with open(results_file, 'wb') as f:
            pickle.dump(df_results, f)
        print("Results saved successfully.")
    
    print("Creating heatmaps...")
    coeff_names = ['Intercept', 'x1_center', 'x2_outward', 'x1_x2_interaction']
    os.makedirs(args.save_dir, exist_ok=True)
    for coeff_name in coeff_names:
        # データの準備
        heatmap_data = df_results.pivot(index='i', columns='j', values=coeff_name)

        # 指定されたフォーマットでプロット
        plt.figure(figsize=(5, 5))
        # データの範囲を対称にして、白が0になるようにする
        print(heatmap_data)
        print(heatmap_data.values)
        print(np.nanmax(np.abs(heatmap_data.values)))
        max_abs_val = np.nanmax(np.abs(heatmap_data.values))
        im = plt.imshow(heatmap_data, aspect="equal", cmap='coolwarm', 
                       vmin=-max_abs_val, vmax=max_abs_val)
        plt.colorbar(im, label=f'"{coeff_name}" Coefficient')
        plt.xticks([])
        plt.yticks([])

        # 画像の保存
        save_path = os.path.join(args.save_dir, f'heatmap_{coeff_name}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved heatmap to {save_path}")


def main():
    """Main function to orchestrate the analysis."""
    args, unknown = parse_arguments()
    
    combinations = get_combinations(args.nonzero_alpha, args.flow_type)
    
    # <<<<<<<<<<<<<<<<< 変更点: 新しいロジックのジェネレータを格納するリスト >>>>>>>>>>>>>>>>>
    concentric_generators = []
    hub_comparison_generators = []
    
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
         _, _) = load_similarity_data( # similarities_dot/cosine は平均計算後は不要なため受け取らない
            load_dir, force_recompute=args.recompute_mean_similarity
        )
        
        # 注意: この部分もファイル数が多いとメモリを圧迫する可能性があります。
        # もしここでメモリ不足になる場合は、この部分も逐次処理に書き換える必要があります。
        # distances = np.stack([np.load(f) for f in distance_files], axis=0)  # この行を削除
        
        # mean_distanceの計算も逐次処理に変更
        mean_distance = None
        if args.plot_distance:
            # 平均距離を逐次計算
            total_distance = None
            count = 0
            for f in distance_files:
                snapshot_d = np.load(f)
                if total_distance is None:
                    total_distance = snapshot_d.copy()
                else:
                    total_distance += snapshot_d
                count += 1
            mean_distance = total_distance / count
        
        # Distance analysis
        if args.plot_distance:
            # plot_distance_analysis関数は完全なdistances配列を必要とするため、
            # この部分は現在の変更では動作しません。必要に応じて修正が必要です。
            print("Warning: plot_distance_analysis requires full distances array which is not loaded in this version.")
            # plot_distance_analysis(distances, save_dir, args.N_i)
            plot_mean_distance_analysis(mean_distance, save_dir, args.agent_id, args.N_i)
        
        # Age analysis
        if args.plot_age:
            plot_age_analysis(save_dir)
        
        # Similarity analysis
        if args.plot_similarity:
            # 注意: plot_similarity_analysis関数は完全なsimilarities配列を必要とするため、
            # この部分は現在の変更では動作しません。必要に応じて修正が必要です。
            print("Warning: plot_similarity analysis requires full similarity arrays which are not loaded in this version.")
            # similarities_dot_0_7 = None
            # similarities_dot_0_10 = None
            # similarities_cosine_0_7 = None
            # similarities_cosine_0_10 = None
            # if similarities_dot is not None:
            #     similarities_dot_0_7, similarities_dot_0_10 = plot_similarity_analysis(
            #         similarities_dot, mean_similarity_dot, save_dir, 'dot'
            #     )
            
            # if similarities_cosine is not None:
            #     similarities_cosine_0_7, similarities_cosine_0_10 = plot_similarity_analysis(
            #         similarities_cosine, mean_similarity_cosine, save_dir, 'cosine'
            #     )
        
        # Concentric distribution check
        if args.check_concentric:
            # distances配列が利用できないため、この部分は現在の変更では動作しません。
            print("Warning: check_concentric analysis requires full distances array which is not loaded in this version.")
            # distances_0_7 = distances[:, 0, 7]
            # distances_0_10 = distances[:, 0, 10]
            
            # 注意: perform_binomial_tests関数は完全なsimilarities配列を必要とするため、
            # この部分は現在の変更では動作しません。必要に応じて修正が必要です。
            print("Warning: perform_binomial_tests requires full similarity arrays which are not loaded in this version.")
            # perform_binomial_tests(
            #     distances_0_7, distances_0_10,
            #     similarities_dot_0_7, similarities_dot_0_10,
            #     similarities_cosine_0_7, similarities_cosine_0_10,
            #     mean_distance, mean_similarity_dot, mean_similarity_cosine
            # )
        
        # <<<<<<<<<<<<<<<<< 変更点: 新しいロジックのジェネレータを呼び出す >>>>>>>>>>>>>>>>>
        if args.logistic_regression:
            # Concentric判定のジェネレータ
            concentric_gen = generate_concentric_data(distance_files, na, ft)
            concentric_generators.append(concentric_gen)
            
            # ジェネレータには、配列ではなくファイルパスのリストを渡す
            dist_gen = generate_hub_comparison_data_from_files(
                distance_files, na, ft, args.agents_count, hub_agent=7
            )
            hub_comparison_generators.append(dist_gen)
    
    # <<<<<<<<<<<<<<<<< 変更点: 連結したジェネレータをstatsmodelsで分析 >>>>>>>>>>>>>>>>>
    if args.logistic_regression and hub_comparison_generators:
        print("\n" + "="*60)
        print("Performing final logistic regression on all combined data...")
        print("="*60)

        # 1. Concentric Distribution の分析
        final_concentric_gen = itertools.chain.from_iterable(concentric_generators)
        perform_logistic_regression(final_concentric_gen, "Concentric Distribution")
        
        # 2. 距離比較の分析（新しいロジック）
        final_dist_gen = itertools.chain.from_iterable(hub_comparison_generators)
        perform_logistic_regression(final_dist_gen, "Hub Comparison: d(i, 7) > d(i, j)")

    if hasattr(args, 'pairwise_regression') and args.pairwise_regression:
        if not hasattr(args, 'save_dir') or args.save_dir is None:
            # デフォルトの保存先を設定
            args.save_dir = 'data/naive_simulation/fig/pairwise_regression_heatmaps'
        os.makedirs(args.save_dir, exist_ok=True)
        analyze_and_plot_by_pair(args)


if __name__ == "__main__":
    main()