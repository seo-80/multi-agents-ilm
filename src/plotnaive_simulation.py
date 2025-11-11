"""
論文のモデルパラメータとコードの対応：
- agents_count ≡ M: エージェント数
- coupling_strength ≡ m: 相互作用の重み（interaction weight）
- center_agent ≡ c: 中心エージェントの位置（c = (M+1)/2）
- N_i: データプールサイズ
- alpha_per_data ≡ α/N_i: 1データあたりの新語生成率
- α: 新語生成パラメータ（innovation parameter）
- μ = α/(N_i + α): 新語生成確率（mutation probability）
"""

import numpy as np
import glob
import os
import matplotlib
matplotlib.use('Agg')  # SSH経由での実行のため非インタラクティブモードに設定
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
from config import get_data_raw_dir, get_data_fig_dir, get_data_distance_dir


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Language evolution analysis script')

    # Basic parameters (論文表記: M, m, c, N_i, α)
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
                       help='Number of agents M (default: 15)')
    parser.add_argument('--N_i', '-n', type=int, default=100,
                       help='Data pool size N_i (default: 100)')
    parser.add_argument('--coupling_strength', '-c', type=float, default=0.01,
                       help='Interaction weight m (default: 0.01)')
    parser.add_argument('--alpha_per_data', type=float, default=0.001, help='Innovation rate α/N_i')
    parser.add_argument('--force_recompute', action='store_true',
                       help='If set, force recompute cached data (similarity, F-matrix, etc.)')
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
    parser.add_argument('--skip_kde', action='store_true',
                       help='If set, skip KDE overlay plots (useful when KDE calculation is very slow)')

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
    load_dir = os.path.join(get_data_raw_dir(), subdir)
    save_dir = os.path.join(get_data_fig_dir(), subdir)
    distance_dir = os.path.join(get_data_distance_dir(), subdir)

    return load_dir, save_dir, distance_dir


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


def is_concentric_distribution_similarity(similarity_matrix):
    """
    Check if the similarity matrix shows a concentric distribution.
    For similarity (higher is more similar), concentric means the opposite side
    has higher similarity than the center.
    """
    center = len(similarity_matrix) // 2

    for base in range(len(similarity_matrix)):
        if base == center:
            continue

        for reference in range(len(similarity_matrix)):
            is_opposite_side = (base - center) * (reference - center) < 0
            # 類似度は高いほど近いので、反対側が中心より高い場合にconcentric
            if is_opposite_side and similarity_matrix[base][reference] > similarity_matrix[base][center]:
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

    # 既存ファイルがあればロード
    if not force_recompute and os.path.exists(mean_dot_path) and os.path.exists(mean_cosine_path):
        mean_similarity_dot = np.load(mean_dot_path)
        mean_similarity_cosine = np.load(mean_cosine_path)
        print("Loaded mean similarities from file.")
        return (mean_similarity_dot, mean_similarity_cosine, None, None)

    # なければ計算
    dot_sim_files = sorted(glob.glob(os.path.join(load_dir, "similarity_dot_*.npy")))
    cosine_sim_files = sorted(glob.glob(os.path.join(load_dir, "similarity_cosine_*.npy")))
    
    # 元の一括読み込み処理に戻す
    if dot_sim_files:
        print(f"Computing mean dot similarity from {len(dot_sim_files)} files...")
        dot_similarities = np.array([np.load(f) for f in dot_sim_files])
        mean_similarity_dot = np.mean(dot_similarities, axis=0)
        np.save(mean_dot_path, mean_similarity_dot)
        print(f"Saved mean dot similarity to {mean_dot_path}")
    else:
        mean_similarity_dot = None
        dot_similarities = None
        
    if cosine_sim_files:
        print(f"Computing mean cosine similarity from {len(cosine_sim_files)} files...")
        cosine_similarities = np.array([np.load(f) for f in cosine_sim_files])
        mean_similarity_cosine = np.mean(cosine_similarities, axis=0)
        np.save(mean_cosine_path, mean_similarity_cosine)
        print(f"Saved mean cosine similarity to {mean_cosine_path}")
    else:
        mean_similarity_cosine = None
        cosine_similarities = None

    # 個別データも返すように変更
    return (mean_similarity_dot, mean_similarity_cosine, dot_similarities, cosine_similarities)


def load_f_matrix_data(load_dir, force_recompute=False):
    """
    Load F-matrix, 1-F, and Nei's distance data, with caching.

    F-matrix (f_matrix_*.npy) represents the IBD (Identity By Descent) probability.
    If f_matrix files don't exist, computes from similarity_dot files (they are identical).
    1-F is computed as: (1-F)_ij = 1 - F_ij
    Nei's distance is computed as: D_ij = -ln(F_ij / sqrt(F_ii * F_jj))

    Args:
        load_dir: Directory containing the data files
        force_recompute: If True, recompute even if cached files exist

    Returns:
        tuple: (mean_f_matrix, mean_one_minus_f, mean_nei_distance, f_matrices, one_minus_f_distances, nei_distances)
               - mean_f_matrix: Time-averaged F-matrix
               - mean_one_minus_f: Time-averaged 1-F
               - mean_nei_distance: Time-averaged Nei's distance
               - f_matrices: All F-matrices (or None if cached)
               - one_minus_f_distances: All 1-F values (or None if cached)
               - nei_distances: All Nei's distances (or None if cached)
    """
    mean_f_path = os.path.join(load_dir, "mean_f_matrix.npy")
    mean_one_minus_f_path = os.path.join(load_dir, "mean_one_minus_f.npy")
    mean_nei_path = os.path.join(load_dir, "mean_nei_distance.npy")

    # Check cache
    if not force_recompute and os.path.exists(mean_f_path) and os.path.exists(mean_one_minus_f_path) and os.path.exists(mean_nei_path):
        mean_f_matrix = np.load(mean_f_path)
        mean_one_minus_f = np.load(mean_one_minus_f_path)
        mean_nei_distance = np.load(mean_nei_path)
        print("Loaded mean F-matrix, 1-F, and Nei's distance from cache.")
        return (mean_f_matrix, mean_one_minus_f, mean_nei_distance, None, None, None)

    # Look for F-matrix files first
    f_files = sorted(glob.glob(os.path.join(load_dir, "f_matrix_*.npy")),
                     key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[2]))

    if not f_files:
        # Fall back to similarity_dot files (they are identical to F-matrix)
        print("F-matrix files not found. Computing from similarity_dot files...")
        f_files = sorted(glob.glob(os.path.join(load_dir, "similarity_dot_*.npy")),
                        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[2]))
        if not f_files:
            raise FileNotFoundError(f"Neither f_matrix_*.npy nor similarity_dot_*.npy found in {load_dir}")

    # Load and compute mean
    print(f"Computing mean F-matrix from {len(f_files)} files...")
    f_matrices = np.array([np.load(f) for f in f_files])
    mean_f_matrix = np.mean(f_matrices, axis=0)

    # Compute 1-F
    one_minus_f_distances = 1.0 - f_matrices
    mean_one_minus_f = 1.0 - mean_f_matrix

    # Compute Nei's distance: D_ij = -ln(F_ij / sqrt(F_ii * F_jj))
    # First compute for all time steps
    nei_distances = np.zeros_like(f_matrices)
    for t in range(len(f_matrices)):
        f_diag = np.diag(f_matrices[t]).copy()
        # Avoid division by zero and log of zero/negative
        denominator = np.sqrt(np.outer(f_diag, f_diag))
        ratio = np.divide(f_matrices[t], denominator, where=denominator > 0, out=np.ones_like(f_matrices[t]))
        ratio = np.clip(ratio, 1e-10, None)  # Avoid log(0)
        nei_distances[t] = -np.log(ratio)

    # Compute mean Nei's distance
    mean_f_diag = np.diag(mean_f_matrix).copy()
    mean_denominator = np.sqrt(np.outer(mean_f_diag, mean_f_diag))
    mean_ratio = np.divide(mean_f_matrix, mean_denominator, where=mean_denominator > 0, out=np.ones_like(mean_f_matrix))
    mean_ratio = np.clip(mean_ratio, 1e-10, None)
    mean_nei_distance = -np.log(mean_ratio)

    # Save cache
    np.save(mean_f_path, mean_f_matrix)
    np.save(mean_one_minus_f_path, mean_one_minus_f)
    np.save(mean_nei_path, mean_nei_distance)
    print(f"Saved mean F-matrix to {mean_f_path}")
    print(f"Saved mean 1-F to {mean_one_minus_f_path}")
    print(f"Saved mean Nei's distance to {mean_nei_path}")

    return (mean_f_matrix, mean_one_minus_f, mean_nei_distance, f_matrices, one_minus_f_distances, nei_distances)


def plot_histogram_comparison(data1, data2, labels, colors, save_path, title_suffix="",
                            log_scale=False, shift_amount=None, orientation='vertical', overlay_kde=False, kde_bandwidth=None, unit_max_for_beta=None, n_bins=None):
    """Plot histogram comparison between two datasets."""
    plt.figure(figsize=(5, 5))
    # bin幅を計算
    all_data = np.concatenate([data1, data2])

    if n_bins is None:
        # デフォルトの動作：shift方式
        bin_edges = np.linspace(np.min(all_data), np.max(all_data), 202)
        bin_width = bin_edges[1] - bin_edges[0]
        if shift_amount is None:
            shift_amount = bin_width / 2

        for i, (data, label, color) in enumerate(zip([data1, data2], labels, colors)):
            mean_val = np.mean(data)
            shift = -shift_amount if i == 0 else shift_amount
            hist_alpha = 0.3 if overlay_kde else 1.0
            plt.hist(
                data + shift,
                bins=bin_edges + shift,
                label=label,
                color=color,
                density=True,
                orientation='horizontal' if orientation == 'horizontal' else 'vertical',
                alpha=hist_alpha
            )
            if orientation == 'horizontal':
                plt.axhline(mean_val + shift, color=color, linestyle='--',
                            label=f'Mean {label}: {mean_val:.3f} (shifted)')
            else:
                plt.axvline(mean_val + shift, color=color, linestyle='--',
                            label=f'Mean {label}: {mean_val:.3f} (shifted)')
    else:
        # n_binsが指定されている場合：並べて表示
        bin_edges = np.linspace(np.min(all_data), np.max(all_data), n_bins + 1)
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bar_width = bin_width * 0.4  # 各バーの幅をビン幅の40%に

        for i, (data, label, color) in enumerate(zip([data1, data2], labels, colors)):
            mean_val = np.mean(data)
            hist_alpha = 0.3 if overlay_kde else 1.0

            # ヒストグラムを計算
            counts, _ = np.histogram(data, bins=bin_edges, density=True)

            # バーの位置を計算（左右にずらす）
            offset = -bar_width / 2 if i == 0 else bar_width / 2
            positions = bin_centers + offset

            if orientation == 'horizontal':
                plt.barh(positions, counts, height=bar_width, label=label,
                        color=color, alpha=hist_alpha)
                plt.axhline(mean_val, color=color, linestyle='--',
                           label=f'Mean {label}: {mean_val:.3f}')
            else:
                plt.bar(positions, counts, width=bar_width, label=label,
                       color=color, alpha=hist_alpha)
                plt.axvline(mean_val, color=color, linestyle='--',
                           label=f'Mean {label}: {mean_val:.3f}')
    
    if log_scale:
        if orientation == 'horizontal':
            plt.xscale('log')
        else:
            plt.yscale('log')
        plt.grid(True, alpha=0.3, which='both')
    else:
        plt.grid(True, alpha=0.3)
    
    # KDE overlay using statsmodels beta kernel with boundary correction
    if overlay_kde:
        from statsmodels.nonparametric.kernels_asymmetric import pdf_kernel_asym
        # 可変上限 (例: 距離やcosineは1、dotはN_i)
        xmax = unit_max_for_beta if unit_max_for_beta is not None else np.max(all_data)
        xmin = 0.0
        # 正規化して[0,1]にスケーリング
        scale = xmax - xmin
        if scale <= 0:
            scale = 1.0

        # 境界処理用のパラメータ
        eps = 1e-10
        grid_unit = np.linspace(eps, 1.0 - eps, 1000)
        # 自動的にbandwidthを調整: データの標準偏差ベース (Scott's rule修正版)
        combined_data_unit = (all_data - xmin) / scale
        combined_data_unit = np.clip(combined_data_unit, eps, 1.0 - eps)
        std_unit = np.std(combined_data_unit)
        # Scott's rule: bw = n^(-1/5) * std, but multiply by larger factor for beta kernel
        auto_bw = 1.5 * (len(combined_data_unit) ** (-0.2)) * std_unit
        bw = kde_bandwidth if kde_bandwidth is not None else max(0.02, min(0.2, auto_bw))
        print(f"Using KDE bandwidth: {bw:.4f} (auto_bw={auto_bw:.4f}, std={std_unit:.4f})")

        # 各データセットに対して個別にKDEを計算
        for i, (data, color, label) in enumerate(zip([data1, data2], colors, labels)):
            data_unit = (data - xmin) / scale
            data_unit = np.clip(data_unit, eps, 1.0 - eps)

            # データサイズが大きい場合はサンプリング
            if len(data_unit) > 10000:
                sample_indices = np.random.choice(len(data_unit), 10000, replace=False)
                data_unit = data_unit[sample_indices]

            # pdf_kernel_asymでKDE計算
            y_unit = pdf_kernel_asym(grid_unit, data_unit, bw, kernel_type='beta', batch_size=100)

            # 単位区間から元のスケールへ変換（密度は1/scaleでスケーリング）
            x_grid = grid_unit * scale + xmin
            y_density = y_unit / scale

            if orientation == 'horizontal':
                plt.plot(y_density, x_grid, color=color, linewidth=2.0, linestyle='-', alpha=1.0, label=f'KDE {label}')
            else:
                plt.plot(x_grid, y_density, color=color, linewidth=2.0, linestyle='-', alpha=1.0, label=f'KDE {label}')
    
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_distance_data(distances, distance_dir, center_agent, opposite_agent):
    """Save distance data as npy files."""
    os.makedirs(distance_dir, exist_ok=True)

    # Save full distance matrix [timesteps, agents, agents]
    np.save(os.path.join(distance_dir, 'distance_full.npy'), distances)
    print(f"Saved distance_full.npy to {distance_dir}")

    # Save distances from agent 0 to center and opposite agents [timesteps, 2]
    distances_0_center_opposite = np.stack([
        distances[:, 0, center_agent],
        distances[:, 0, opposite_agent]
    ], axis=1)
    np.save(os.path.join(distance_dir, 'distance_0_center_opposite.npy'), distances_0_center_opposite)
    print(f"Saved distance_0_center_opposite.npy to {distance_dir}")

    # Also save as xlsx
    df_distances = pd.DataFrame(
        distances_0_center_opposite,
        columns=[f'distance_0_{center_agent}', f'distance_0_{opposite_agent}']
    )
    xlsx_path = os.path.join(distance_dir, 'distance_0_center_opposite.xlsx')
    df_distances.to_excel(xlsx_path, index=False)
    print(f"Saved distance_0_center_opposite.xlsx to {distance_dir}")


def plot_distance_analysis(distances, save_dir, N_i, center_agent, opposite_agent, args=None):
    """Plot comprehensive distance analysis."""
    # 正規化
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
    # Histogram comparison with fixed 25 bins
    plot_histogram_comparison(
        distances_0_center, distances_0_opposite,
        [f'Agent 0-{center_agent}', f'Agent 0-{opposite_agent}'],
        ['blue', 'red'],
        os.path.join(save_dir, 'agent_pair_distances_histogram_bins25.png'),
        n_bins=25
    )
    # Histogram comparison with fixed 25 bins (horizontal)
    plot_histogram_comparison(
        distances_0_center, distances_0_opposite,
        [f'Agent 0-{center_agent}', f'Agent 0-{opposite_agent}'],
        ['blue', 'red'],
        os.path.join(save_dir, 'agent_pair_distances_histogram_bins25_horizontal.png'),
        n_bins=25, orientation='horizontal'
    )
    # Horizontal (axis-swapped) histogram
    plot_histogram_comparison(
        distances_0_center, distances_0_opposite,
        [f'Agent 0-{center_agent}', f'Agent 0-{opposite_agent}'],
        ['blue', 'red'],
        os.path.join(save_dir, 'agent_pair_distances_histogram_horizontal.png'),
        orientation='horizontal'
    )
    # KDE overlay (Beta kernel over [0,1])
    # Note: Can be very slow with large datasets
    # Use --skip_kde flag to skip this computation
    if args is None or not args.skip_kde:
        plot_histogram_comparison(
            distances_0_center, distances_0_opposite,
            [f'Agent 0-{center_agent}', f'Agent 0-{opposite_agent}'],
            ['blue', 'red'],
            os.path.join(save_dir, 'agent_pair_distances_histogram_kde.png'),
            overlay_kde=True, unit_max_for_beta=1.0
        )
    
    # Log scale histogram
    plot_histogram_comparison(
        distances_0_center, distances_0_opposite, 
        [f'Agent 0-{center_agent}', f'Agent 0-{opposite_agent}'], 
        ['blue', 'red'],
        os.path.join(save_dir, 'agent_pair_distances_histogram_log.png'),
        log_scale=True
    )
    
    # Distance difference histogram
    diff_distances = distances_0_center - distances_0_opposite
    plt.figure(figsize=(5, 5))
    bin_edges = np.linspace(np.min(diff_distances), np.max(diff_distances), 402)
    mean_diff = np.mean(diff_distances)
    
    plt.hist(diff_distances, bins=bin_edges, alpha=0.75, 
            label=f'd(0, {center_agent}) - d(0, {opposite_agent})', color='green', density=True)
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
    plt.scatter(distances_0_center, distances_0_opposite, alpha=0.2, s=15, edgecolors='none')
    max_val = max(np.max(distances_0_center), np.max(distances_0_opposite))
    min_val = min(np.min(distances_0_center), np.min(distances_0_opposite))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label=f'd(0,{center_agent}) = d(0,{opposite_agent})')
    # plt.xlabel('Distance d(0, 7)')
    # plt.ylabel('Distance d(0, 10)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.savefig(os.path.join(save_dir, 'distance_scatter_plot.png'), dpi=300)
    plt.close()
    
    # Bubble charts and heatmaps
    plot_bubble_charts(distances_0_center, distances_0_opposite, save_dir)
    plot_2d_heatmaps(distances_0_center, distances_0_opposite, save_dir)

    # --- Rank Matrix Heatmap ---
    # 距離行列の平均を計算
    mean_distance_matrix = distances.mean(axis=0)

    # Normalize distance matrix (Manhattan distance with N_i=1 gives 0 or 2, normalize to 0 or 1)
    mean_distance_matrix_normalized = mean_distance_matrix / norm_factor

    # Save mean distance matrix as CSV
    df_mean_distance = pd.DataFrame(mean_distance_matrix_normalized)
    csv_path = os.path.join(save_dir, 'mean_distance_matrix.csv')
    df_mean_distance.to_csv(csv_path, index=True)
    print(f"Saved mean_distance_matrix.csv to {save_dir}")

    # 各行ごとに値が大きいほど順位が大きくなるように順位を計算（1始まり）


def plot_bubble_charts(distances_x, distances_y, save_dir):
    """Plot bubble charts for distance data."""
    df_scatter = pd.DataFrame({
        'x': distances_x,
        'y': distances_y
    })
    bubble_data = df_scatter.groupby(['x', 'y']).size().reset_index(name='count')
    
    # Linear bubble chart
    plt.figure(figsize=(7, 6))
    scale_factor = 5
    bubble_sizes = bubble_data['count'] * scale_factor
    plt.scatter(bubble_data['x'], bubble_data['y'], s=bubble_sizes, 
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
    plt.scatter(bubble_data['x'], bubble_data['y'], s=bubble_sizes_log, 
               alpha=0.8, edgecolors="w", linewidth=0.5)
    # plt.xlabel('Distance d(0, 7)')
    # plt.ylabel('Distance d(0, 10)')
    plt.grid(True, alpha=0.3, zorder=-1)
    plt.axis('equal')
    plt.savefig(os.path.join(save_dir, 'distance_log_bubble_exact_counts_plot.png'), dpi=300)
    plt.close()


def plot_2d_heatmaps(distances_x, distances_y, save_dir):
    """Plot 2D heatmaps for distance data."""
    bins = 50
    counts, xedges, yedges = np.histogram2d(distances_x, distances_y, bins=bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    # Linear scale heatmap
    plt.figure(figsize=(5, 5))
    im = plt.imshow(counts.T, extent=extent, origin='lower', 
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
        im_log = plt.imshow(counts_masked.T, extent=extent, origin='lower', 
                           cmap=cmap, aspect='equal', norm=colors.LogNorm())
        plt.colorbar(im_log)
        # plt.xlabel('Distance d(0, 7)')
        # plt.ylabel('Distance d(0, 10)')
        plt.savefig(os.path.join(save_dir, f'distance_heatmap_log_{cmap}.png'), dpi=300)
        plt.close()


def plot_mean_distance_analysis(mean_distance, save_dir, agent_id, N_i, center_agent, plot_discrete_colorbar=False, mean_distance_euclidean_sq=None):
    """Plot mean distance heatmap in the same format as plot_2d_heatmaps.

    Args:
        mean_distance: E[D_Manhattan] - Mean Manhattan distance
        mean_distance_euclidean_sq: E[D_Euclidean^2] - Mean of Euclidean distance squared (NOT E[D_Manhattan]^2)
    """
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

    # Agent-specific distance plot with logarithmic fitting
    # フィッティング: 言語距離 = ln(a·地理距離 + 1)
    from scipy.optimize import curve_fit

    geographic_dist = np.arange(mean_distance.shape[0])
    linguistic_dist = mean_distance[agent_id]

    # フィッティング関数の定義: y = ln(a*x + 1)
    def log_model(x, a):
        return np.log(a * x + 1)

    # 初期値を設定してフィッティング実行
    try:
        # aの初期値は適当に設定（0.1程度）
        popt, pcov = curve_fit(log_model, geographic_dist, linguistic_dist, p0=[0.1], bounds=(0, np.inf))
        a = popt[0]

        # フィッティング曲線の計算
        Y_fit = log_model(geographic_dist, a)

        # 決定係数 R² を計算
        ss_res = np.sum((linguistic_dist - Y_fit) ** 2)
        ss_tot = np.sum((linguistic_dist - np.mean(linguistic_dist)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # プロット
        plt.figure(figsize=(5, 5))
        plt.plot(geographic_dist, linguistic_dist, marker='o', label='Data')
        plt.plot(geographic_dist, Y_fit, 'r-', linewidth=2, label=f'Fit: y=ln({a:.3f}x+1)\nR²={r_squared:.3f}')
        plt.legend()
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(save_dir, f"mean_distance_from_agent{agent_id}_with_fit.png"), dpi=300)
        plt.close()

        # フィッティングパラメータを表示
        print(f"Logarithmic fitting for agent {agent_id}:")
        print(f"  a = {a:.6f}")
        print(f"  R² = {r_squared:.6f}")
    except Exception as e:
        print(f"Fitting failed for agent {agent_id}: {e}")
        # フィッティングが失敗した場合はデータのみプロット
        plt.figure(figsize=(5, 5))
        plt.plot(geographic_dist, linguistic_dist, marker='o', label='Data')
        plt.legend()
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(save_dir, f"mean_distance_from_agent{agent_id}_with_fit.png"), dpi=300)
        plt.close()

    # グラフの線を描画
    plt.figure(figsize=(5, 5))
    plt.plot(np.arange(mean_distance.shape[0]), mean_distance[agent_id])

    # 0番目のデータ点を 'x' マーカーでプロット
    plt.plot(0, mean_distance[agent_id][0], marker='x', color='C0', markersize=8, markeredgewidth=1.5)
    # 1番目以降のデータ点を 'o' マーカーでプロット
    plt.plot(np.arange(mean_distance.shape[0])[1:], mean_distance[agent_id][1:], marker='o', color='C0', linestyle='None')

    # center_agent のデータ点の座標を取得
    x_point_center = np.arange(mean_distance.shape[0])[center_agent]
    y_point_center = mean_distance[agent_id][center_agent]

    # 7番目の点から軸に向かって垂直・水平線を引く
    plt.vlines(x=x_point_center, ymin=0, ymax=1, colors='gray', linestyles='--')
    plt.hlines(y=y_point_center, xmin=0, xmax=mean_distance.shape[1]-1, colors='gray', linestyles='--')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(save_dir, f"mean_distance_from_agent{agent_id}_with_line.png"), dpi=300)
    plt.close()

    rank_matrix = mean_distance.argsort(axis=1).argsort(axis=1) + 1
    # 同心円分布かどうかを判定（枠線付き出力の条件に使用）
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
            
        # plt.colorbar(im)
        ax.set_xticks([])
        ax.set_yticks([])
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
        # 追加出力: 常に framed 版を保存。枠の有無は条件で切り替え
        framed_filename = f'distance_rank_matrix_heatmap_{cmap}_with_border_framed.png'
        if is_concentric and cmap == 'Blues':
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(8.0)
                spine.set_edgecolor('black')
            
            # 離散的なカラーバーを作成
            unique_ranks = np.unique(rank_matrix)
            num_ranks = len(unique_ranks)
            
            # 離散的なカラーマップを作成
            # Bluesカラーマップから必要な色数だけ取得
            blues_colors = plt.cm.Blues(np.linspace(0.2, 1.0, num_ranks))
            discrete_cmap = ListedColormap(blues_colors)
            
            # 境界を設定（各順位の境界）
            bounds = np.arange(unique_ranks.min() - 0.5, unique_ranks.max() + 1.5, 1)
            norm = BoundaryNorm(bounds, discrete_cmap.N)
            
            # 離散カラーマップでimageを再描画
            ax.clear()  # 既存のimageをクリア
            im_discrete = ax.imshow(rank_matrix, cmap=discrete_cmap, norm=norm, aspect='equal')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 枠線を再描画
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
            
            # 枠線を再設定
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(8.0)
                spine.set_edgecolor('black')
            
            # 離散的なカラーバーを追加
            if plot_discrete_colorbar:
                cbar = plt.colorbar(im_discrete, ax=ax, boundaries=bounds, 
                              ticks=unique_ranks, shrink=0.8, spacing='uniform')
                cbar.set_label('Distance Rank', rotation=270, labelpad=15)
                cbar.ax.tick_params(labelsize=8)
        else:
            for spine in ax.spines.values():
                spine.set_visible(False)
        plt.savefig(os.path.join(save_dir, framed_filename), dpi=300)
        plt.close(fig)

    # E[D^2]のプロット（mean_distance_euclidean_sqが提供されている場合）
    if mean_distance_euclidean_sq is not None:
        # 正規化（分散は(2*N_i)^2で正規化）
        mean_distance_squared_norm = mean_distance_euclidean_sq / ((2 * N_i) ** 2)
        extent_sq = [0, mean_distance_squared_norm.shape[0], 0, mean_distance_squared_norm.shape[1]]

        # Linear scale heatmap for E[D^2]
        plt.figure(figsize=(5, 5))
        im_sq = plt.imshow(mean_distance_squared_norm, extent=extent_sq,
                        cmap='Blues', aspect='equal', vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(save_dir, "mean_distance_squared_heatmap_Blues_no_colorbar.png"), dpi=300)
        plt.close()

        for cmap in ['Blues', 'Reds', 'Greens', 'Blues_Reds']:
            plt.figure(figsize=(5, 5))
            if cmap == 'Blues_Reds':
                custom_cmap = colors.LinearSegmentedColormap.from_list('Blues_Reds', ['blue','red'])
                im_sq = plt.imshow(mean_distance_squared_norm, extent=extent_sq,
                                cmap=custom_cmap, aspect='equal', vmin=0, vmax=1)
                filename_sq = "mean_distance_squared_heatmap_Blues_Reds.png"
            else:
                im_sq = plt.imshow(mean_distance_squared_norm, extent=extent_sq,
                                cmap=cmap, aspect='equal', vmin=0, vmax=1)
                filename_sq = f"mean_distance_squared_heatmap_{cmap}.png"
            plt.colorbar(im_sq)
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join(save_dir, filename_sq), dpi=300)
            plt.close()

        # Agent-specific E[D^2] plot
        plt.figure(figsize=(5, 5))
        plt.plot(np.arange(mean_distance_squared_norm.shape[0]), mean_distance_squared_norm[agent_id], marker='o')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(save_dir, f"mean_distance_squared_from_agent{agent_id}.png"), dpi=300)
        plt.close()

        # Rank matrix for E[D^2]
        rank_matrix_sq = mean_distance_euclidean_sq.argsort(axis=1).argsort(axis=1) + 1
        is_concentric_sq = is_concentric_distribution(mean_distance_euclidean_sq)

        for cmap in ['Blues', 'Reds', 'Greens', 'bwr', 'Blues_Reds', 'Greys', 'White_Blue']:
            filename_sq = f'distance_squared_rank_matrix_heatmap_{cmap}_with_border.png'

            fig_sq, ax_sq = plt.subplots(figsize=(5, 5))

            if cmap == 'Blues_Reds':
                custom_cmap = colors.LinearSegmentedColormap.from_list('Blues_Reds', ['blue', 'red'])
                im_sq = ax_sq.imshow(rank_matrix_sq, cmap=custom_cmap, aspect='equal')
            elif cmap == 'White_Blue':
                custom_cmap = colors.LinearSegmentedColormap.from_list('White_Blue', ['white', 'blue'])
                im_sq = ax_sq.imshow(rank_matrix_sq, cmap=custom_cmap, aspect='equal')
            else:
                im_sq = ax_sq.imshow(rank_matrix_sq, cmap=cmap, aspect='equal')

            ax_sq.set_xticks([])
            ax_sq.set_yticks([])
            rows_sq, cols_sq = rank_matrix_sq.shape
            center_edgecolor = "black"
            concentric_cell_edgecolor = "red"
            if cols_sq > center_agent:
                for i in range(rows_sq):
                    for j in range(cols_sq):
                        if j == center_agent and i != center_agent:
                            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                                fill=False,
                                                edgecolor=center_edgecolor,
                                                linewidth=2)
                            ax_sq.add_patch(rect)

                        if rank_matrix_sq[i, center_agent] > rank_matrix_sq[i, j] and (i - center_agent) * (j - center_agent) < 0:
                            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                                fill=False,
                                                edgecolor=concentric_cell_edgecolor,
                                                linewidth=2)
                            ax_sq.add_patch(rect)

            plt.savefig(os.path.join(save_dir, filename_sq), dpi=300)

            # Framed version for E[D^2]
            framed_filename_sq = f'distance_squared_rank_matrix_heatmap_{cmap}_with_border_framed.png'
            if is_concentric_sq and cmap == 'White_Blue':
                for spine in ax_sq.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(8.0)
                    spine.set_edgecolor('black')

                # 離散的なカラーバーを作成
                unique_ranks_sq = np.unique(rank_matrix_sq)
                num_ranks_sq = len(unique_ranks_sq)

                # 離散的なカラーマップを作成
                white_blue_colors = [(1, 1, 1), (0, 0, 1)]  # White to Blue
                discrete_cmap_sq = colors.LinearSegmentedColormap.from_list('WhiteBlue', white_blue_colors, N=num_ranks_sq)

                # 境界を設定（各順位の境界）
                bounds_sq = np.arange(unique_ranks_sq.min() - 0.5, unique_ranks_sq.max() + 1.5, 1)
                norm_sq = BoundaryNorm(bounds_sq, discrete_cmap_sq.N)

                # 離散カラーマップでimageを再描画
                ax_sq.clear()
                im_discrete_sq = ax_sq.imshow(rank_matrix_sq, cmap=discrete_cmap_sq, norm=norm_sq, aspect='equal')
                ax_sq.set_xticks([])
                ax_sq.set_yticks([])

                # 枠線を再描画
                if cols_sq > center_agent:
                    for i in range(rows_sq):
                        for j in range(cols_sq):
                            if j == center_agent and i != center_agent:
                                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                                    fill=False,
                                                    edgecolor=center_edgecolor,
                                                    linewidth=2)
                                ax_sq.add_patch(rect)

                            if rank_matrix_sq[i, center_agent] > rank_matrix_sq[i, j] and (i - center_agent) * (j - center_agent) < 0:
                                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                                    fill=False,
                                                    edgecolor=concentric_cell_edgecolor,
                                                    linewidth=2)
                                ax_sq.add_patch(rect)

                # 枠線を再設定
                for spine in ax_sq.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(8.0)
                    spine.set_edgecolor('black')

                # 離散的なカラーバーを追加
                if plot_discrete_colorbar:
                    cbar_sq = plt.colorbar(im_discrete_sq, ax=ax_sq, boundaries=bounds_sq,
                                  ticks=unique_ranks_sq, shrink=0.8, spacing='uniform')
                    cbar_sq.set_label('Distance Squared Rank', rotation=270, labelpad=15)
                    cbar_sq.ax.tick_params(labelsize=8)
            else:
                for spine in ax_sq.spines.values():
                    spine.set_visible(False)
            plt.savefig(os.path.join(save_dir, framed_filename_sq), dpi=300)
            plt.close(fig_sq)


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


def plot_similarity_analysis(similarities, mean_similarity, save_dir, similarity_type, center_agent, opposite_agent, args=None):
    """Plot similarity analysis (dot product or cosine)."""
    if similarities is None or mean_similarity is None:
        return None, None
    
    # Check if similarities is 3D (individual data) or 2D (mean data)
    if similarities.ndim == 3:
        # Individual similarity data available
        similarities_0_center = similarities[:, 0, center_agent]
        similarities_0_opposite = similarities[:, 0, opposite_agent]
        
        # Save similarity data
        df_similarities = pd.DataFrame({
            f'{similarity_type}_similarity_0_{center_agent}': similarities_0_center,
            f'{similarity_type}_similarity_0_{opposite_agent}': similarities_0_opposite
        })
        csv_path = os.path.join(save_dir, f'{similarity_type}_similarities_0_{center_agent}_0_{opposite_agent}.csv')
        df_similarities.to_csv(csv_path, index=False)
        
        # Histogram comparison
        shift_amount = 0.005 if similarity_type == 'dot' else 0.001
        plot_histogram_comparison(
            similarities_0_center, similarities_0_opposite,
            [f'Agent 0-{center_agent}', f'Agent 0-{opposite_agent}'],
            ['blue', 'red'],
            os.path.join(save_dir, f'agent_pair_{similarity_type}_similarities_histogram.png'),
            shift_amount=shift_amount
        )
        # Horizontal (axis-swapped) histogram
        plot_histogram_comparison(
            similarities_0_center, similarities_0_opposite,
            [f'Agent 0-{center_agent}', f'Agent 0-{opposite_agent}'],
            ['blue', 'red'],
            os.path.join(save_dir, f'agent_pair_{similarity_type}_similarities_histogram_horizontal.png'),
            shift_amount=shift_amount, orientation='horizontal'
        )
        # KDE overlay (Beta kernel). For cosine, data in [0,1]; for dot, scale by data max.
        # Note: Can be very slow with large datasets
        # Use --skip_kde flag to skip this computation
        if not hasattr(args, 'skip_kde') or not args.skip_kde:
            plot_histogram_comparison(
                similarities_0_center, similarities_0_opposite,
                [f'Agent 0-{center_agent}', f'Agent 0-{opposite_agent}'],
                ['blue', 'red'],
                os.path.join(save_dir, f'agent_pair_{similarity_type}_similarities_histogram_kde.png'),
                shift_amount=shift_amount, overlay_kde=True,
                unit_max_for_beta=(1.0 if similarity_type == 'cosine' else None)
            )
        
        # Individual similarities heatmap (new addition)
        plt.figure(figsize=(8, 6))
        vmax = 1 if similarity_type == 'cosine' else None
        im = plt.imshow(similarities, vmin=0, vmax=vmax, aspect="equal", cmap='viridis')
        plt.colorbar(im, label=f"{similarity_type.title()} Similarity")
        plt.title(f'Individual {similarity_type.title()} Similarities Over Time')
        plt.xlabel('Agent ID')
        plt.ylabel('Time Step')
        plt.xticks(np.arange(similarities.shape[2]))
        plt.yticks(np.arange(0, similarities.shape[0], max(1, similarities.shape[0]//10)))
        plt.savefig(os.path.join(save_dir, f"individual_{similarity_type}_similarities_heatmap.png"), dpi=300)
        plt.close()
        
        return similarities_0_center, similarities_0_opposite
    else:
        # Only mean similarity data available (memory-efficient mode)
        print(f"Note: Only mean {similarity_type} similarity data available. Skipping individual analysis plots.")
    
    # Heatmap (always available with mean data)
    plt.figure(figsize=(5, 5))
    vmax = 1 if similarity_type == 'cosine' else None
    im = plt.imshow(mean_similarity, vmin=0, vmax=vmax, aspect="equal", cmap='viridis')
    plt.colorbar(im, label=f"Mean {similarity_type.title()} Similarity")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(save_dir, f"mean_{similarity_type}_similarity_heatmap.png"), dpi=300)
    plt.close()
    
    # Agent 0 similarity plot (always available with mean data)
    plt.figure(figsize=(5, 5))
    plt.plot(np.arange(mean_similarity.shape[0]), mean_similarity[0], marker='o')
    plt.xticks([])
    plt.savefig(os.path.join(save_dir, f"mean_{similarity_type}_similarity_from_agent0.png"), dpi=300)
    plt.close()
    
    return None, None


def plot_similarity_matrix_heatmaps(mean_similarity, save_dir, similarity_type, N_i, center_agent):
    """Plot similarity matrix heatmaps in the same format as plot_mean_distance_analysis."""
    if mean_similarity is None:
        return
    
    # Normalize similarity matrix to 0-1 range
    similarity_normalized = mean_similarity.copy()
    if similarity_type == 'dot':
        # For dot product, normalize by N_i (maximum value is N_i)
        similarity_normalized = similarity_normalized / N_i
    # Cosine similarity is already in 0-1 range, no normalization needed
    
    extent = [0, similarity_normalized.shape[0], 0, similarity_normalized.shape[1]]
    
    # 1. Raw similarity matrix heatmap (Blues)
    plt.figure(figsize=(5, 5))
    im = plt.imshow(similarity_normalized, extent=extent, 
                    cmap='Blues', aspect='equal', vmin=0, vmax=1)
    # plt.colorbar(im, label=f"{similarity_type.title()} Similarity")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(save_dir, f"mean_{similarity_type}_similarity_heatmap_Blues.png"), dpi=300)
    plt.close()
    
    # 2. Rank matrix heatmap (Blues) with borders like distance analysis
    # Assign rank 1 to the largest value (descending order)
    rank_matrix = mean_similarity.argsort(axis=1)[:, ::-1].argsort(axis=1) + 1
    
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(rank_matrix, cmap='Blues', aspect='equal')
    # plt.colorbar(im, label=f"{similarity_type.title()} Similarity Rank")
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add borders like distance analysis
    rows, cols = rank_matrix.shape
    plt.savefig(os.path.join(save_dir, f'{similarity_type}_similarity_rank_matrix_heatmap_Blues.png'), dpi=300)
    center_edgecolor = "black"
    concentric_cell_edgecolor = "red"
    if cols > center_agent:
        for i in range(rows):
            for j in range(cols):
                if j == center_agent and i != center_agent:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                        fill=False, 
                                        edgecolor=center_edgecolor,
                                        linewidth=0.7)
                    ax.add_patch(rect)
                
                if rank_matrix[i, center_agent] > rank_matrix[i, j] and (i - center_agent) * (j - center_agent) < 0:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                        fill=False, 
                                        edgecolor=concentric_cell_edgecolor,
                                        linewidth=0.7)
                    ax.add_patch(rect)
    
    plt.savefig(os.path.join(save_dir, f'{similarity_type}_similarity_rank_matrix_heatmap_Blues_with_border.png'), dpi=300)

    # 3. Framed version (same style as distance_squared)
    # Always generate framed version regardless of concentric distribution
    is_concentric = is_concentric_distribution_similarity(mean_similarity)
    framed_filename = f'{similarity_type}_similarity_rank_matrix_heatmap_White_Blue_with_border_framed.png'

    if True:  # Always generate (was: if is_concentric)
        # 新しいfigureを作成
        fig_framed, ax_framed = plt.subplots(figsize=(5, 5))

        # 離散的なカラーバーを作成
        unique_ranks = np.unique(rank_matrix)
        num_ranks = len(unique_ranks)

        # White to Blue カラーマップ
        white_blue_colors = [(1, 1, 1), (0, 0, 1)]
        discrete_cmap = colors.LinearSegmentedColormap.from_list('WhiteBlue', white_blue_colors, N=num_ranks)

        # 境界を設定（各順位の境界）
        bounds = np.arange(unique_ranks.min() - 0.5, unique_ranks.max() + 1.5, 1)
        norm = BoundaryNorm(bounds, discrete_cmap.N)

        # 離散カラーマップでimageを描画
        im_discrete = ax_framed.imshow(rank_matrix, cmap=discrete_cmap, norm=norm, aspect='equal')
        ax_framed.set_xticks([])
        ax_framed.set_yticks([])

        # 枠線を描画
        rows, cols = rank_matrix.shape
        center_edgecolor = "black"
        concentric_cell_edgecolor = "red"
        if cols > center_agent:
            for i in range(rows):
                for j in range(cols):
                    # 黒い枠線（center agent列）は常に描画
                    if j == center_agent and i != center_agent:
                        rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            fill=False,
                                            edgecolor=center_edgecolor,
                                            linewidth=2)
                        ax_framed.add_patch(rect)

                    # 類似度の場合も同じ条件: rankが大きい=類似度が低い
                    # 赤い枠線（concentric cells）は is_concentric の時のみ描画
                    if is_concentric and rank_matrix[i, center_agent] > rank_matrix[i, j] and (i - center_agent) * (j - center_agent) < 0:
                        rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            fill=False,
                                            edgecolor=concentric_cell_edgecolor,
                                            linewidth=2)
                        ax_framed.add_patch(rect)

        # 外枠を追加 (only if is_concentric)
        if is_concentric:
            for spine in ax_framed.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(8.0)
                spine.set_edgecolor('black')

        plt.savefig(os.path.join(save_dir, framed_filename), dpi=300)
        plt.close(fig_framed)
    else:
        # concentricでない場合は枠なし
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.close(fig)


def plot_f_matrix_heatmaps(mean_f_matrix, mean_one_minus_f, mean_nei_distance, save_dir, N_i, center_agent):
    """
    Plot F-matrix, 1-F, and Nei's distance heatmaps in the same style as similarity heatmaps.

    This creates heatmaps visualizing:
    1. F-matrix (IBD probability): F_ij = P(data from agent i and j are identical)
    2. 1-F: (1-F)_ij = 1 - F_ij
    3. Nei's distance: D_ij = -ln(F_ij / sqrt(F_ii * F_jj))
    4. Rank matrices with concentric distribution detection

    Args:
        mean_f_matrix: Time-averaged F-matrix (M x M)
        mean_one_minus_f: Time-averaged 1-F (M x M)
        mean_nei_distance: Time-averaged Nei's distance (M x M)
        save_dir: Directory to save plots
        N_i: Population size per agent
        center_agent: Index of center agent for border detection
    """
    if mean_f_matrix is None:
        return

    # Check for concentric distribution independently for each metric
    # F-matrix is similarity (higher = more similar), so use similarity version
    is_concentric_f = is_concentric_distribution_similarity(mean_f_matrix)
    # 1-F and Nei's distance are distances (higher = less similar), so use distance version
    is_concentric_1f = is_concentric_distribution(mean_one_minus_f)
    is_concentric_nei = is_concentric_distribution(mean_nei_distance)

    # ========== F-Matrix Heatmaps ==========
    # 1. Raw F-matrix heatmap (Blues)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(mean_f_matrix, cmap='Blues', aspect='equal', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(save_dir, 'f_matrix_heatmap_Blues.png'), dpi=300)
    plt.close()

    # 2. F-matrix rank matrix heatmap (higher F = more similar = lower rank)
    # Assign rank 1 to the largest F value (descending order, like similarity)
    rank_matrix_f = mean_f_matrix.argsort(axis=1)[:, ::-1].argsort(axis=1) + 1

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(rank_matrix_f, cmap='Blues', aspect='equal')
    ax.set_xticks([])
    ax.set_yticks([])

    # Add borders
    rows, cols = rank_matrix_f.shape
    plt.savefig(os.path.join(save_dir, 'f_matrix_rank_matrix_heatmap_Blues.png'), dpi=300)

    center_edgecolor = "black"
    concentric_cell_edgecolor = "red"
    if cols > center_agent:
        for i in range(rows):
            for j in range(cols):
                if j == center_agent and i != center_agent:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        fill=False,
                                        edgecolor=center_edgecolor,
                                        linewidth=0.7)
                    ax.add_patch(rect)

                # For F-matrix: rank > rank[center] means less similar
                if rank_matrix_f[i, center_agent] > rank_matrix_f[i, j] and (i - center_agent) * (j - center_agent) < 0:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        fill=False,
                                        edgecolor=concentric_cell_edgecolor,
                                        linewidth=0.7)
                    ax.add_patch(rect)

    plt.savefig(os.path.join(save_dir, 'f_matrix_rank_matrix_heatmap_Blues_with_border.png'), dpi=300)

    # 3. Framed version with White-Blue colormap
    framed_filename = 'f_matrix_rank_matrix_heatmap_White_Blue_with_border_framed.png'

    # Always generate framed version
    fig_framed, ax_framed = plt.subplots(figsize=(5, 5))

    # Discrete colormap
    unique_ranks = np.unique(rank_matrix_f)
    num_ranks = len(unique_ranks)

    # White to Blue colormap
    white_blue_colors = [(1, 1, 1), (0, 0, 1)]
    discrete_cmap = colors.LinearSegmentedColormap.from_list('WhiteBlue', white_blue_colors, N=num_ranks)

    # Boundaries
    bounds = np.arange(unique_ranks.min() - 0.5, unique_ranks.max() + 1.5, 1)
    norm = BoundaryNorm(bounds, discrete_cmap.N)

    # Draw with discrete colormap
    im_discrete = ax_framed.imshow(rank_matrix_f, cmap=discrete_cmap, norm=norm, aspect='equal')
    ax_framed.set_xticks([])
    ax_framed.set_yticks([])

    # Add borders
    if cols > center_agent:
        for i in range(rows):
            for j in range(cols):
                # Black border for center agent column (always)
                if j == center_agent and i != center_agent:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        fill=False,
                                        edgecolor=center_edgecolor,
                                        linewidth=2)
                    ax_framed.add_patch(rect)

                # Red border for concentric cells (only if is_concentric_f)
                if is_concentric_f and rank_matrix_f[i, center_agent] > rank_matrix_f[i, j] and (i - center_agent) * (j - center_agent) < 0:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        fill=False,
                                        edgecolor=concentric_cell_edgecolor,
                                        linewidth=2)
                    ax_framed.add_patch(rect)

    # Outer frame (only if is_concentric_f)
    if is_concentric_f:
        for spine in ax_framed.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(8.0)
            spine.set_edgecolor('black')

    plt.savefig(os.path.join(save_dir, framed_filename), dpi=300)
    plt.close(fig_framed)
    plt.close(fig)

    # ========== 1-F Heatmaps ==========
    # Similar to F-matrix but for 1-F
    # Higher 1-F = less similar (like Manhattan distance)
    # Assign rank 1 to the smallest value (ascending order)
    rank_matrix_one_minus_f = mean_one_minus_f.argsort(axis=1).argsort(axis=1) + 1

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(rank_matrix_one_minus_f, cmap='Blues', aspect='equal')
    ax.set_xticks([])
    ax.set_yticks([])

    rows, cols = rank_matrix_one_minus_f.shape
    plt.savefig(os.path.join(save_dir, 'one_minus_f_rank_matrix_heatmap_Blues.png'), dpi=300)

    # Add borders
    if cols > center_agent:
        for i in range(rows):
            for j in range(cols):
                if j == center_agent and i != center_agent:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        fill=False,
                                        edgecolor=center_edgecolor,
                                        linewidth=0.7)
                    ax.add_patch(rect)

                # For 1-F: rank > rank[center] means farther
                if rank_matrix_one_minus_f[i, center_agent] > rank_matrix_one_minus_f[i, j] and (i - center_agent) * (j - center_agent) < 0:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        fill=False,
                                        edgecolor=concentric_cell_edgecolor,
                                        linewidth=0.7)
                    ax.add_patch(rect)

    plt.savefig(os.path.join(save_dir, 'one_minus_f_rank_matrix_heatmap_Blues_with_border.png'), dpi=300)

    # Framed version for 1-F
    framed_filename_one_minus_f = 'one_minus_f_rank_matrix_heatmap_White_Blue_with_border_framed.png'

    fig_framed_one_minus_f, ax_framed_one_minus_f = plt.subplots(figsize=(5, 5))

    unique_ranks_one_minus_f = np.unique(rank_matrix_one_minus_f)
    num_ranks_one_minus_f = len(unique_ranks_one_minus_f)

    discrete_cmap_one_minus_f = colors.LinearSegmentedColormap.from_list('WhiteBlue', white_blue_colors, N=num_ranks_one_minus_f)
    bounds_one_minus_f = np.arange(unique_ranks_one_minus_f.min() - 0.5, unique_ranks_one_minus_f.max() + 1.5, 1)
    norm_one_minus_f = BoundaryNorm(bounds_one_minus_f, discrete_cmap_one_minus_f.N)

    im_discrete_one_minus_f = ax_framed_one_minus_f.imshow(rank_matrix_one_minus_f, cmap=discrete_cmap_one_minus_f, norm=norm_one_minus_f, aspect='equal')
    ax_framed_one_minus_f.set_xticks([])
    ax_framed_one_minus_f.set_yticks([])

    if cols > center_agent:
        for i in range(rows):
            for j in range(cols):
                if j == center_agent and i != center_agent:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        fill=False,
                                        edgecolor=center_edgecolor,
                                        linewidth=2)
                    ax_framed_one_minus_f.add_patch(rect)

                if is_concentric_1f and rank_matrix_one_minus_f[i, center_agent] > rank_matrix_one_minus_f[i, j] and (i - center_agent) * (j - center_agent) < 0:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        fill=False,
                                        edgecolor=concentric_cell_edgecolor,
                                        linewidth=2)
                    ax_framed_one_minus_f.add_patch(rect)

    if is_concentric_1f:
        for spine in ax_framed_one_minus_f.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(8.0)
            spine.set_edgecolor('black')

    plt.savefig(os.path.join(save_dir, framed_filename_one_minus_f), dpi=300)
    plt.close(fig_framed_one_minus_f)
    plt.close(fig)

    # ========== Nei's Distance Heatmaps ==========
    # Nei's distance: D_ij = -ln(F_ij / sqrt(F_ii * F_jj))
    # Higher Nei's distance = less similar (like genetic distance)
    # Assign rank 1 to the smallest distance (ascending order)
    rank_matrix_nei = mean_nei_distance.argsort(axis=1).argsort(axis=1) + 1

    fig_nei, ax_nei = plt.subplots(figsize=(5, 5))
    im_nei = ax_nei.imshow(rank_matrix_nei, cmap='Blues', aspect='equal')
    ax_nei.set_xticks([])
    ax_nei.set_yticks([])

    rows, cols = rank_matrix_nei.shape
    plt.savefig(os.path.join(save_dir, 'nei_distance_rank_matrix_heatmap_Blues.png'), dpi=300)

    # Add borders
    if cols > center_agent:
        for i in range(rows):
            for j in range(cols):
                if j == center_agent and i != center_agent:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        fill=False,
                                        edgecolor=center_edgecolor,
                                        linewidth=0.7)
                    ax_nei.add_patch(rect)

                # For Nei's distance: rank > rank[center] means farther
                if rank_matrix_nei[i, center_agent] > rank_matrix_nei[i, j] and (i - center_agent) * (j - center_agent) < 0:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        fill=False,
                                        edgecolor=concentric_cell_edgecolor,
                                        linewidth=0.7)
                    ax_nei.add_patch(rect)

    plt.savefig(os.path.join(save_dir, 'nei_distance_rank_matrix_heatmap_Blues_with_border.png'), dpi=300)

    # Framed version for Nei's distance
    framed_filename_nei = 'nei_distance_rank_matrix_heatmap_White_Blue_with_border_framed.png'

    fig_framed_nei, ax_framed_nei = plt.subplots(figsize=(5, 5))

    unique_ranks_nei = np.unique(rank_matrix_nei)
    num_ranks_nei = len(unique_ranks_nei)

    discrete_cmap_nei = colors.LinearSegmentedColormap.from_list('WhiteBlue', white_blue_colors, N=num_ranks_nei)
    bounds_nei = np.arange(unique_ranks_nei.min() - 0.5, unique_ranks_nei.max() + 1.5, 1)
    norm_nei = BoundaryNorm(bounds_nei, discrete_cmap_nei.N)

    im_discrete_nei = ax_framed_nei.imshow(rank_matrix_nei, cmap=discrete_cmap_nei, norm=norm_nei, aspect='equal')
    ax_framed_nei.set_xticks([])
    ax_framed_nei.set_yticks([])

    if cols > center_agent:
        for i in range(rows):
            for j in range(cols):
                if j == center_agent and i != center_agent:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        fill=False,
                                        edgecolor=center_edgecolor,
                                        linewidth=2)
                    ax_framed_nei.add_patch(rect)

                if is_concentric_nei and rank_matrix_nei[i, center_agent] > rank_matrix_nei[i, j] and (i - center_agent) * (j - center_agent) < 0:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        fill=False,
                                        edgecolor=concentric_cell_edgecolor,
                                        linewidth=2)
                    ax_framed_nei.add_patch(rect)

    if is_concentric_nei:
        for spine in ax_framed_nei.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(8.0)
            spine.set_edgecolor('black')

    plt.savefig(os.path.join(save_dir, framed_filename_nei), dpi=300)
    plt.close(fig_framed_nei)
    plt.close(fig_nei)

    print(f"✓ F-matrix, 1-F, and Nei's distance heatmaps saved")
    print(f"  is_concentric_f: {is_concentric_f}, is_concentric_1f: {is_concentric_1f}, is_concentric_nei: {is_concentric_nei}")


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
    
    # Similarity tests
    if similarities_dot_0_center is not None:
        print("\n=== Dot Product Similarity Binomial test results ===")
        over_count_dot = np.sum(similarities_dot_0_center > similarities_dot_0_opposite)
        under_count_dot = np.sum(similarities_dot_0_center < similarities_dot_0_opposite)
        total_dot = len(similarities_dot_0_center)
        p_value_dot = stats.binomtest(over_count_dot, n=total_dot, p=0.5, alternative='two-sided').pvalue
        
        print(f"Number of times 0-{center_agent} > 0-{opposite_agent}: {over_count_dot}/{total_dot}")
        print(f"p-value: {p_value_dot:.3e}")
        print(f"mean dot similarity is concentric: {is_concentric_distribution(-mean_similarity_dot)}")
    
    if similarities_cosine_0_center is not None:
        print("\n=== Cosine Similarity Binomial test results ===")
        over_count_cosine = np.sum(similarities_cosine_0_center > similarities_cosine_0_opposite)
        under_count_cosine = np.sum(similarities_cosine_0_center < similarities_cosine_0_opposite)
        total_cosine = len(similarities_cosine_0_center)
        p_value_cosine = stats.binomtest(over_count_cosine, n=total_cosine, p=0.5, alternative='two-sided').pvalue
        
        print(f"Number of times 0-{center_agent} > 0-{opposite_agent}: {over_count_cosine}/{total_cosine}")
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


# =============================================================================
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 変更・追加箇所 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# =============================================================================

def run_single_pair_regression(df_pair):
    """特定の(i, j)ペアのデータフレームを受け取り、回帰分析を実行して係数とp値を返す"""
    # データが少なすぎる、またはyの値が全て0か1の場合は分析できない
    if len(df_pair) < 4 or df_pair['y'].nunique() < 2:
        # 係数とp値の両方をNaNで返すように変更
        return [np.nan] * 4, [np.nan] * 4 

    try:
        X = df_pair[['x1', 'x2', 'x1_x2']]
        y = df_pair['y']
        X_sm = sm.add_constant(X, prepend=True)
        model = sm.Logit(y, X_sm).fit(disp=0)
        # 係数 (model.params) と p値 (model.pvalues) の両方を返すように変更
        return model.params.values, model.pvalues.values
    except Exception:
        # エラー時も同様に係数とp値の両方をNaNで返すように変更
        return [np.nan] * 4, [np.nan] * 4


def plot_pvalue_heatmaps(df_results, save_dir):
    """
    ロジスティック回帰の各係数のp値についてヒートマップを作成する。
    p値は「係数が0である」という帰無仮説が正しい場合に、観測データ以上の結果が得られる確率。
    """
    print("Creating p-value heatmaps...")
    coeff_names = ['Intercept', 'x1_center', 'x2_outward', 'x1_x2_interaction']
    
    for coeff_name in coeff_names:
        p_name = f'p_{coeff_name}'
        # p値の列が存在するかチェック
        if p_name not in df_results.columns:
            print(f"P-value column '{p_name}' not found in results. Skipping heatmap.")
            continue
            
        heatmap_data = df_results.pivot(index='i', columns='j', values=p_name)
        print(save_dir)
        print(heatmap_data)
        
        plt.figure(figsize=(5, 5))
        # p値が小さいほど色が濃くなるように、cmapに`_r`をつける (例: viridis_r)
        # vmax=0.1とすることで、有意水準(e.g., 0.05)付近の変化を強調する
        im = plt.imshow(heatmap_data, aspect="equal", cmap='viridis_r', 
                       vmin=0, vmax=0.1) 
        plt.colorbar(im, label=f'P-value for "{coeff_name}"')
        plt.xticks([])
        plt.yticks([])

        save_path = os.path.join(save_dir, f'heatmap_pvalue_{coeff_name}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved p-value heatmap to {save_path}")


def analyze_and_plot_by_pair(args):
    """
    各(i, j)ペアに対してロジスティック回帰を実行し、結果をヒートマップとして保存する。
    結果は保存され、既に結果があればそれをロードする。
    """

    results_file = os.path.join(args.save_dir, 'pair_analysis_results.pkl')
    
    if os.path.exists(results_file):
        print(f"Loading existing results from {results_file}")
        with open(results_file, 'rb') as f:
            df_results = pickle.load(f)
        print("Results loaded successfully.")
    else:
        print("Starting analysis for each (i, j) pair. This may take a very long time...")
        combinations = get_combinations(args.nonzero_alpha, args.flow_type)
        hub_agent = args.center_agent
        agents_count = args.agents_count
        results = []
        valid_pairs = []
        for i in range(agents_count):
            for j in range(agents_count):
                if i == hub_agent or j == hub_agent or i == j:
                    continue
                if (i - hub_agent) * (j - hub_agent) < 0:
                    valid_pairs.append((i, j))

        for i, j in tqdm(valid_pairs, desc="Analyzing pairs"):
            pair_data = []
            for na, ft in combinations:
                x1 = 1 if na == 'center' else 0
                x2 = 1 if ft == 'outward' else 0
                load_dir, _, _ = get_directory_paths(ft, na, args.coupling_strength, args.agents_count, args.N_i, args.alpha_per_data)
                distance_files = sorted(glob.glob(os.path.join(load_dir, "distance_*.npy")))
                for file_path in distance_files[args.skip:]:
                    snapshot_d = np.load(file_path)
                    val_i_hub = snapshot_d[i, hub_agent]
                    val_i_j = snapshot_d[i, j]
                    y_val = int(val_i_hub > val_i_j)
                    pair_data.append({'y': y_val, 'x1': x1, 'x2': x2, 'x1_x2': x1 * x2})
            df_pair = pd.DataFrame(pair_data)

            # run_single_pair_regressionから係数とp値の両方を受け取るように変更
            coeffs, pvals = run_single_pair_regression(df_pair)
            
            # 結果辞書にp値も追加
            results.append({
                'i': i, 'j': j,
                'Intercept': coeffs[0],
                'x1_center': coeffs[1],
                'x2_outward': coeffs[2],
                'x1_x2_interaction': coeffs[3],
                'p_Intercept': pvals[0],
                'p_x1_center': pvals[1],
                'p_x2_outward': pvals[2],
                'p_x1_x2_interaction': pvals[3]
            })
        
        df_results = pd.DataFrame(results)
        
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"Saving results to {results_file}")
        with open(results_file, 'wb') as f:
            pickle.dump(df_results, f)
        print("Results saved successfully.")
    
    # --- 係数のヒートマップ作成 ---
    print("Creating coefficient heatmaps...")
    coeff_names = ['Intercept', 'x1_center', 'x2_outward', 'x1_x2_interaction']
    os.makedirs(args.save_dir, exist_ok=True)
    for coeff_name in coeff_names:
        heatmap_data = df_results.pivot(index='i', columns='j', values=coeff_name)

        plt.figure(figsize=(5, 5))
        # heatmap_data.valuesが全てnanの場合、nanmaxはnanを返すため、チェックを追加
        if not np.all(np.isnan(heatmap_data.values)):
            max_abs_val = np.nanmax(np.abs(heatmap_data.values))
        else:
            max_abs_val = 1.0 # 全てnanならデフォルト値を使用

        # max_abs_valが0やnanの場合のフォールバック処理
        if np.isnan(max_abs_val) or max_abs_val == 0:
            max_abs_val = 1.0

        im = plt.imshow(heatmap_data, aspect="equal", cmap='coolwarm', 
                       vmin=-max_abs_val, vmax=max_abs_val)
        plt.colorbar(im, label=f'Coefficient for "{coeff_name}"')
        plt.xticks([])
        plt.yticks([])

        save_path = os.path.join(args.save_dir, f'heatmap_{coeff_name}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved coefficient heatmap to {save_path}")

    # --- p値のヒートマップ作成（新規追加） ---
    # 新しく定義した関数を呼び出す
    plot_pvalue_heatmaps(df_results, args.save_dir)



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
    
    concentric_generators = []
    hub_comparison_generators = []
    
    for na, ft in combinations:
        print(f"\n{'='*50}")
        print(f"Processing: {na} + {ft}")
        print(f"{'='*50}")
        
        load_dir, save_dir, distance_dir = get_directory_paths(
            ft, na, args.coupling_strength, args.agents_count, args.N_i, args.alpha_per_data
        )
        print(load_dir)
        os.makedirs(save_dir, exist_ok=True)
        
        distance_files = sorted(glob.glob(os.path.join(load_dir, "distance_*.npy")))
        # Exclude distance_euclidean_sq files from distance_files
        distance_files = [f for f in distance_files if "distance_euclidean_sq" not in f]
        print(f"Number of distance files found: {len(distance_files)}")

        distance_euclidean_sq_files = sorted(glob.glob(os.path.join(load_dir, "distance_euclidean_sq_*.npy")))
        print(f"Number of Euclidean squared distance files found: {len(distance_euclidean_sq_files)}")

        if not distance_files:
            print(f"No distance files found in {load_dir}. Skipping...")
            continue

        distance_files = distance_files[args.skip:]
        distance_euclidean_sq_files = distance_euclidean_sq_files[args.skip:]
        
        # Create age files proactively if requested or required by plotting
        if args.make_age_files:
            state_files = sorted(glob.glob(os.path.join(load_dir, "state_*.npy")))
            create_age_files(load_dir, save_dir, state_files)
        elif args.plot_age:
            # If age plotting is requested but age files are missing, generate them
            age_mean_file = os.path.join(save_dir, 'word_age_mean_per_agent.csv')
            if not os.path.exists(age_mean_file):
                state_files = sorted(glob.glob(os.path.join(load_dir, "state_*.npy")))
                if state_files:
                    print("Age files not found; generating them for plot_age...")
                    create_age_files(load_dir, save_dir, state_files)
                else:
                    print("No state_*.npy files found; cannot generate age files for plot_age.")
        
        # 必要な時だけ類似度データを読み込み
        mean_similarity_dot = None
        mean_similarity_cosine = None
        dot_similarities = None
        cosine_similarities = None
        
        if args.plot_similarity or args.check_concentric:
            (mean_similarity_dot, mean_similarity_cosine,
             dot_similarities, cosine_similarities) = load_similarity_data(
                load_dir, force_recompute=args.force_recompute
            )
        
        mean_distance = None
        mean_distance_euclidean_sq = None
        distances = None
        distances_euclidean_sq = None
        if args.plot_distance:
            print(f"Loading distance data from {len(distance_files)} files...")
            # ヒストグラム生成のため個別データも読み込む
            distances = np.array([np.load(f) for f in distance_files])
            mean_distance = np.mean(distances, axis=0)
            print(mean_distance)
            print("Distance data loaded successfully.")

            # ユークリッド距離の2乗を読み込み（最初から2乗で計算されたもの）
            if len(distance_euclidean_sq_files) > 0:
                print(f"Loading Euclidean squared distance data from {len(distance_euclidean_sq_files)} files...")
                distances_euclidean_sq = np.array([np.load(f) for f in distance_euclidean_sq_files])
                mean_distance_euclidean_sq = np.mean(distances_euclidean_sq, axis=0)
                print("Euclidean squared distance data loaded successfully.")
            else:
                print("No Euclidean squared distance files found. Skipping Euclidean distance analysis.")

        if args.plot_distance and mean_distance is not None:
            # Save distance data
            if distances is not None:
                save_distance_data(distances, distance_dir, args.center_agent, args.opposite_agent)
            # 平均距離の分析（E[D]とE[D_euclidean^2]の両方）
            # mean_distance_euclidean_sqが必須（マンハッタン距離の2乗とユークリッド距離の2乗は異なるため）
            if mean_distance_euclidean_sq is None:
                raise ValueError(
                    f"Euclidean squared distance files not found in {load_dir}.\n"
                    f"Please run simulation with --recompute_distance flag to generate distance_euclidean_sq_*.npy files.\n"
                    f"Note: E[(D_Manhattan)^2] and E[D_Euclidean^2] are fundamentally different metrics."
                )
            plot_mean_distance_analysis(mean_distance, save_dir, args.agent_id, args.N_i, args.center_agent, args.plot_discrete_colorbar, mean_distance_euclidean_sq)
            # ヒストグラムを含む距離分析
            if distances is not None:
                plot_distance_analysis(distances, save_dir, args.N_i, args.center_agent, args.opposite_agent, args)
        
        if args.plot_age:
            plot_age_analysis(save_dir)
        
        if args.plot_similarity:
            plot_similarity_analysis(dot_similarities, mean_similarity_dot, save_dir, 'dot', args.center_agent, args.opposite_agent, args)
            plot_similarity_analysis(cosine_similarities, mean_similarity_cosine, save_dir, 'cosine', args.center_agent, args.opposite_agent, args)

            # Add similarity matrix heatmaps (similar to distance heatmaps)
            plot_similarity_matrix_heatmaps(mean_similarity_dot, save_dir, 'dot', args.N_i, args.center_agent)
            plot_similarity_matrix_heatmaps(mean_similarity_cosine, save_dir, 'cosine', args.N_i, args.center_agent)

            # F-matrix, 1-F, and Nei's distance analysis (dot product similarity = F-matrix = IBD probability)
            print("\n" + "="*60)
            print("F-Matrix, 1-F, and Nei's Distance Analysis")
            print("="*60)
            mean_f_matrix, mean_one_minus_f, mean_nei_distance, f_matrices, one_minus_f_distances, nei_distances = load_f_matrix_data(load_dir, force_recompute=args.force_recompute)
            plot_f_matrix_heatmaps(mean_f_matrix, mean_one_minus_f, mean_nei_distance, save_dir, args.N_i, args.center_agent)

        if args.check_concentric:
            print("Checking concentric distribution patterns using memory-efficient processing...")
            # メモリ効率的に同心円分布をチェック
            concentric_results = []
            for f in distance_files:
                d = np.load(f)
                is_concentric = is_concentric_distribution(d)
                concentric_results.append(is_concentric)
            concentric_rate = np.mean(concentric_results)
            print(f"Concentric distribution rate: {concentric_rate:.3f}")
            
            # 二項検定は個別データが必要なので、メモリ効率的に計算
            if len(distance_files) > 0:
                # 必要な部分のみを計算
                distances_0_center = []
                distances_0_opposite = []
                
                for f in distance_files:
                    snapshot_d = np.load(f)
                    distances_0_center.append(snapshot_d[0, args.center_agent] / (2 * args.N_i))
                    distances_0_opposite.append(snapshot_d[0, args.opposite_agent] / (2 * args.N_i))
                
                distances_0_center = np.array(distances_0_center)
                distances_0_opposite = np.array(distances_0_opposite)
                
                # 類似度データも同様に処理
                similarities_dot_0_center = similarities_dot_0_opposite = None
                similarities_cosine_0_center = similarities_cosine_0_opposite = None
                
                if dot_similarities is not None:
                    similarities_dot_0_center = dot_similarities[:, 0, args.center_agent]
                    similarities_dot_0_opposite = dot_similarities[:, 0, args.opposite_agent]
                    
                if cosine_similarities is not None:
                    similarities_cosine_0_center = cosine_similarities[:, 0, args.center_agent]
                    similarities_cosine_0_opposite = cosine_similarities[:, 0, args.opposite_agent]
                
                perform_binomial_tests(
                    distances_0_center, distances_0_opposite,
                    similarities_dot_0_center, similarities_dot_0_opposite,
                    similarities_cosine_0_center, similarities_cosine_0_opposite,
                    mean_distance, mean_similarity_dot, mean_similarity_cosine,
                    args.center_agent, args.opposite_agent
                )
        
        if args.logistic_regression:
            concentric_gen = generate_concentric_data(distance_files, na, ft)
            concentric_generators.append(concentric_gen)
            
            dist_gen = generate_hub_comparison_data_from_files(
                distance_files, na, ft, args.agents_count, hub_agent=args.center_agent
            )
            hub_comparison_generators.append(dist_gen)
    
    if args.logistic_regression and hub_comparison_generators:
        print("\n" + "="*60)
        print("Performing final logistic regression on all combined data...")
        print("="*60)

        final_concentric_gen = itertools.chain.from_iterable(concentric_generators)
        perform_logistic_regression(final_concentric_gen, "Concentric Distribution")
        
        final_dist_gen = itertools.chain.from_iterable(hub_comparison_generators)
        perform_logistic_regression(final_dist_gen, f"Hub Comparison: d(i, {args.center_agent}) > d(i, j)")

    if hasattr(args, 'pairwise_regression') and args.pairwise_regression:
        if not hasattr(args, 'save_dir') or not args.save_dir:
            args.save_dir = os.path.join(get_data_fig_dir(), 'pairwise_regression_heatmaps')
        os.makedirs(args.save_dir, exist_ok=True)
        analyze_and_plot_by_pair(args)


if __name__ == "__main__":
    main()
