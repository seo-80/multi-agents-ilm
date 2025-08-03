import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import argparse
import scipy.stats as stats
from tqdm import tqdm
from sklearn.metrics import classification_report # 評価のためにこれは残す
import pandas as pd
import statsmodels.api as sm
import matplotlib.colors as colors


parser = argparse.ArgumentParser()
# 省略形も受け付けるようにする
parser.add_argument('--nonzero_alpha', '-a', type=str, default='all', choices=['evenly', 'center', 'all'], help='nonzero_alpha: "evenly", "center", or "all"')
parser.add_argument('--flow_type', '-f', type=str, default='all', choices=['bidirectional', 'outward', 'all'], help='flow_type: "bidirectional", "outward", or "all"')
parser.add_argument('--skip', '-s', type=int, default=0, help='何ファイルスキップするか (default: 0)')
parser.add_argument('--agent_id', '-i', type=int, default=0, help='棒グラフで表示するエージェントID (default: 0)')
parser.add_argument('--agents_count', '-m', type=int, default=15, help='Number of agents (default: 15)')
parser.add_argument('--N_i', '-n', type=int, default=100, help='Number of data per subpopulation (default: 100)')
parser.add_argument('--coupling_strength', '-c', type=float, default=0.01, help='Coupling strength (default: 0.05)')
parser.add_argument('--check_concentric', action='store_true', help='If set, check for concentric distribution')
parser.add_argument('--plot_age', action='store_true', help='If set, plot age of words')
parser.add_argument('--plot_similarity', action='store_true', help='If set, plot similarity of words')
parser.add_argument('--plot_similarity_heatmap', action='store_true', help='If set, plot similarity heatmap')
parser.add_argument('--make_age_files', action='store_true', help='If set, make age files from state files')
args, unknown = parser.parse_known_args()

nonzero_alpha = args.nonzero_alpha
flow_type = args.flow_type
skip = args.skip
agent_id = args.agent_id
agents_count = args.agents_count
N_i = args.N_i
coupling_strength = args.coupling_strength


# --- 設定 ---
def get_combinations(nonzero_alpha, flow_type):
    nonzero_alpha_options = ['evenly', 'center'] if nonzero_alpha == 'all' else [nonzero_alpha]
    flow_type_options = ['bidirectional', 'outward'] if flow_type == 'all' else [flow_type]
    return [(na, ft) for na in nonzero_alpha_options for ft in flow_type_options]


def is_concentric_distribution(distance_matrix):
    """
    Check if the distance matrix shows a concentric distribution.
    Returns True if there exists at least one base point where the distribution is concentric.
    """
    center = len(distance_matrix) // 2
    
    for base in range(len(distance_matrix)):
        if base == center:
            continue
        
        for reference in range(len(distance_matrix)):
            # referenceがcenterを挟んでbaseの反対側にあるか判定
            is_opposite_side = (base - center) * (reference - center) < 0
            
            # 反対側にあり、かつ言語的に近い場合
            if is_opposite_side and distance_matrix[base][reference] < distance_matrix[base][center]:
                return True

    return False

 
combinations = get_combinations(nonzero_alpha, flow_type)

# ロジスティック回帰のためのデータを収集
logistic_data_concentric = []  # 同心円分布用
logistic_data_distance = []    # 距離大小関係用
logistic_data_dot_sim = []     # ドット積類似度用  
logistic_data_cosine_sim = []  # コサイン類似度用

for na, ft in combinations:
    # --- subdir logic to match naive_simulation.py ---
    if ft == 'bidirectional':
        flow_str = 'bidirectional_flow-'
    elif ft == 'outward':
        flow_str = 'outward_flow-'
    else:
        raise ValueError(f"Unknown flow_type: {ft}")
    subdir = f"{flow_str}nonzero_alpha_{na}_fr_{coupling_strength}_agents_{agents_count}_N_i_{N_i}"
    load_dir = f"data/naive_simulation/raw/{subdir}"
    save_dir = f"data/naive_simulation/fig/{subdir}"
    distance_files = sorted(glob.glob(os.path.join(load_dir, "distance_*.npy")))
    print(f"[{ft}, {na}] Number of distance files found: {len(distance_files)}")
    distance_files = distance_files[skip:]  # スキップ数を適用

    if args.make_age_files:
    
    
        print(f" Generating age files...")
        state_files = sorted(glob.glob(os.path.join(load_dir, "state_*.npy")))
        idx_t_map = np.loadtxt(os.path.join(load_dir, "save_idx_t_map.csv"), delimiter=',', dtype=int, skiprows=1)
        id2t = {id_: t for id_, t in idx_t_map}
        if not state_files:
            print(f"No state files found in {load_dir}. Skipping...")
        else:
            # --- ここから修正: エージェントごとのage_mean, age_varをファイル分割で保存 ---
            age_means_per_agent = []
            age_vars_per_agent = []
            file_ids_per_agent = []
            agent_ids_per_agent = []
            for state_file in tqdm(state_files):
                basename = os.path.basename(state_file)
                file_id = int(basename.split('_')[1].split('.')[0])
                file_t = id2t[file_id]
                state = np.load(state_file)  # (agents_count, N_i, 3)
                word_ts = state[..., 0]      # (agents_count, N_i)
                ages = file_t - word_ts      # (agents_count, N_i)
                for agent in range(ages.shape[0]):
                    age_means_per_agent.append(np.mean(ages[agent]))
                    age_vars_per_agent.append(np.var(ages[agent]))
                    file_ids_per_agent.append(file_id)
                    agent_ids_per_agent.append(agent)
            import pandas as pd
            # age_meanファイル
            df_age_mean = pd.DataFrame({
                'file_id': file_ids_per_agent,
                'agent_id': agent_ids_per_agent,
                'age_mean': age_means_per_agent
            })
            csv_path_mean = os.path.join(save_dir, 'word_age_mean_per_agent.csv')
            df_age_mean.to_csv(csv_path_mean, index=False)
            print(f"Saved per-agent word age mean to {csv_path_mean}")
            # age_varファイル
            df_age_var = pd.DataFrame({
                'file_id': file_ids_per_agent,
                'agent_id': agent_ids_per_agent,
                'age_var': age_vars_per_agent
            })
            csv_path_var = os.path.join(save_dir, 'word_age_var_per_agent.csv')
            df_age_var.to_csv(csv_path_var, index=False)
            print(f"Saved per-agent word age var to {csv_path_var}")
            # --- 既存の全体平均の保存 ---
            age_means = []
            age_vars = []
            file_ids = []
            for state_file in tqdm(state_files):
                basename = os.path.basename(state_file)
                file_id = int(basename.split('_')[1].split('.')[0])
                file_ids.append(file_id)
                file_t = id2t[file_id]
                state = np.load(state_file)  # (agents_count, N_i, 3)
                word_ts = state[..., 0]      # (agents_count, N_i)
                ages = file_t - word_ts      # (agents_count, N_i)
                ages_flat = ages.flatten()
                age_means.append(np.mean(ages_flat))
                age_vars.append(np.var(ages_flat))
            df_age = pd.DataFrame({
                'file_id': file_ids,
                'age_mean': age_means,
                'age_var': age_vars
            })
            csv_path = os.path.join(save_dir, 'word_age_stats.csv')
            df_age.to_csv(csv_path, index=False)
            print(f"Saved word age stats to {csv_path}")
        age_files = np.loadtxt(os.path.join(save_dir, "word_age_stats.csv"), delimiter=',', skiprows=1, dtype=float)

    if not distance_files:
        print(f"No distance files found in {load_dir}. Skipping...")
        continue
    dot_sim_files = sorted(glob.glob(os.path.join(load_dir, "similarity_dot_*.npy")))
    cosine_sim_files = sorted(glob.glob(os.path.join(load_dir, "similarity_cosine_*.npy")))

    # Dot Product Similarity の平均を計算
    if dot_sim_files:
        print(f"Found {len(dot_sim_files)} dot similarity files. Averaging...")
        similarities_dot = np.stack([np.load(f) for f in dot_sim_files], axis=0)
        mean_similarity_dot = similarities_dot.mean(axis=0)
        print(f"Mean dot similarity: {mean_similarity_dot}")
    else:
        mean_similarity_dot = None # ファイルが見つからない場合はNoneに設定

    # Cosine Similarity の平均を計算
    if cosine_sim_files:
        print(f"Found {len(cosine_sim_files)} cosine similarity files. Averaging...")
        similarities_cosine = np.stack([np.load(f) for f in cosine_sim_files], axis=0)
        mean_similarity_cosine = similarities_cosine.mean(axis=0)
    else:
        mean_similarity_cosine = None # ファイルが見つからない場合はNoneに設定

    os.makedirs(save_dir, exist_ok=True)
    # --- データ読込・平均 ---
    distances = []
    total_positive = 0
    total = len(distance_files)
    
    # ロジスティック回帰用の変数
    x1 = 1 if na == 'center' else 0  # nonzero_alpha_options = center なら1
    x2 = 1 if ft == 'outward' else 0  # flow_type_options = outward なら1
    
    for f in distance_files:

        d = np.load(f)
        is_concentric = is_concentric_distribution(d)
        if is_concentric:
            # INSERT_YOUR_REWRITE_HERE
            # 対応するstateファイルをloadして
            # import pandas as pd

            # state_file = f.replace("distance_", "state_")
            # if os.path.exists(state_file):
            #     with open(state_file, "rb") as sf:
            #         state = np.load(sf)
            #     print(f"State file {state_file} loaded successfully.")
            #     # 必要に応じてstateを使った処理をここに追加
            # else:
            #     print(f"Warning: State file {state_file} not found.")
            # state_reshaped = state.reshape((-1, 3))  # (N_i, agent_num, 2)
            # # uniqueなものに通し番号つけて
            # unique_rows, inverse_indices = np.unique(state_reshaped, axis=0, return_inverse=True)
            # inverse_indices = inverse_indices.reshape(agents_count, N_i)
            # inverse_indices.sort(axis=1)
            # print(inverse_indices)  # (agent_num, N_i)

            # # CSVで保存
            # csv_path = os.path.join(save_dir, f"unique_state_indices_{os.path.splitext(os.path.basename(f))[0]}.csv")
            # df = pd.DataFrame(inverse_indices)
            # print(csv_path)
            # df.to_csv(csv_path, index=False)

            
            # unique_rows: (n_unique, 3)
            # inverse_indices: (N_i * agent_num,)
            # それぞれのstate_reshapedの行が何番目のuniqueかを示す
            # 例: print(list(zip(state_reshaped, inverse_indices)))
            # for idx, (row, uniq_idx) in enumerate(zip(state_reshaped, inverse_indices)):
            #     print(f"Row {idx}: {row} -> Unique ID: {uniq_idx}")

            total_positive += 1
        
        # ロジスティック回帰用データを収集
        logistic_data_concentric.append({
            'y': int(is_concentric),  # 同心円分布かどうか (0 or 1)
            'x1': x1,  # nonzero_alpha = center なら1
            'x2': x2,  # flow_type = outward なら1
            'x1_x2': x1 * x2,  # 交互作用項
            'nonzero_alpha': na,
            'flow_type': ft,
            'file': os.path.basename(f)
        })
        
        distances.append(d)
    distances = np.stack(distances, axis=0)  # (num_snapshots, agent_num, agent_num)
# ▼▼▼ このブロックに置き換え ▼▼▼
    # ==============================================================================
    # --- 全ての同心円ペアの頻度を計算 ---
    # ==============================================================================
    print(f"\n--- [{na}, {ft}] Calculating frequency of d(i, 7) > d(i, j) for all relevant pairs ---")

    center_agent = agents_count // 2
    pair_frequencies = []

    # baseエージェント `i` でループ (中心を除く)
    for i in range(agents_count):
        if i == center_agent:
            continue

        # referenceエージェント `j` でループ
        for j in range(agents_count):
            is_opposite_side = (i - center_agent) * (j - center_agent) < 0
            if is_opposite_side:
                # 全スナップショットにおける d(i, 7) と d(i, j) の距離データを抽出
                d_i_7 = distances[:, i, center_agent]
                d_i_j = distances[:, i, j]

                # d(i, 7) > d(i, j) となる頻度を計算 (True=1, False=0として平均をとる)
                frequency = (d_i_7 > d_i_j).mean()

                # 結果をリストに保存
                # 全スナップショットにおける d(i, 7) と d(i, j) の距離データを抽出
                d_i_7 = distances[:, i, center_agent]
                d_i_j = distances[:, i, j]

                # d(i, 7) > d(i, j) となる頻度を計算 (True=1, False=0として平均をとる)
                frequency = (d_i_7 > d_i_j).mean()

                # 結果をリストに保存
                pair_frequencies.append({
                    'base_agent_i': i,
                    'reference_agent_j': j,
                    'frequency': frequency
                })

    if pair_frequencies:
        # --- 1. リスト形式（縦長データ）での保存 ---
        df_pair_freq_long = pd.DataFrame(pair_frequencies)
        df_pair_freq_long_sorted = df_pair_freq_long.sort_values(by='frequency', ascending=False)

        # 結果の一部をコンソールに表示
        print("Top 10 most frequent pairs for d(i, 7) > d(i, j):")
        print(df_pair_freq_long_sorted.head(10).round(4))
        
        # リスト形式でCSVファイルに保存
        long_freq_csv_path = os.path.join(save_dir, 'concentric_pair_frequencies_long.csv')
    distances_0_7 = distances[:, 0, 7]  
    distances_0_10 = distances[:, 0, 10]  
    mean_distance = distances.mean(axis=0)   # (agent_num, agent_num)
    
    # 距離の大小関係をロジスティック回帰用データに追加
    for i, (d_0_7, d_0_10) in enumerate(zip(distances_0_7, distances_0_10)):
        logistic_data_distance.append({
            'y': int(d_0_7 > d_0_10),  # 0-7の距離 > 0-10の距離 なら1
            'x1': x1,  # nonzero_alpha = center なら1
            'x2': x2,  # flow_type = outward なら1
            'x1_x2': x1 * x2,  # 交互作用項
            'nonzero_alpha': na,
            'flow_type': ft,
            'distance_0_7': d_0_7,
            'distance_0_10': d_0_10,
            'snapshot': i
        })
    # INSERT_YOUR_CODE
    # --- ヒストグラム: agent 0-7, 0-10 の距離分布 ---

    agent_pairs = [(0, 7), (0, 10)]
    agent_pair_colors = ['blue', 'red']  # 各エージェントペアの色

    # データを収集
    distances_0_7 = distances[:, 0, 7]
    distances_0_10 = distances[:, 0, 10]
    # INSERT_YOUR_CODE
    import pandas as pd

    # Save distances_0_7 and distances_0_10 as CSV
    df_distances = pd.DataFrame({
        'distance_0_7': distances_0_7,
        'distance_0_10': distances_0_10
    })
    csv_path = os.path.join(save_dir, 'distances_0_7_0_10.csv')
    df_distances.to_csv(csv_path, index=False)
    print(f"Saved distances_0_7 and distances_0_10 to {csv_path}")


    # 2項検定の準備
    diff_0_7_0_10 = distances_0_7 - distances_0_10
    over_count_0_7 = np.sum(diff_0_7_0_10 > 0)
    under_count_0_7 = np.sum(diff_0_7_0_10 < 0)
    equal_count_0_7 = np.sum(diff_0_7_0_10 == 0)

    # =を含む場合（元のやり方）
    total = len(distances_0_7)

    # =を除外した場合
    total_exclude_equal = over_count_0_7 + under_count_0_7
    
    # --- 2項検定（=含む/除外 両方表示） ---
    # =を含む場合
    over_count_0_7 = np.sum(distances_0_7 > distances_0_10)
    under_count_0_7 = np.sum(distances_0_7 < distances_0_10)
    equal_count_0_7 = np.sum(distances_0_7 == distances_0_10)
    total = len(distances_0_7)
    total_exclude_equal = over_count_0_7 + under_count_0_7

    # =を含む場合
    p_value = stats.binomtest(over_count_0_7, n=total, p=0.5, alternative='two-sided').pvalue
    # =を除外した場合
    if total_exclude_equal > 0:
        p_value_exclude_equal = stats.binomtest(over_count_0_7, n=total_exclude_equal, p=0.5, alternative='two-sided').pvalue
    else:
        p_value_exclude_equal = np.nan

    # プロット
    for i, (agent1, agent2) in enumerate(agent_pairs):
        # 1マスずらすために、全体を+/-0.5シフト
        dists = distances[:, agent1, agent2]
        mean_dist = np.mean(dists)
        # binsの中心をずらす
        bin_edges = np.linspace(np.min(dists), np.max(dists), 202)
        shift = -0.5 if i == 0 else 0.5 
        plt.hist(dists + shift, bins=bin_edges + shift, label=f'Agent {agent1}-{agent2}', color=agent_pair_colors[i], density=True)
        plt.axvline(mean_dist + shift, color=agent_pair_colors[i], linestyle='--', 
                    label=f'Mean {agent1}-{agent2}: {mean_dist:.2f} (shifted)')
    # plt.xlabel('Distance')
    # plt.ylabel('Frequency')
    # plt.title(f'Distribution of Distances Between Agent Pairs\nBinomial test p-value: {p_value:.3e}')
    # plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(save_dir, 'agent_pair_distances_histogram.png'),dpi=300)
    plt.show()
    
    # --- ヒストグラム: 対数軸バージョン ---
    plt.figure(figsize=(5, 5))
    for i, (agent1, agent2) in enumerate(agent_pairs):
        dists = distances[:, agent1, agent2]
        mean_dist = np.mean(dists)
        bin_edges = np.linspace(np.min(dists), np.max(dists), 202)
        shift = -0.5 if i == 0 else 0.5
        plt.hist(dists + shift, bins=bin_edges + shift, label=f'Agent {agent1}-{agent2}', color=agent_pair_colors[i], density=True)
        plt.axvline(mean_dist + shift, color=agent_pair_colors[i], linestyle='--', 
                    label=f'Mean {agent1}-{agent2}: {mean_dist:.2f} (shifted)')
    # plt.xlabel('Distance')
    # plt.ylabel('Frequency (log scale)')
    # plt.title(f'Distribution of Distances di Agent Pairs (Log Y-axis)\nBinomial test p-value: {p_value:.3e}')
    plt.yscale('log')
    # plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    plt.savefig(os.path.join(save_dir, 'agent_pair_distances_histogram_log.png'),dpi=300)
    plt.show()
    # --- ヒストグラム: agent 0-7 と agent 0-10 の距離の「差」の分布 (スタイル調整版) ---
    plt.figure(figsize=(6, 5))

    # 距離の差を計算
    diff_distances = distances_0_7 - distances_0_10

    # ビン(階級)の数を他のプロットの2倍に設定
    # np.linspaceで402個の点を生成すると、階級(ビン)は401個になる
    bin_edges = np.linspace(np.min(diff_distances), np.max(diff_distances), 402)
    mean_diff = np.mean(diff_distances)

    # ヒストグラムを描画
    plt.axvline(mean_diff, color='blue', linestyle='--', linewidth=0.5, label=f'Mean Difference: {mean_diff:.2f}')

    # 差が0の基準線を引く (大小関係の境界として重要)
    # plt.axvline(0, color='red', linestyle='-', linewidth=0.5, label='Zero')
    plt.hist(diff_distances, bins=bin_edges, alpha=0.75, label='d(0, 7) - d(0, 10)', color='green', density=True)

    # 差の平均値の線を引く

    # グラフの体裁を整える
    # plt.title(f'Histogram of Distance Difference (d(0,7) - d(0,10))\n{ft} flow, {na}')
    plt.xlabel('Distance Difference (d(0,7) - d(0,10))')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 画像を保存して表示
    plt.savefig(os.path.join(save_dir, 'distance_difference_histogram.png'), dpi=300)
    plt.show()
    # --- 散布図: d(0,7) vs d(0,10) ---
    plt.figure(figsize=(6, 6))

    # 点が重なるため、alphaで透明度を指定
    plt.scatter(distances_0_7, distances_0_10, alpha=0.2, s=15, edgecolors='none')

    # y=x の基準線を追加（2つの距離が等しい場所を示す）
    # プロット範囲の最小値と最大値を取得して線を引く
    max_val = max(np.max(distances_0_7), np.max(distances_0_10))
    min_val = min(np.min(distances_0_7), np.min(distances_0_10))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='d(0,7) = d(0,10)')

    # グラフの体裁
    plt.xlabel('Distance d(0, 7)')
    plt.ylabel('Distance d(0, 10)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal') # 縦横のスケールを合わせる
    # plt.title(f'Scatter Plot of d(0,7) vs d(0,10)\n{ft} flow, {na}')
    plt.savefig(os.path.join(save_dir, 'distance_scatter_plot.png'), dpi=300)
    plt.show()

    # --- バブルチャート (同一座標の点を集計): d(0,7) vs d(0,10) ---
    # ご指摘の通り、データは離散値のため、全く同じ座標を持つ点の数を正確に集計し、バブルサイズに反映します。
    plt.figure(figsize=(7, 6))

    # pandas DataFrameを作成して、同一座標のペアを効率的に集計
    df_scatter = pd.DataFrame({
        'd_0_7': distances_0_7,
        'd_0_10': distances_0_10
    })

    # 同一の(d_0_7, d_0_10)ペアの出現回数をカウント
    bubble_data = df_scatter.groupby(['d_0_7', 'd_0_10']).size().reset_index(name='count')

    # バブルのサイズを頻度(count)に比例させる (見た目を調整するための係数)
    scale_factor = 5
    bubble_sizes = bubble_data['count'] * scale_factor

    # バブルチャートを描画
    plt.scatter(
        bubble_data['d_0_7'],
        bubble_data['d_0_10'],
        s=bubble_sizes,
        alpha=0.1,
        edgecolors="w", # バブルの境界線を白にすると見やすい
        linewidth=0.5
    )

    # y=x の基準線を追加 (zorderでプロットの背面になるように調整)
    max_val = max(np.max(distances_0_7), np.max(distances_0_10))
    min_val = min(np.min(distances_0_7), np.min(distances_0_10))
   

    # グラフの体裁
    plt.xlabel('Distance d(0, 7)')
    plt.ylabel('Distance d(0, 10)')
    plt.grid(True, alpha=0.3, zorder=-1) # gridも背面に
    plt.axis('equal')
    # plt.title(f'Bubble Chart of d(0,7) vs d(0,10) Point Counts\n{ft} flow, {na}')
    plt.savefig(os.path.join(save_dir, 'distance_bubble_exact_counts_plot.png'), dpi=300)

    # --- バブルチャート (同一座標の点を集計): d(0,7) vs d(0,10) ---
    # ご指摘の通り、データは離散値のため、全く同じ座標を持つ点の数を正確に集計し、バブルサイズに反映します。
    plt.figure(figsize=(7, 6))

    # pandas DataFrameを作成して、同一座標のペアを効率的に集計
    df_scatter = pd.DataFrame({
        'd_0_7': distances_0_7,
        'd_0_10': distances_0_10
    })

    # 同一の(d_0_7, d_0_10)ペアの出現回数をカウント
    bubble_data = df_scatter.groupby(['d_0_7', 'd_0_10']).size().reset_index(name='count')

    # バブルのサイズを頻度(count)に比例させる (見た目を調整するための係数)
    scale_factor = 5
    bubble_sizes = np.log(1+bubble_data['count'] * scale_factor)

    # バブルチャートを描画
    plt.scatter(
        bubble_data['d_0_7'],
        bubble_data['d_0_10'],
        s=bubble_sizes,
        alpha=0.8,
        edgecolors="w", # バブルの境界線を白にすると見やすい
        linewidth=0.5
    )

    # y=x の基準線を追加 (zorderでプロットの背面になるように調整)
    max_val = max(np.max(distances_0_7), np.max(distances_0_10))
    min_val = min(np.min(distances_0_7), np.min(distances_0_10))
   

    # グラフの体裁
    plt.xlabel('Distance d(0, 7)')
    plt.ylabel('Distance d(0, 10)')
    plt.grid(True, alpha=0.3, zorder=-1) # gridも背面に
    plt.axis('equal')
    # plt.title(f'Bubble Chart of d(0,7) vs d(0,10) Point Counts\n{ft} flow, {na}')
    plt.savefig(os.path.join(save_dir, 'distance_log_bubble_exact_counts_plot.png'), dpi=300)
    plt.show()

    # --- 2Dヒストグラム (Heatmap) ---
    # imshow を使った四角いセルのヒートマップを作成します。
    # まず、2次元のヒストグラムデータを計算 (binsで解像度を調整)
    bins = 50
    counts, xedges, yedges = np.histogram2d(distances_0_7, distances_0_10, bins=bins)

    # 軸の範囲を定義
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # --- バージョン1: 通常のリニアスケール ---
    plt.figure(figsize=(7, 6))
    # `imshow`でヒートマップを描画。counts.Tで転置し、origin='lower'で原点を左下に設定。
    im = plt.imshow(counts.T, origin='lower', extent=extent, cmap='viridis', aspect='equal')

    # カラーバーと基準線
    plt.colorbar(im, label='Counts')
    

    # グラフの体裁
    plt.xlabel('Distance d(0, 7)')
    plt.ylabel('Distance d(0, 10)')
    # plt.title('2D Histogram (Linear Scale)')
    plt.savefig(os.path.join(save_dir, 'distance_heatmap_linear.png'), dpi=300)
    plt.show()


    # --- バージョン2: 対数カラースケール ---
    # `matplotlib.colors`をインポート
    for cmap in ['Blues', 'Reds', 'Greens']:
        plt.figure(figsize=(7, 6))
        
        # 度数が0のセルはエラーになるため、マスクする
        counts_masked = np.ma.masked_where(counts == 0, counts)

        # LogNorm() で0を無視し、1以上の値を対数スケールで表現
        im_log = plt.imshow(
            counts_masked.T, origin='lower', extent=extent, cmap=cmap, aspect='equal',
            norm=colors.LogNorm()
        )

        # カラーバーと基準線
        plt.colorbar(im_log, label='Counts (log scale)')
        

        # グラフの体裁
        plt.xlabel('Distance d(0, 7)')
        plt.ylabel('Distance d(0, 10)')
        # plt.title('2D Histogram (Log Scale)')
        plt.savefig(os.path.join(save_dir, f'distance_heatmap_log_{cmap}.png'), dpi=300)
        plt.show()
    # --- ヒートマップ ---
    plt.figure(figsize=(5, 5))
    # plt.title(f"Mean Agent Distance (Heatmap)\n{ft} flow, {na}")
    # plt.xlabel("Agent")
    # plt.ylabel("Agent")
    plt.imshow(mean_distance, aspect="equal", cmap='bwr')
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, "mean_distance_heatmap_bwr.png"),dpi=300)
    plt.show()
        
    # --- agent 0 から見た距離（棒グラフ） ---
    plt.figure(figsize=(5, 5))
    plt.plot(np.arange(mean_distance.shape[0]), mean_distance[agent_id], marker='o')
    plt.xticks([])
    # plt.xlabel("Other Agent")
    # plt.ylabel("Mean Distance")
    # plt.title(f"Mean Distance from Agent {agent_id}\n{ft} flow, {na}")
    plt.savefig(os.path.join(save_dir, f"mean_distance_from_agent{agent_id}.png"),dpi=300)


    if args.plot_age:
        # --- word age mean and sd plot ---
        age_mean_file = os.path.join(save_dir, 'word_age_mean_per_agent.csv')
        plt.figure(figsize=(5, 5))
        # 追加: 各エージェントのage_meanの時間平均を折れ線グラフで描画
        df = pd.read_csv(age_mean_file)
        mean_by_agent = df.groupby('agent_id')['age_mean'].mean()
        std_by_agent = df.groupby('agent_id')['age_mean'].std()
        # 平均のみのグラフ
        plt.xticks([])
        plt.plot(mean_by_agent.index, mean_by_agent.values, marker='o')
        # plt.xlabel('Agent')
        # plt.ylabel('Time-averaged age_mean')
        # plt.title('Time-averaged word age mean per agen¨t')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'age_mean_timeavg_per_agent.png'),dpi=300)
        plt.show()
        # 標準偏差付きのグラフ
        plt.figure(figsize=(5, 5))
        plt.plot(mean_by_agent.index, mean_by_agent.values, marker='o', label='Mean')
        plt.fill_between(mean_by_agent.index,
                         mean_by_agent.values - std_by_agent.values,
                         mean_by_agent.values + std_by_agent.values,
                         color='blue', alpha=0.2, label='Std')
        # plt.xlabel('Agent')
        # plt.ylabel('Time-averaged age_mean')
        # plt.title('Time-averaged word age mean per agent (with std)')
        plt.xticks([])
        plt.grid(True, alpha=0.3)
        # plt.legend()
        plt.savefig(os.path.join(save_dir, 'age_mean_timeavg_per_agent_with_std.png'),dpi=300)
        plt.show()
    
    # --- Dot Product Similarity ヒートマップ ---
    if mean_similarity_dot is not None:
        # --- Dot Product Similarity の2項検定を追加 ---
        similarities_dot_0_7 = similarities_dot[:, 0, 7]
        similarities_dot_0_10 = similarities_dot[:, 0, 10]
        
        # ドット積類似度の大小関係をロジスティック回帰用データに追加
        for i, (s_0_7, s_0_10) in enumerate(zip(similarities_dot_0_7, similarities_dot_0_10)):
            logistic_data_dot_sim.append({
                'y': int(s_0_7 > s_0_10),  # 0-7の類似度 > 0-10の類似度 なら1
                'x1': x1,  # nonzero_alpha = center なら1
                'x2': x2,  # flow_type = outward なら1
                'x1_x2': x1 * x2,  # 交互作用項
                'nonzero_alpha': na,
                'flow_type': ft,
                'similarity_0_7': s_0_7,
                'similarity_0_10': s_0_10,
                'snapshot': i
            })
        
        # Save dot similarities as CSV
        df_dot_similarities = pd.DataFrame({
            'dot_similarity_0_7': similarities_dot_0_7,
            'dot_similarity_0_10': similarities_dot_0_10
        })
        csv_path_dot = os.path.join(save_dir, 'dot_similarities_0_7_0_10.csv')
        df_dot_similarities.to_csv(csv_path_dot, index=False)
        print(f"Saved dot similarities_0_7 and similarities_0_10 to {csv_path_dot}")
        
        # 2項検定の準備（類似度では0-7 > 0-10が期待される）
        over_count_dot_0_7 = np.sum(similarities_dot_0_7 > similarities_dot_0_10)
        under_count_dot_0_7 = np.sum(similarities_dot_0_7 < similarities_dot_0_10)
        equal_count_dot_0_7 = np.sum(similarities_dot_0_7 == similarities_dot_0_10)
        total_dot = len(similarities_dot_0_7)
        total_exclude_equal_dot = over_count_dot_0_7 + under_count_dot_0_7

        # =を含む場合
        p_value_dot = stats.binomtest(over_count_dot_0_7, n=total_dot, p=0.5, alternative='two-sided').pvalue
        # =を除外した場合
        if total_exclude_equal_dot > 0:
            p_value_exclude_equal_dot = stats.binomtest(over_count_dot_0_7, n=total_exclude_equal_dot, p=0.5, alternative='two-sided').pvalue
        else:
            p_value_exclude_equal_dot = np.nan

        # --- Dot Product Similarity ヒストグラム ---
        plt.figure(figsize=(5, 5))
        for i, (agent1, agent2) in enumerate(agent_pairs):
            sims = similarities_dot[:, agent1, agent2]
            mean_sim = np.mean(sims)
            bin_edges = np.linspace(np.min(sims), np.max(sims), 202)
            shift = -0.005 if i == 0 else 0.005  # 類似度は距離より小さい範囲なので小さいshift
            plt.hist(sims + shift, bins=bin_edges + shift, label=f'Agent {agent1}-{agent2}', color=agent_pair_colors[i], density=True)
            plt.axvline(mean_sim + shift, color=agent_pair_colors[i], linestyle='--', 
                        label=f'Mean {agent1}-{agent2}: {mean_sim:.3f} (shifted)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'agent_pair_dot_similarities_histogram.png'),dpi=300)
        plt.show()

        plt.figure(figsize=(5, 5))
        # 'viridis' は非負の値の可視化に適したカラーマップです
        im = plt.imshow(mean_similarity_dot, aspect="equal", cmap='viridis')
        plt.colorbar(im, label="Mean Dot Product")
        # plt.title(f"Mean Dot Product Similarity\n{ft} flow, {na}")
        # plt.xlabel("Agent")
        # plt.ylabel("Agent")
        plt.xticks([])  # x軸の目盛りを非表示
        plt.yticks([])  # y軸の目盛りを非表示
        plt.savefig(os.path.join(save_dir, "mean_dot_similarity_heatmap.png"), dpi=300)
        plt.show()
        print(f"mean dot similarity is concentric: {is_concentric_distribution(-mean_similarity_dot)}")
        # --- agent 0 から見たドット積類似度（棒グラフ） ---
        plt.figure(figsize=(5, 5))
        plt.plot(np.arange(mean_similarity_dot.shape[0]), mean_similarity_dot[0], marker='o')
        plt.xticks([])
        # plt.xlabel("Other Agent")
        # plt.ylabel("Mean Dot Product Similarity")
        # plt.title(f"Mean Dot Product Similarity from Agent 0\n{ft} flow, {na}")
        plt.savefig(os.path.join(save_dir, f"mean_dot_similarity_from_agent0.png"), dpi=300)
        plt.show()

    # --- Cosine Similarity ヒートマップ ---
    if mean_similarity_cosine is not None:
        # --- Cosine Similarity の2項検定を追加 ---
        similarities_cosine_0_7 = similarities_cosine[:, 0, 7]
        similarities_cosine_0_10 = similarities_cosine[:, 0, 10]
        
        # コサイン類似度の大小関係をロジスティック回帰用データに追加
        for i, (s_0_7, s_0_10) in enumerate(zip(similarities_cosine_0_7, similarities_cosine_0_10)):
            logistic_data_cosine_sim.append({
                'y': int(s_0_7 > s_0_10),  # 0-7の類似度 > 0-10の類似度 なら1
                'x1': x1,  # nonzero_alpha = center なら1
                'x2': x2,  # flow_type = outward なら1
                'x1_x2': x1 * x2,  # 交互作用項
                'nonzero_alpha': na,
                'flow_type': ft,
                'similarity_0_7': s_0_7,
                'similarity_0_10': s_0_10,
                'snapshot': i
            })
        
        # Save cosine similarities as CSV
        df_cosine_similarities = pd.DataFrame({
            'cosine_similarity_0_7': similarities_cosine_0_7,
            'cosine_similarity_0_10': similarities_cosine_0_10
        })
        csv_path_cosine = os.path.join(save_dir, 'cosine_similarities_0_7_0_10.csv')
        df_cosine_similarities.to_csv(csv_path_cosine, index=False)
        print(f"Saved cosine similarities_0_7 and similarities_0_10 to {csv_path_cosine}")
        
        # 2項検定の準備（類似度では0-7 > 0-10が期待される）
        over_count_cosine_0_7 = np.sum(similarities_cosine_0_7 > similarities_cosine_0_10)
        under_count_cosine_0_7 = np.sum(similarities_cosine_0_7 < similarities_cosine_0_10)
        equal_count_cosine_0_7 = np.sum(similarities_cosine_0_7 == similarities_cosine_0_10)
        total_cosine = len(similarities_cosine_0_7)
        total_exclude_equal_cosine = over_count_cosine_0_7 + under_count_cosine_0_7

        # =を含む場合
        p_value_cosine = stats.binomtest(over_count_cosine_0_7, n=total_cosine, p=0.5, alternative='two-sided').pvalue
        # =を除外した場合
        if total_exclude_equal_cosine > 0:
            p_value_exclude_equal_cosine = stats.binomtest(over_count_cosine_0_7, n=total_exclude_equal_cosine, p=0.5, alternative='two-sided').pvalue
        else:
            p_value_exclude_equal_cosine = np.nan

        # --- Cosine Similarity ヒストグラム ---
        plt.figure(figsize=(5, 5))
        for i, (agent1, agent2) in enumerate(agent_pairs):
            sims = similarities_cosine[:, agent1, agent2]
            mean_sim = np.mean(sims)
            bin_edges = np.linspace(np.min(sims), np.max(sims), 202)
            shift = -0.001 if i == 0 else 0.001  # コサイン類似度は0-1範囲なので更に小さいshift
            plt.hist(sims + shift, bins=bin_edges + shift, label=f'Agent {agent1}-{agent2}', color=agent_pair_colors[i], density=True)
            plt.axvline(mean_sim + shift, color=agent_pair_colors[i], linestyle='--', 
                        label=f'Mean {agent1}-{agent2}: {mean_sim:.3f} (shifted)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'agent_pair_cosine_similarities_histogram.png'),dpi=300)
        plt.show()

        plt.figure(figsize=(5, 5))
        # コサイン類似度は通常0から1の範囲なので、vmax=1でスケールを固定すると比較しやすくなります
        im = plt.imshow(mean_similarity_cosine, vmin=0, vmax=1, aspect="equal", cmap='viridis')
        plt.colorbar(im, label="Mean Cosine Similarity")
        # plt.title(f"Mean Cosine Similarity\n{ft} flow, {na}")
        # plt.xlabel("Agent")
        # plt.ylabel("Agent")
        plt.xticks([])  # x軸の目盛りを非表示
        plt.yticks([])  # y軸の目盛りを非表示
        plt.savefig(os.path.join(save_dir, "mean_cosine_similarity_heatmap.png"), dpi=300)
        plt.show()
        print(f"mean cosine similarity is concentric: {is_concentric_distribution(-mean_similarity_cosine)}")
        # --- agent 0 から見たコサイン類似度（棒グラフ） ---
        plt.figure(figsize=(5, 5))
        plt.plot(np.arange(mean_similarity_cosine.shape[0]), mean_similarity_cosine[0], marker='o')
        plt.xticks([])
        # plt.xlabel("Other Agent")
        # plt.ylabel("Mean Cosine Similarity")
        # plt.title(f"Mean Cosine Similarity from Agent 0\n{ft} flow, {na}")
        plt.savefig(os.path.join(save_dir, f"mean_cosine_similarity_from_agent0.png"), dpi=300)
        plt.show()
    
    if args.check_concentric:
        print("=== Distance Binomial test results (including equal distances) ===")
        print(f"Number of times 0-7 > 0-10: {over_count_0_7}/{total}")
        print(f"Number of times 0-7 < 0-10: {under_count_0_7}/{total}")
        print(f"Number of times 0-7 = 0-10: {equal_count_0_7}/{total}")
        print(f"p-value (including equal distances): {p_value:.3e}")
        print(f"mean distance is concentric: {is_concentric_distribution(mean_distance)}")
        print()
        print("=== Distance Binomial test results (excluding equal distances) ===")
        print(f"Number of times 0-7 > 0-10: {over_count_0_7}/{total_exclude_equal}")
        print(f"Number of times 0-7 < 0-10: {under_count_0_7}/{total_exclude_equal}")
        print(f"Number of equal distances: {equal_count_0_7}/{total}")
        print(f"p-value (excluding equal distances): {p_value_exclude_equal:.3e}")
        
        # --- Dot Product Similarity の2項検定結果 ---
        if mean_similarity_dot is not None:
            print()
            print("=== Dot Product Similarity Binomial test results (including equal similarities) ===")
            print(f"Number of times 0-7 > 0-10: {over_count_dot_0_7}/{total_dot}")
            print(f"Number of times 0-7 < 0-10: {under_count_dot_0_7}/{total_dot}")
            print(f"Number of times 0-7 = 0-10: {equal_count_dot_0_7}/{total_dot}")
            print(f"p-value (including equal similarities): {p_value_dot:.3e}")
            print(f"mean dot similarity is concentric: {is_concentric_distribution(-mean_similarity_dot)}")
            print()
            print("=== Dot Product Similarity Binomial test results (excluding equal similarities) ===")
            print(f"Number of times 0-7 > 0-10: {over_count_dot_0_7}/{total_exclude_equal_dot}")
            print(f"Number of times 0-7 < 0-10: {under_count_dot_0_7}/{total_exclude_equal_dot}")
            print(f"Number of equal similarities: {equal_count_dot_0_7}/{total_dot}")
            print(f"p-value (excluding equal similarities): {p_value_exclude_equal_dot:.3e}")
        
        # --- Cosine Similarity の2項検定結果 ---
        if mean_similarity_cosine is not None:
            print()
            print("=== Cosine Similarity Binomial test results (including equal similarities) ===")
            print(f"Number of times 0-7 > 0-10: {over_count_cosine_0_7}/{total_cosine}")
            print(f"Number of times 0-7 < 0-10: {under_count_cosine_0_7}/{total_cosine}")
            print(f"Number of times 0-7 = 0-10: {equal_count_cosine_0_7}/{total_cosine}")
            print(f"p-value (including equal similarities): {p_value_cosine:.3e}")
            print(f"mean cosine similarity is concentric: {is_concentric_distribution(-mean_similarity_cosine)}")
            print()
            print("=== Cosine Similarity Binomial test results (excluding equal similarities) ===")
            print(f"Number of times 0-7 > 0-10: {over_count_cosine_0_7}/{total_exclude_equal_cosine}")
            print(f"Number of times 0-7 < 0-10: {under_count_cosine_0_7}/{total_exclude_equal_cosine}")
            print(f"Number of equal similarities: {equal_count_cosine_0_7}/{total_cosine}")
            print(f"p-value (excluding equal similarities): {p_value_exclude_equal_cosine:.3e}")

# 全ての組み合わせが処理された後、ロジスティック回帰分析を実行
if len(logistic_data_concentric) > 0:
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION ANALYSIS (Concentric Distribution)")
    print("="*60)
    
    df_logistic = pd.DataFrame(logistic_data_concentric)
    print(f"Total observations: {len(df_logistic)}")
    print(f"Concentric distributions: {df_logistic['y'].sum()} ({df_logistic['y'].mean()*100:.1f}%)")
    print("\nConcentric distribution rates by condition:")
    condition_stats = df_logistic.groupby(['nonzero_alpha', 'flow_type']).agg(
        {'y': ['count', 'sum', 'mean']}
    ).round(3)
    condition_stats.columns = ['count', 'concentric_count', 'concentric_rate']
    print(condition_stats)
    print()
    
    unique_conditions = df_logistic[['x1', 'x2']].drop_duplicates()
    if len(unique_conditions) > 1:
        # 説明変数と目的変数を準備
        X = df_logistic[['x1', 'x2', 'x1_x2']].values
        y = df_logistic['y'].values
        
        # --- statsmodelsによる分析 ---
        print("\n--- Logistic Regression Results (statsmodels) ---")
        # statsmodelsは切片項を自動で追加しないため、定数項の列を追加
        X_sm = sm.add_constant(X, prepend=True)
        
        try:
            # モデルを学習
            logit_model = sm.Logit(y, X_sm).fit(disp=0) # disp=0で収束メッセージを非表示に
            
            # .summary()で係数、標準誤差、p値、95%信頼区間などをまとめて表示
            print(logit_model.summary(
                xname=['Intercept', 'x1_center', 'x2_outward', 'x1_x2_interaction']
            ))
            
            # --- モデルの評価 ---
            print("\n--- Model Performance ---")
            # 予測確率を計算
            y_pred_proba_sm = logit_model.predict(X_sm)
            # 確率が0.5より大きい場合に1、それ以外は0とする
            y_pred_sm = (y_pred_proba_sm > 0.5).astype(int)
            
            # 正解率
            accuracy = (y_pred_sm == y).mean()
            print(f"Accuracy: {accuracy:.3f}")
            
            # 分類レポート
            print("\nClassification Report:")
            print(classification_report(y, y_pred_sm, zero_division=0))

            # --- 各条件での予測確率 ---
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
            # statsmodels用に定数項を追加
            conditions_sm = sm.add_constant(np.array(conditions), prepend=True)
            # 各条件での確率を予測
            predicted_probs = logit_model.predict(conditions_sm)

            for name, prob in zip(condition_names, predicted_probs):
                print(f"{name}: {prob:.3f}")

        except Exception as e:
            print(f"Could not fit statsmodels Logit model. Error: {e}")
            print("This might be due to perfect separation in the data.")
        
    else:
        print("All observations have the same condition. Cannot perform logistic regression.")

# ==============================================================================
# --- 0-7 vs 0-10 距離比較のロジスティック回帰分析 ---
# ==============================================================================
if len(logistic_data_distance) > 0:
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION ANALYSIS FOR DISTANCE COMPARISON (d(0,7) > d(0,10))")
    print("="*60)

    df_dist_comp = pd.DataFrame(logistic_data_distance)
    print(f"Total observations: {len(df_dist_comp)}")
    print(f"Instances where d(0,7) > d(0,10): {df_dist_comp['y'].sum()} ({df_dist_comp['y'].mean()*100:.1f}%)")
    print("\nRates of d(0,7) > d(0,10) by condition:")
    condition_stats_dist = df_dist_comp.groupby(['nonzero_alpha', 'flow_type']).agg(
        {'y': ['count', 'sum', 'mean']}
    ).round(3)
    condition_stats_dist.columns = ['count', 'positive_count', 'positive_rate']
    print(condition_stats_dist)
    print()

    unique_conditions_dist = df_dist_comp[['x1', 'x2']].drop_duplicates()
    if len(unique_conditions_dist) > 1:
        # 説明変数と目的変数を準備
        X_dist = df_dist_comp[['x1', 'x2', 'x1_x2']].values
        y_dist = df_dist_comp['y'].values

        # --- statsmodelsによる分析 ---
        print("\n--- Logistic Regression Results (statsmodels) ---")
        # statsmodelsは切片項を自動で追加しないため、定数項の列を追加
        X_dist_sm = sm.add_constant(X_dist, prepend=True)

        try:
            # モデルを学習
            logit_model_dist = sm.Logit(y_dist, X_dist_sm).fit(disp=0)

            # .summary()で係数、標準誤差、p値、95%信頼区間などをまとめて表示
            print(logit_model_dist.summary(
                xname=['Intercept', 'x1_center', 'x2_outward', 'x1_x2_interaction']
            ))

            # --- モデルの評価 ---
            print("\n--- Model Performance ---")
            # 予測確率を計算
            y_pred_proba_dist_sm = logit_model_dist.predict(X_dist_sm)
            # 確率が0.5より大きい場合に1、それ以外は0とする
            y_pred_dist_sm = (y_pred_proba_dist_sm > 0.5).astype(int)

            # 正解率
            accuracy_dist = (y_pred_dist_sm == y_dist).mean()
            print(f"Accuracy: {accuracy_dist:.3f}")

            # 分類レポート
            print("\nClassification Report:")
            print(classification_report(y_dist, y_pred_dist_sm, zero_division=0))

            # --- 各条件での予測確率 ---
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
            # statsmodels用に定数項を追加
            conditions_sm = sm.add_constant(np.array(conditions), prepend=True)
            # 各条件での確率を予測
            predicted_probs_dist = logit_model_dist.predict(conditions_sm)
            
            for name, prob in zip(condition_names, predicted_probs_dist):
                print(f"{name}: {prob:.3f}")

        except Exception as e:
            print(f"Could not fit statsmodels Logit model for distance comparison. Error: {e}")
            print("This might be due to perfect separation in the data.")

    else:
        print("All observations have the same condition. Cannot perform logistic regression for distance comparison.")