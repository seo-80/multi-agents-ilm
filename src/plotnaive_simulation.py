import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import argparse
import scipy.stats as stats


parser = argparse.ArgumentParser()
# 省略形も受け付けるようにする
parser.add_argument('--nonzero_alpha', '-a', type=str, default='all', choices=['evenly', 'center', 'all'], help='nonzero_alpha: "evenly", "center", or "all"')
parser.add_argument('--flow_type', '-f', type=str, default='all', choices=['bidirectional', 'outward', 'all'], help='flow_type: "bidirectional", "outward", or "all"')
parser.add_argument('--skip', '-s', type=int, default=0, help='何ファイルスキップするか (default: 0)')
parser.add_argument('--agent_id', '-i', type=int, default=0, help='棒グラフで表示するエージェントID (default: 0)')
parser.add_argument('--agents_count', '-m', type=int, default=15, help='Number of agents (default: 15)')
parser.add_argument('--N_i', '-n', type=int, default=100, help='Number of data per subpopulation (default: 100)')
args, unknown = parser.parse_known_args()

nonzero_alpha = args.nonzero_alpha
flow_type = args.flow_type
skip = args.skip
agent_id = args.agent_id
agents_count = args.agents_count
N_i = args.N_i


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
        # Check if there exists a reference point that is further from base than center
        # but linguistically closer to base than center
        for reference in range(len(distance_matrix)):
            if abs(reference - base) > abs(center - base) and distance_matrix[base][reference] < distance_matrix[base][center]:
                return True
 
 
combinations = get_combinations(nonzero_alpha, flow_type)

for na, ft in combinations:
    subdir = f"{ft}_flow-nonzero_alpha_{na}_agents_{agents_count}_N_i_{N_i}"
    load_dir = f"data/naive_simulation/raw/{subdir}"
    save_dir = f"data/naive_simulation/fig/{subdir}"
    distance_files = sorted(glob.glob(os.path.join(load_dir, "distance_*.npy")))
    print(f"[{ft}, {na}] Number of distance files found: {len(distance_files)}")
    distance_files = distance_files[skip:]  # スキップ数を適用

    if not distance_files:
        print(f"No distance files found in {load_dir}. Skipping...")
        continue

    os.makedirs(save_dir, exist_ok=True)
    # --- データ読込・平均 ---
    distances = []
    total_positive = 0
    total = len(distance_files)
    
    for f in distance_files:
        d = np.load(f)
        if is_concentric_distribution(d):
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
        distances.append(d)
    distances = np.stack(distances, axis=0)  # (num_snapshots, agent_num, agent_num)
    distances_0_7 = distances[:, 0, 7]  
    distances_0_10 = distances[:, 0, 10]  
    mean_distance = distances.mean(axis=0)   # (agent_num, agent_num)
    # INSERT_YOUR_CODE
    # --- ヒストグラム: agent 0-7, 0-10 の距離分布 ---

    agent_pairs = [(0, 7), (0, 10)]
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'red']  # 各エージェントペアの色

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
        dists = distances[:, agent1, agent2]
        mean_dist = np.mean(dists)
        plt.hist(dists, bins=201, alpha=0.5, label=f'Agent {agent1}-{agent2}', color=colors[i])
        plt.axvline(mean_dist, color=colors[i], linestyle='--', 
                    label=f'Mean {agent1}-{agent2}: {mean_dist:.2f}')

    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Distances Between Agent Pairs\nBinomial test p-value: {p_value:.3e}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(save_dir, 'agent_pair_distances_histogram.png'))
    plt.show()

    # --- ヒートマップ ---
    plt.figure(figsize=(8, 6))
    plt.title(f"Mean Agent Distance (Heatmap)\n{ft} flow, {na}")
    plt.xlabel("Agent")
    plt.ylabel("Agent")
    plt.imshow(mean_distance, aspect="auto")
    plt.colorbar(label="Mean Distance")
    plt.savefig(os.path.join(save_dir, "mean_distance_heatmap.png"))
    plt.show()
        
    # --- agent 0 から見た距離（棒グラフ） ---
    plt.figure(figsize=(8, 4))
    plt.title(f"Mean Distance from Agent {agent_id}\n{ft} flow, {na}")
    plt.bar(np.arange(mean_distance.shape[0]), mean_distance[agent_id])
    plt.xlabel("Other Agent")
    plt.ylabel("Mean Distance")
    plt.savefig(os.path.join(save_dir, f"mean_distance_from_agent{agent_id}.png"))


    print("=== Binomial test results (including equal distances) ===")
    print(f"Number of times 0-7 > 0-10: {over_count_0_7}/{total}")
    print(f"Number of times 0-7 < 0-10: {under_count_0_7}/{total}")
    print(f"Number of times 0-7 = 0-10: {equal_count_0_7}/{total}")
    print(f"p-value (including equal distances): {p_value:.3e}")
    print(f"mean distance is concentric: {is_concentric_distribution(mean_distance)}")
    print()
    print("=== Binomial test results (excluding equal distances) ===")
    print(f"Number of times 0-7 > 0-10: {over_count_0_7}/{total_exclude_equal}")
    print(f"Number of times 0-7 < 0-10: {under_count_0_7}/{total_exclude_equal}")
    print(f"Number of equal distances: {equal_count_0_7}/{total}")
    print(f"p-value (excluding equal distances): {p_value_exclude_equal:.3e}")