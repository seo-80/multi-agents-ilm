import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import itertools
import pandas as pd

# 'ilm' はカスタムモジュールと想定
from ilm import networks

# --- 関数定義 (変更なし) ---

def is_concentric_distribution(distance_matrix):
    """
    距離行列が同心円状の分布を示しているかチェックする。
    分布が同心円状である基点が少なくとも1つ存在する場合にTrueを返す。
    """
    center = len(distance_matrix) // 2
    
    for base in range(len(distance_matrix)):
        if base == center:
            continue
        # baseよりも中心から遠いが、言語的にはbaseに近い参照点が存在するかチェック
        for reference in range(len(distance_matrix)):
            if abs(reference - base) > abs(center - base) and distance_matrix[base][reference] < distance_matrix[base][center]:
                return True
    return False


def update_f(f, W, mu, N):
    """
    f: (m, m) array, f[i, j] = f_{ij}
    W: (m, m) migration matrix, W[i, k]
    mu: (m,) mutation rates for each colony, mu[i]
    N: (m,) population size for each colony, N[i]
    Returns:
        f_new: (m, m) array, updated f_{ij}
    """
    m = len(mu)
    f_new = np.zeros_like(f)
    for i in range(m):
        for j in range(m):
            # Term 1: sum_{k,l} W_{ik} W_{jl} f_{kl}
            term1 = 0.0
            for k in range(m):
                for l in range(m):
                    term1 += W[i, k] * W[j, l] * f[k, l]
            # Term 2: sum_k W_{ik} W_{jk} (1 - f_{kk})/(2N_k)
            term2 = 0.0
            for k in range(m):
                term2 += W[i, k] * W[j, k] * (1 - f[k, k]) / (2 * N[k])
            # Update
            f_new[i, j] = (1 - mu[i]) * (1 - mu[j]) * (term1 + term2)
    return f_new


def run_until_convergence(f_init, W, mu, N, tol=1e-8, max_iter=10000):
    f = f_init.copy()
    for i in range(max_iter):
        f_new = update_f(f, W, mu, N)
        diff = np.max(np.abs(f_new - f))
        if diff < tol:
            return f_new
        f = f_new
    print(f"Warning: Did not converge within {max_iter} iterations. Final diff={diff}")
    return f

# --- メイン処理 ---

def main():
    # ==================================================================
    # === パラメータ設定エリア ===
    # ==================================================================
    N_i_list = [100, 200, 500]
    coupling_strength_list = [0.01, 0.001, 0.1]
    alpha_per_data_list = [0.001, 0.01]
    nonzero_alpha_list = ["evenly", "center"]
    network_types = {
        "bidirectional_flow_rate": "bidirectional",
        "outward_flow_rate": "outward"
    }
    agents_count = 15
    agent_id_to_plot = 0
    base_save_dir = "data/probability_of_identity"
    summary_filename = "is_concentric_summary.csv"
    
    # --- パラメータの全組み合わせを生成 ---
    param_combinations = list(itertools.product(
        N_i_list, coupling_strength_list, alpha_per_data_list,
        nonzero_alpha_list, network_types.keys()
    ))
    
    all_results = []
    print(f"Total simulations to run: {len(param_combinations)}")

    # --- 各パラメータの組み合わせでループ ---
    for params in tqdm(param_combinations, desc="Running Simulations"):
        N_i, coupling_strength, alpha_per_data, nonzero_alpha, network_key = params
        
        # --- 変数設定 ---
        alpha = alpha_per_data * N_i
        N = np.ones(agents_count) * N_i
        network_args = {network_key: coupling_strength}
        network_matrix = networks.network(agents_count, args=network_args)

        if nonzero_alpha == "evenly":
            alphas = np.ones(agents_count) * alpha
        elif nonzero_alpha == "center":
            alphas = np.zeros(agents_count)
            alphas[agents_count // 2] = alpha
        
        mu = alphas / (N_i + alphas)

        # ==================================================================
        # === 保存ディレクトリ設定 (変更箇所) ===
        # ==================================================================
        # パラメータ設定ごとに一意なサブディレクトリ名を生成
        param_subdir_name = f"{network_types[network_key]}_"
        param_subdir_name += f"nonzero_{nonzero_alpha}_"
        param_subdir_name += f"Ni_{N_i}_coupling_{coupling_strength}_alpha_{alpha_per_data}"
        
        # データ種別ごとにパスを構築
        raw_dir = os.path.join(base_save_dir, "raw", param_subdir_name)
        fig_dir = os.path.join(base_save_dir, "fig", param_subdir_name)
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(fig_dir, exist_ok=True)
        
        # --- 計算実行 ---
        raw_save_path = os.path.join(raw_dir, "probability_of_identity.npy")
        if os.path.exists(raw_save_path):
            f_final = np.load(raw_save_path)
        else:
            f_init = np.eye(agents_count)
            W = network_matrix
            f_final = run_until_convergence(f_init, W, mu, N)
            np.save(raw_save_path, f_final)

        # --- is_concentricの判定と結果保存 ---
        distance_matrix = 1 / (f_final + 1e-10)
        is_concentric = is_concentric_distribution(distance_matrix)
        all_results.append({
            'N_i': N_i, 'coupling_strength': coupling_strength, 'alpha_per_data': alpha_per_data,
            'nonzero_alpha': nonzero_alpha, 'network_type': network_types[network_key],
            'is_concentric': is_concentric
        })

        # ==================================================================
        # === グラフ描画 (変更箇所) ===
        # ==================================================================
        plt.figure(figsize=(8, 6))
        plt.imshow(f_final, cmap="viridis", interpolation="nearest")
        plt.colorbar(label="Probability of Identity")
        plt.title(f"f (Ni={N_i}, coupling={coupling_strength}, alpha_per_data={alpha_per_data})")
        plt.xlabel("Agent j")
        plt.ylabel("Agent i")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "probability_of_identity_heatmap.png")) # 保存先を fig_dir に変更
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.bar(np.arange(f_final.shape[0]), f_final[agent_id_to_plot])
        plt.title(f"Probability of Identity from Agent {agent_id_to_plot}")
        plt.xlabel("Other Agent")
        plt.ylabel("Probability of Identity")
        plt.savefig(os.path.join(fig_dir, f"probability_of_identity_from_agent{agent_id_to_plot}.png")) # 保存先を fig_dir に変更
        plt.close()

    # --- 結果の集計とCSV保存 (変更なし) ---
    if not all_results:
        print("No simulations were run. Exiting without creating summary file.")
        return

    results_df = pd.DataFrame(all_results)
    pivot_df = results_df.pivot_table(
        index=['N_i', 'coupling_strength', 'alpha_per_data'],
        columns=['nonzero_alpha', 'network_type'],
        values='is_concentric',
        aggfunc='first'
    )
    pivot_df.columns = ['_'.join(col) for col in pivot_df.columns.values]
    pivot_df.to_csv(summary_filename)
    
    print("\n" + "="*50)
    print("All simulations completed.")
    print(f"Summary table saved to: {summary_filename}")
    print("="*50)
    print("Summary Table:")
    print(pivot_df)

if __name__ == "__main__":
    main()