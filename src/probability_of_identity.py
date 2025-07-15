
import numpy as np
from tqdm import tqdm

from ilm import networks

import os
import matplotlib.pyplot as plt



agents_count = 15  # M
N_i = 100         # 各サブ集団のデータ数
alpha_per_data = 0.001  # 新語生成バイアス
nonzero_alpha = "center"  # "evenly" or "center"
nonzero_alpha = "evenly"  # "evenly" or "center".
coupling_strength = 0.01  # m
save_dir = "data/probability_of_identity"  # 保存先ディレクトリ
os.makedirs(save_dir, exist_ok=True)
agent_id = 0  # 棒グラフで表示するエージェントID

alpha = alpha_per_data * N_i  # 各エージェントのバイアス合計

N = np.ones(agents_count) * N_i  # 各エージェントのデータ数

# --- ネットワーク生成 ---
network_args = {"bidirectional_flow_rate": coupling_strength}
# network_args = {"outward_flow_rate": coupling_strength}
network_matrix = networks.network(agents_count, args=network_args)

if nonzero_alpha == "evenly":
    # 各エージェントに均等にバイアスを設定
    alphas = np.ones(agents_count) * alpha
elif nonzero_alpha == "center":
    # 中央のエージェントだけにバイアスを設定
    alphas = np.zeros(agents_count)
    alphas[agents_count // 2] = alpha
else:
    raise ValueError("nonzero_alpha must be 'evenly' or 'center'")


mu = alphas / (N_i + alphas)  # 各エージェントの突然変異率

subdir = ""
# 保存ディレクトリ名にbidirectional/outwardとnonzero_alphaの値を追加
if "bidirectional_flow_rate" in network_args:
    subdir += "bidirectional_flow-"
elif "outward_flow_rate" in network_args:
    subdir += "outward_flow-"
else:
    raise ValueError("network_args must contain either 'bidirectional_flow_rate' or 'outward_flow_rate'")

subdir += f"nonzero_alpha_{nonzero_alpha}"
subdir += f"_agents_{agents_count}_N_i_{N_i}"

save_dir = os.path.join(
    save_dir,
    subdir,
)
os.makedirs(save_dir, exist_ok=True)

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


def update_f(f, W, mu, N):
    """
    f:  (m, m) array, f[i, j] = f_{ij}
    W:  (m, m) migration matrix, W[i, k]
    mu: (m,)   mutation rates for each colony, mu[i]
    N:  (m,)   population size for each colony, N[i]
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
            print(f"Converged at iteration {i}, diff={diff}")
            return f_new
        f = f_new
    print(f"Did not converge within {max_iter} iterations. Final diff={diff}")
    return f

if __name__ == "__main__":
    raw_save_path = os.path.join(save_dir, "raw", "probability_of_identity.npy")
    if os.path.exists(raw_save_path):
        print("Loading existing heatmap data...")
        f_final = np.load(raw_save_path)
    else:
        print("Calculating probability of identity heatmap...")
        print(raw_save_path)
        m = agents_count
        # 初期値: 対角成分1, それ以外0
        f_init = np.eye(m)
        W = network_matrix
        # 収束まで回す
        f_final = run_until_convergence(f_init, W, mu, N)
        # 保存
        os.makedirs(os.path.dirname(raw_save_path), exist_ok=True)
        np.save(raw_save_path, f_final)
    distance_matrix_path = 1 / (f_final + 1e-10)  # ゼロ除算を避けるために小さな値を加える
    print("is_concentric_distribution:", is_concentric_distribution(distance_matrix_path))
    # ヒートマップ描画
    fig_path = os.path.join(save_dir, "fig")
    os.makedirs(fig_path, exist_ok=True)


    plt.figure(figsize=(8, 6))
    plt.imshow(f_final, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Probability of Identity")
    plt.title("Probability of Identity Heatmap (f)")
    plt.xlabel("Agent j")
    plt.ylabel("Agent i")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, "probability_of_identity_heatmap.png"))
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.title(f"Mean Distance from Agent {agent_id}")
    plt.bar(np.arange(f_final.shape[0]), f_final[agent_id])
    plt.xlabel("Other Agent")
    plt.ylabel("Mean Distance")
    plt.savefig(os.path.join(fig_path, f"probability_of_identity_from_agent{agent_id}.png"))


