
import numpy as np

from ilm import networks

import os
import matplotlib.pyplot as plt


agents_count = 15  # M
N_i = 100         # 各サブ集団のデータ数
alpha_per_data = 0.001  # 新語生成バイアス
nonzero_alpha = "center"  # "evenly" or "center"
# nonzero_alpha = "evenly"  # "evenly" or "center".
coupling_strength = 0.01  # m
save_interval_min = 1000
save_interval_max = 2000
save_dir = "data/naive_simulation/raw"  # 保存先ディレクトリ
os.makedirs(save_dir, exist_ok=True)
alpha = alpha_per_data * N_i  # 各エージェントのバイアス合計

N = np.ones(agents_count) * N_i  # 各エージェントのデータ数

# --- ネットワーク生成 ---
network_args = {"bidirectional_flow_rate": coupling_strength}
network_args = {"outward_flow_rate": coupling_strength}
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
    m = agents_count
    # 初期値: 対角成分1, それ以外0
    f_init = np.eye(m)
    W = network_matrix
    # 収束まで回す
    f_final = run_until_convergence(f_init, W, mu, N)

    # ヒートマップ描画
    plt.figure(figsize=(8, 6))
    plt.imshow(f_final, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Probability of Identity")
    plt.title("Probability of Identity Heatmap (f)")
    plt.xlabel("Agent j")
    plt.ylabel("Agent i")
    plt.tight_layout()
    plt.show()

