import numpy as np
from ilm import networks
import random
import os
import glob
import csv
import pickle
import time

import collections
# --- パラメータ設定例 ---
agents_count = 15  # M
N_i = 100          # 各サブ集団のデータ数
alpha_per_data = 0.001  # 新語生成バイアス
nonzero_alpha = "center"  # "evenly" or "center"
# nonzero_alpha = "evenly"  # "evenly" or "center"
coupling_strength = 0.01  # m
save_interval_min = 1000
save_interval_max = 2000
save_dir = "data/naive_simulation/raw"  # 保存先ディレクトリ
os.makedirs(save_dir, exist_ok=True)
alpha = alpha_per_data * N_i  # 各エージェントのバイアス合計


# --- ネットワーク生成 ---
network_args = {"bidirectional_flow_rate": coupling_strength}
# network_args = {"outward_flow_rate": coupling_strength}
network_matrix = networks.network(agents_count, args=network_args)
alphas = np.zeros(agents_count, dtype=float)
if nonzero_alpha == "evenly":
    # 各エージェントに均等にバイアスを設定
    alphas = np.ones(agents_count) * alpha
elif nonzero_alpha == "center":
    # 中央のエージェントだけにバイアスを設定
    alphas = np.zeros(agents_count)
    alphas[agents_count // 2] = alpha
else:
    raise ValueError("nonzero_alpha must be 'evenly' or 'center'")

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
# --- 突然変異率 ---
mu = alphas / (N_i + alphas)  # 各エージェントの突然変異率

# --- 再開用: 既存ファイルの確認 ---
state_files = sorted(glob.glob(os.path.join(save_dir, "state_*.npy")), key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))
distance_files = sorted(glob.glob(os.path.join(save_dir, "distance_*.npy")), key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))
random_state_files = sorted(glob.glob(os.path.join(save_dir, "random_state_*.pkl")), key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[2]))

if state_files:
    # 既存ファイルがあれば再開
    last_idx = int(os.path.splitext(os.path.basename(state_files[-1]))[0].split('_')[1])
    state = np.load(state_files[-1])
    # 正確なtをcsvから取得
    csv_path = os.path.join(save_dir, "save_idx_t_map.csv")
    if os.path.exists(csv_path):
        import csv
        with open(csv_path, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            t = None
            for row in reader:
                if int(row["save_idx"]) == last_idx:
                    t = int(row["t"])
            if t is None:
                raise ValueError(f"t_log.csvにsave_idx={last_idx}の記録がありません")
    else:
        # t_log.csvがない場合は従来通り
        print(f"t_log.csvが見つかりません。おおよそのtを計算します。それでもよければ y を入力してください。")
        if input().strip().lower() != 'y':
            raise FileNotFoundError(f"t_log.csvが見つかりません。save_dir={save_dir}を確認してください。")
        t = (last_idx + 1) * save_interval_max  # おおよそのt（厳密には保存間隔がランダムなので正確なtは記録しておくのが理想）
    # 乱数状態も復元
    if random_state_files:
        with open(random_state_files[-1], "rb") as f:
            np.random.set_state(pickle.load(f))
    save_idx = last_idx + 1
else:
    # 新規開始
    np.random.seed(0)  # 再現性のためのシード設定
    state = np.zeros((agents_count, N_i, 3), dtype=int)
    for i in range(agents_count):
        for j in range(N_i):
            state[i, j] = [0, i, j]  # [タイムステップ, エージェント番号, 何個目か]
    t = 0
    save_idx = 0
# --- 距離計算関数 ---t


def calc_agent_distance(state):
    agents_count, N_i, _ = state.shape
    # 各エージェントの全データを多重集合化
    counters = [collections.Counter([tuple(d) for d in state[i]]) for i in range(agents_count)]
    dist = np.zeros((agents_count, agents_count), dtype=int)
    for i in range(agents_count):
        for j in range(i, agents_count):
            all_keys = set(counters[i]) | set(counters[j])
            dist_ij = sum(abs(counters[i][k] - counters[j][k]) for k in all_keys)
            dist[i, j] = dist[j, i] = dist_ij
    return dist

# --- メインループ ---
while True:
    # --- save_intervalをサンプル ---
    save_interval = np.random.randint(save_interval_min, save_interval_max + 1)
    next_save = t + save_interval
    start = time.time()

    while t < next_save:
        # 1. データ受け渡し数のサンプル
        data_flow_count = networks.generate_data_flow_count(
            data_flow_rate=network_matrix,
            total_data_count=N_i
        )
        total_num = N_i * agents_count
        i_idx, j_idx, local_k = np.empty(total_num, dtype=int), np.empty(total_num, dtype=int), np.empty(total_num, dtype=int)
        ptr = 0
        for i in range(agents_count):
            cnt = 0
            for j, count in enumerate(data_flow_count[i]):
                for _ in range(int(count)):
                    i_idx[ptr] = i
                    j_idx[ptr] = j
                    local_k[ptr] = cnt
                    ptr += 1
                    cnt += 1
        i_idx = np.array(i_idx)
        j_idx = np.array(j_idx)
        local_k = np.array(local_k)
        num_copies = len(i_idx)

        # 2. j_idxごとのmu_jで突然変異判定
        mu_j = mu[j_idx]
        mutation_flags = np.random.rand(num_copies) < mu_j

        # 3. コピー元のデータindexをまとめて決定
        random_indices = np.random.randint(N_i, size=num_copies)

        # 4. 新しい状態配列を一括初期化
        next_state = np.zeros_like(state)

        # 5. 突然変異：j_idxのmu_jで判定したぶんだけ
        if mutation_flags.any():
            next_state[i_idx[mutation_flags], local_k[mutation_flags]] = np.stack(
                [np.full(mutation_flags.sum(), t), i_idx[mutation_flags], local_k[mutation_flags]], axis=1
            )

        # 6. コピー：jのデータから
        if (~mutation_flags).any():
            next_state[i_idx[~mutation_flags], local_k[~mutation_flags]] = state[j_idx[~mutation_flags], random_indices[~mutation_flags]]

        state = next_state
        t += 1

    print(f"t={t}, save_interval={save_interval}, elapsed={time.time() - start:.2f} seconds")

    # --- 保存処理 ---
    np.save(os.path.join(save_dir, f"state_{save_idx}.npy"), state)
    dist = calc_agent_distance(state)
    np.save(os.path.join(save_dir, f"distance_{save_idx}.npy"), dist)
    with open(os.path.join(save_dir, f"random_state_{save_idx}.pkl"), "wb") as f:
        pickle.dump(np.random.get_state(), f)
    # save_idxとtの対応をcsvで保存

    # 保存用のcsvファイルパス
    csv_path = os.path.join(save_dir, "save_idx_t_map.csv")
    # 追記モードで開き、ヘッダがなければ書く
    write_header = not os.path.exists(csv_path) or save_idx == 0
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["save_idx", "t"])
        writer.writerow([save_idx, t])
    save_idx += 1

# --- 終了 ---
