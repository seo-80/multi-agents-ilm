import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- パラメータ ---
state_dir = "data/naive_simulation/raw/bidirectional_flow-nonzero_alpha_center_agents_15_N_i_100"  # 必要に応じて変更
# state_dir = "data/naive_simulation/raw/outward_flow-nonzero_alpha_center_agents_15_N_i_100"  # 必要に応じて変更
# state_dir = "data/naive_simulation/raw/outward_flow-nonzero_alpha_evenly_agents_15_N_i_100"  # 必要に応じて変更
# state_dir = "data/naive_simulation/raw/bidirectional_flow-nonzero_alpha_evenly_agents_15_N_i_100"  # 必要に応じて変更


state_dir = "data/naive_simulation/raw/bidirectional_flow-nonzero_alpha_center_fr_0.01_agents_7_N_i_25_alpha_0.001"  # 必要に応じて変更
state_dir = "data/naive_simulation/raw/bidirectional_flow-nonzero_alpha_center_fr_0.01_agents_7_N_i_100_alpha_0.001"  # 必要に応じて変更
# state_dir = "data/naive_simulation/raw/bidirectional_flow-nonzero_alpha_center_fr_0.01_agents_7_N_i_1600_alpha_0.001"  # 必要に応じて変更

# state_dir = "data/naive_simulation/raw/outward_flow-nonzero_alpha_evenly_fr_0.01_agents_7_N_i_25_alpha_0.001"  # 必要に応じて変更
# state_dir = "data/naive_simulation/raw/outward_flow-nonzero_alpha_evenly_fr_0.01_agents_7_N_i_100_alpha_0.001"  # 必要に応じて変更
state_dir = "data/naive_simulation/raw/outward_flow-nonzero_alpha_evenly_fr_0.01_agents_7_N_i_1600_alpha_0.001"  # 必要に応じて変更

start_idx = 900  # 最初のstateファイル番号
end_idx = 1000   # 最後のstateファイル番号

cmap_type = "discrete"  # "continuous" or "discrete"

save_dir = "data/naive_simulation/animation"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, os.path.basename(state_dir) + ".gif")
# --- stateファイルを一括load ---
all_states = []
file_indices = []
for idx in range(start_idx, end_idx + 1):
    path = os.path.join(state_dir, f"state_{idx}.npy")
    if not os.path.exists(path):
        print(f"ファイルが見つかりません: {path}")
        continue
    state = np.load(path)
    all_states.append(state)
    file_indices.append(idx)

if not all_states:
    raise RuntimeError("指定範囲にstateファイルがありませんでした。")

# --- 全stateを連結してユニークなvariantを抽出 ---
flat_states = [s.reshape(-1, 3) for s in all_states]
all_flat = np.concatenate(flat_states, axis=0)
unique_variants, inverse_indices = np.unique(all_flat, axis=0, return_inverse=True)


# --- 各stateごとにvariantインデックスを作成 ---
variant_indices_list = []
ptr = 0
for s in all_states:
    n = s.shape[0] * s.shape[1]
    indices = inverse_indices[ptr:ptr+n].reshape(s.shape[0], s.shape[1])
    variant_indices_list.append(indices)
    ptr += n
inverse_indices = inverse_indices.reshape(-1, all_states[0].shape[0], all_states[0].shape[1])
# --- 保存 ---
print("inverse_indices.shape:", inverse_indices.shape)
print(f"ユニークなvariant数: {len(unique_variants)}")



# アニメーションの作成
fig, ax = plt.subplots(figsize=(12, 7), layout='constrained')

# 各変異に色を割り当てる
if cmap_type == "continuous":
    colors = plt.get_cmap('viridis') 
elif cmap_type == "discrete": 
    colors = plt.get_cmap('tab20')
else:
    raise ValueError("cmap_type must be 'continuous' or 'discrete'")

# 形状を取得
num_timesteps, num_populations, num_agents = inverse_indices.shape
num_unique_variants = len(unique_variants)

# x軸の位置（集団ID）
x = np.arange(num_populations)

def update(t):
    """各時間ステップのグラフを更新する関数"""
    ax.clear()

    # 現在の時間のデータを取得
    data_t = inverse_indices[t]

    # 各集団における各変異の数をカウント
    counts = np.zeros((num_populations, num_unique_variants))
    for p in range(num_populations):
        variant_ids, variant_counts = np.unique(data_t[p], return_counts=True)
        counts[p, variant_ids] = variant_counts

    # 積み上げ棒グラフを描画
    bottom = np.zeros(num_populations)
    for i in range(num_unique_variants):
        if cmap_type == "continuous":
            color = colors(i / num_unique_variants)
        elif cmap_type == "discrete": 
            color = colors(i % colors.N)  
        ax.bar(x, counts[:, i], bottom=bottom, label=f'Variant {i}', color=color)
        bottom += counts[:, i]

    # グラフの体裁を整える
    ax.set_title(f'Mutation Distribution at Time: {t}')
    ax.set_xticks(x)
    ax.set_xticklabels([f'agent {i+1}' for i in range(num_populations)], rotation=45)
    ax.set_ylim(num_agents, 0) # 引数を逆にすると軸が反転します



# アニメーションオブジェクトの作成
# intervalは描画間隔(ミリ秒)
ani = animation.FuncAnimation(fig, update, frames=num_timesteps, interval=200, repeat=False)

# アニメーションをgifとして保存
ani.save(save_path, writer='pillow')


# 閉じる
plt.close(fig)