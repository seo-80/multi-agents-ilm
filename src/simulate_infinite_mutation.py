import numpy as np
import matplotlib.pyplot as plt
import ilm

# --- パラメータ設定 ---
agents_count = 15
alpha_per_data=0.001
fr = 0.01

agents_count = 15
alpha_per_data=0.1
fr = 0.001

view_agent = 1  # 可視化するエージェントのインデックス

mu = np.ones(agents_count) * alpha_per_data / (1 + alpha_per_data)
# mu =np.zeros(agents_count)
# mu[agents_count//2] = alpha_per_data / (1 + alpha_per_data)
network_args = {"bidirectional_flow_rate": fr}
network_args = {"outward_flow_rate": fr}
threshold = 1e-8  # シミュレーション終了の閾値

max_steps = 10000  # 最大ステップ数


# --- プロット設定 ---
PLOT_SETTINGS = {
    'distance_heatmap': True,
    'distance_bar': True,
    'distance_by_origin_line': True,
    'closeness_heatmap': True,
    'closeness_bar': True,
    'closeness_by_origin_line': True,
    'common_by_origin_line': True,
    'common_by_origin_bar': True,
    'final_share_bar': True,
    'final_share_heatmap': True,
    'nei_by_origin_line': True,
    'nei_by_origin_bar': True,
    'nei_total_heatmap': True,
    'nei_total_bar': True,
}

# --- ネットワーク生成 ---
W = ilm.networks.network(agents_count, network_args)
print(f"Network (W):{W}")

# --- シミュレーション ---
def simulate_infinite_mutation(agents_count, mu, W, max_steps, threshold):
    freq_history = [[] for _ in range(agents_count)]
    for i in range(agents_count):
        freq = np.zeros(agents_count)
        freq[i] = mu[i]
        freq_history[i].append(freq.copy())
        for t in range(max_steps):
            freq_next = np.zeros_like(freq)
            for j in range(agents_count):
                freq_next[j] = np.sum(freq * W[j] * (1 - mu))
            freq_history[i].append(freq_next.copy())
            if np.sum(freq_next) < threshold:
                print(f"Simulation stopped at step {t+1} (total freq < threshold)")
                break
            freq = freq_next
    return freq_history

freq_history = simulate_infinite_mutation(agents_count, mu, W, max_steps, threshold)

# --- 履歴の整形 ---
def pad_freq_history(freq_history):
    freq_history_np = [np.array(h) for h in freq_history]
    max_len = max(h.shape[0] for h in freq_history_np)
    freq_history_pad = np.zeros((agents_count, max_len, agents_count))
    for i, h in enumerate(freq_history_np):
        freq_history_pad[i, :h.shape[0], :] = h
    return freq_history_pad

freq_history_pad = pad_freq_history(freq_history)

# --- 距離・近さ・共通部分の計算 ---
def calc_distance_by_origin(freq_history_pad):
    agents_count = freq_history_pad.shape[0]
    distance_by_origin = np.zeros((agents_count, agents_count, agents_count))
    for origin in range(agents_count):
        for i in range(agents_count):
            for j in range(agents_count):
                distance_by_origin[origin, i, j] = np.sum(np.abs(freq_history_pad[origin, :, i] - freq_history_pad[origin, :, j]))
    return distance_by_origin

def calc_closeness_by_origin(freq_history_pad):
    agents_count = freq_history_pad.shape[0]
    closeness_by_origin = np.zeros((agents_count, agents_count, agents_count))
    for origin in range(agents_count):
        for i in range(agents_count):
            for j in range(agents_count):
                closeness_by_origin[origin, i, j] = np.sum(1 - np.abs(freq_history_pad[origin, :, i] - freq_history_pad[origin, :, j]))
    return closeness_by_origin

def calc_common_by_origin(freq_history_pad):
    agents_count = freq_history_pad.shape[0]
    common_by_origin = np.zeros((agents_count, agents_count, agents_count))
    for origin in range(agents_count):
        for i in range(agents_count):
            for j in range(agents_count):
                common_by_origin[origin, i, j] = np.sum(np.minimum(freq_history_pad[origin, :, i], freq_history_pad[origin, :, j]))
    return common_by_origin

# --- 距離・近さ・共通部分の行列 ---
distance_by_origin = calc_distance_by_origin(freq_history_pad)
closeness_by_origin = calc_closeness_by_origin(freq_history_pad)
common_by_origin = calc_common_by_origin(freq_history_pad)

# --- 全origin混合の距離・近さ ---
def calc_mixed_distance(freq_history_pad):
    agents_count = freq_history_pad.shape[0]
    distance = np.zeros((agents_count, agents_count))
    for i in range(agents_count):
        for j in range(agents_count):
            distance[i, j] = np.sum(np.abs(freq_history_pad[:, :, i] - freq_history_pad[:, :, j]))
    return distance

def calc_mixed_closeness(freq_history_pad):
    agents_count = freq_history_pad.shape[0]
    closeness = np.zeros((agents_count, agents_count))
    for i in range(agents_count):
        for j in range(agents_count):
            closeness[i, j] = np.sum(1 - np.abs(freq_history_pad[:, :, i] - freq_history_pad[:, :, j]))
    return closeness

distance = calc_mixed_distance(freq_history_pad)
closeness = calc_mixed_closeness(freq_history_pad)

# --- Nei’s standard genetic distance（Nei, 1972）---
def calc_nei_distance_by_origin(freq_history_pad):
    agents_count = freq_history_pad.shape[0]
    nei_distance_by_origin = np.zeros((agents_count, agents_count, agents_count))
    for origin in range(agents_count):
        for i in range(agents_count):
            for j in range(agents_count):
                freq_i = freq_history_pad[origin, :, i]
                freq_j = freq_history_pad[origin, :, j]
                J_XY = np.sum(np.sqrt(freq_i * freq_j))
                J_X = np.sum(freq_i ** 2)
                J_Y = np.sum(freq_j ** 2)
                if J_XY > 0 and J_X > 0 and J_Y > 0:
                    nei_distance_by_origin[origin, i, j] = -np.log(J_XY / np.sqrt(J_X * J_Y))
                else:
                    nei_distance_by_origin[origin, i, j] = np.nan
    return nei_distance_by_origin

def calc_mixed_nei_distance(freq_history_pad):
    agents_count = freq_history_pad.shape[0]
    mixed_nei_distance = np.zeros((agents_count, agents_count))
    for i in range(agents_count):
        for j in range(agents_count):
            freq_i = freq_history_pad[:, :, i].flatten()
            freq_j = freq_history_pad[:, :, j].flatten()
            J_XY = np.sum(freq_i * freq_j)
            J_X = np.sum(freq_i ** 2)
            J_Y = np.sum(freq_j ** 2)
            print(J_XY, J_X, J_Y)
            if J_XY > 0 and J_X > 0 and J_Y > 0:
                mixed_nei_distance[i, j] = -np.log(J_XY / np.sqrt(J_X * J_Y))
            else:
                mixed_nei_distance[i, j] = np.nan
    return mixed_nei_distance

nei_distance_by_origin = calc_nei_distance_by_origin(freq_history_pad)
nei_distance_by_origin_sum = np.nansum(nei_distance_by_origin, axis=(1,2))
mixed_nei_distance = calc_mixed_nei_distance(freq_history_pad)

# --- originごとの将来的な割合 ---
def calc_final_share_by_origin(freq_history_pad):
    agents_count = freq_history_pad.shape[0]
    final_freq_by_origin = np.zeros((agents_count, agents_count))
    for origin in range(agents_count):
        final_freq_by_origin[origin] = freq_history_pad[origin].sum(axis=0)
    return final_freq_by_origin

final_freq_by_origin = calc_final_share_by_origin(freq_history_pad)
final_freq_by_origin_mean = final_freq_by_origin.mean(axis=1)

# --- 可視化関数 ---
def plot_heatmap(matrix, title, xlabel, ylabel):
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label=title)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(range(matrix.shape[0]))
    plt.yticks(range(matrix.shape[1]))
    plt.tight_layout()
    plt.show()

def plot_bar(data, title, xlabel, ylabel):
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(data)), data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(range(len(data)))
    plt.tight_layout()
    plt.show()

def plot_line_by_origin(matrix_by_origin, view_agent, title, ylabel):
    plt.figure(figsize=(8, 5))
    for origin in range(matrix_by_origin.shape[0]):
        plt.plot(range(matrix_by_origin.shape[1]), matrix_by_origin[origin, view_agent], label=f'Origin {origin}')
    plt.xlabel('Agent')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.show()


# --- プロット ---
if PLOT_SETTINGS['distance_heatmap']:
    plot_heatmap(distance, 'Distance Heatmap Between Agents', 'Agent', 'Agent')
if PLOT_SETTINGS['distance_bar']:
    plot_bar(distance[view_agent], f'Distance from Agent {view_agent} to Others', 'Agent', f'Distance from Agent {view_agent}')
if PLOT_SETTINGS['distance_by_origin_line']:
    plot_line_by_origin(distance_by_origin, view_agent, f'Distance from Agent {view_agent} to Others (by origin)', f'Distance from Agent {view_agent}')

if PLOT_SETTINGS['closeness_heatmap']:
    plot_heatmap(closeness, 'Closeness Heatmap Between Agents', 'Agent', 'Agent')
if PLOT_SETTINGS['closeness_bar']:
    plot_bar(closeness[view_agent], f'Closeness from Agent {view_agent} to Others', 'Agent', f'Closeness from Agent {view_agent}')
if PLOT_SETTINGS['closeness_by_origin_line']:
    plot_line_by_origin(closeness_by_origin, view_agent, f'Closeness from Agent {view_agent} to Others (by origin)', f'Closeness from Agent {view_agent}')

if PLOT_SETTINGS['common_by_origin_line']:
    plot_line_by_origin(common_by_origin, view_agent, f'Commonality (sum of min freq) from Agent {view_agent} to Others (by origin)', f'Commonality from Agent {view_agent}')
if PLOT_SETTINGS['common_by_origin_bar']:
    plot_bar(common_by_origin.sum(axis=(1,2)), 'Total commonality generated by each origin (sum of min freq)', 'Origin Agent', 'Total commonality (sum of min freq over all pairs)')

if PLOT_SETTINGS['final_share_bar']:
    plot_bar(final_freq_by_origin_mean, 'Average total share of new words from each origin (until extinct)', 'Origin Agent', 'Average total share until extinct')
if PLOT_SETTINGS['final_share_heatmap']:
    plt.figure(figsize=(8, 6))
    plt.imshow(final_freq_by_origin, aspect='auto', cmap='viridis')
    plt.colorbar(label='Total future share (sum of freq)')
    plt.xlabel('Agent')
    plt.ylabel('Origin Agent')
    plt.title('Future share of each origin in each agent (heatmap)')
    plt.xticks(range(agents_count))
    plt.yticks(range(agents_count))
    plt.tight_layout()
    plt.show()

if PLOT_SETTINGS['nei_by_origin_line']:
    plt.figure(figsize=(8, 5))
    for origin in range(agents_count):
        plt.plot(range(agents_count), nei_distance_by_origin[origin, view_agent], label=f'Origin {origin}')
    plt.xlabel('Agent')
    plt.ylabel(f'Nei genetic distance from Agent {view_agent}')
    plt.title(f'Nei genetic distance from Agent {view_agent} to Others (by origin)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.show()
if PLOT_SETTINGS['nei_by_origin_bar']:
    plt.figure(figsize=(8, 5))
    plt.bar(range(agents_count), nei_distance_by_origin_sum)
    plt.xlabel('Origin Agent')
    plt.ylabel('Total Nei genetic distance (sum over all pairs)')
    plt.title('Total Nei genetic distance generated by each origin')
    plt.xticks(range(agents_count))
    plt.tight_layout()
    plt.show()
if PLOT_SETTINGS['nei_total_heatmap']:
    plt.figure(figsize=(6, 5))
    plt.imshow(mixed_nei_distance, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Nei genetic distance (total)')
    plt.title('Total Nei genetic distance Heatmap Between Agents')
    plt.xlabel('Agent')
    plt.ylabel('Agent')
    plt.xticks(range(agents_count))
    plt.yticks(range(agents_count))
    plt.tight_layout()
    plt.show()
if PLOT_SETTINGS['nei_total_bar']:
    plt.figure(figsize=(8, 5))
    plt.bar(range(agents_count), mixed_nei_distance[view_agent])
    plt.xlabel('Agent')
    plt.ylabel(f'Total Nei genetic distance from Agent {view_agent}')
    plt.title(f'Total Nei genetic distance from Agent {view_agent} to Others')
    plt.xticks(range(agents_count))
    plt.tight_layout()
    plt.show()