import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import glob
import os
import time
rng = np.random.Generator(np.random.MT19937())

def simulate_finite_mutation(agents_count, mu, W, sample_size=100, seed=None, alpha=None):
    """
    agents_count: エージェント数
    mu: 各エージェントの新規変異確率 (np.array)
    W: ネットワーク行列 (agents_count x agents_count)
    max_steps: 最大ステップ数
    sample_size: 各エージェントが持つデータ数
    seed: 乱数シード
    alpha: 各エージェントのα（正則化項, Noneなら0で計算）
    """ 
    if seed is not None:
        np.random.seed(seed)
    if alpha is None:
        alpha = np.zeros(agents_count)
    freq_history = [[] for _ in range(agents_count)]
    steps = []
    for origin in range(agents_count):
        # 初期値: originのみ1, 他は0
        freq = np.zeros((agents_count,), dtype=int)
        freq[origin] = 1
        freq_history[origin].append(freq.copy())
        step =0
        while True:
            new_freq = np.zeros_like(freq)
            for i in range(agents_count):
                # 既存バリアントを受け取る確率
                p_existing = 0.0
                for j in range(agents_count):# todo 0かどうかでやる　
                    p_existing += W[i, j] * (freq[j] / (sample_size + alpha[j]))
                # 既存バリアントは全てoriginのものとしてカウント
                new_freq[i]  = np.random.binomial(sample_size, p_existing)

            freq = new_freq
            freq_history[origin].append(freq.copy())
            step += 1
            if np.sum(freq) == 0:
                steps.append(step)
                break
    return freq_history,steps


# 使い方例:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['save', 'plot'], required=True, help='save: simulate and save, plot: load and plot')
    parser.add_argument('--outdir', type=str, default='data/sim_results', help='directory to save/load npy files')
    parser.add_argument('--simulation_count', type=int, default=1000, help='number of simulations per run (save mode)')
    parser.add_argument('--save_fig', action='store_true', help='Save figures instead of showing them (plot mode only)')
    args = parser.parse_args()


    skip_large_values = False  # 大きな値をスキップするかどうか
    agents_count = 15
    alpha_per_data = 0.001
    fr = 0.001
    simulation_count = args.simulation_count
    sample_size = 100
    alpha = np.ones(agents_count) * alpha_per_data * sample_size
    mu = np.ones(agents_count) * alpha_per_data / (1 + alpha_per_data)
    network_args = {"outward_flow_rate": fr}
    # network_args = {"bidirectional_flow_rate": fr}

    def network_args_to_dirname(network_args):
        return "_".join(f"{k}_{v}" for k, v in network_args.items())

    subdir = network_args_to_dirname(network_args)
    outdir = os.path.join(args.outdir, subdir)
    os.makedirs(outdir, exist_ok=True)
    import ilm
    W = ilm.networks.network(agents_count, network_args)

    if args.mode == 'save':
        all_freq_histories = []
        all_steps = []
        timestamp = int(time.time())  # タイムスタンプをここで生成
        np.random.seed(timestamp)     # ここでseedを設定
        for i in tqdm(range(simulation_count), desc="Simulating"):
            freq_history, steps = simulate_finite_mutation(agents_count, mu, W, sample_size, alpha=alpha)
            all_freq_histories.append(freq_history)
            all_steps.append(steps)
        steps_by_origin = np.array(all_steps)  # shape: (simulation_count, agents_count)

        # --- 履歴の整形 ---
        def pad_freq_history(freq_history):
            freq_history_np = [np.array(h) for h in freq_history]
            max_len = max(h.shape[0] for h in freq_history_np)
            freq_history_pad = np.zeros((agents_count, max_len, agents_count))
            for i, h in enumerate(freq_history_np):
                freq_history_pad[i, :h.shape[0], :] = h
            return freq_history_pad
        padded_histories = [pad_freq_history(fh) for fh in all_freq_histories]
        max_steps = max(h.shape[1] for h in padded_histories)
        for i in range(len(padded_histories)):
            if padded_histories[i].shape[1] < max_steps:
                pad = np.zeros((agents_count, max_steps - padded_histories[i].shape[1], agents_count))
                padded_histories[i] = np.concatenate([padded_histories[i], pad], axis=1)
        def calc_distance_by_origin(freq_history_pad):
            agents_count = freq_history_pad.shape[0]
            distance_by_origin = np.zeros((agents_count, agents_count, agents_count))
            for origin in range(agents_count):
                for i in range(agents_count):
                    for j in range(agents_count):
                        distance_by_origin[origin, i, j] = np.sum(np.abs(freq_history_pad[origin, :, i] - freq_history_pad[origin, :, j]))
            return distance_by_origin
        all_distance_by_origin = []
        for fh in padded_histories:
            all_distance_by_origin.append(calc_distance_by_origin(fh))
        all_distance_by_origin = np.array(all_distance_by_origin)
        # 保存
        np.save(os.path.join(outdir, f'steps_by_origin_{timestamp}.npy'), steps_by_origin)
        np.save(os.path.join(outdir, f'all_distance_by_origin_{timestamp}.npy'), all_distance_by_origin)
        print(f'Saved to {outdir} (timestamp: {timestamp})')

    elif args.mode == 'plot':
        # ファイルを全てロードせず平均のみ計算
        dist_files = sorted(glob.glob(os.path.join(outdir, 'all_distance_by_origin_*.npy')))
        print(f'Found {len(dist_files)} distance files in {outdir}')
        mean_distance_by_origin = None
        count = 0
        for i, f  in enumerate(dist_files):
            arr = np.load(f)
            if skip_large_values and np.max(arr) >1000000:
                print(f'Skipping {i} due to large values')
                continue
            if mean_distance_by_origin is None:
                mean_distance_by_origin = np.zeros_like(arr[0], dtype=np.float64)
            mean_distance_by_origin += arr.sum(axis=0)
            count += arr.shape[0]
        mean_distance_by_origin /= count
        print(f'Calculated mean_distance_by_origin with {count} samples')

        # 以降はmean_distance_by_originのみを使ってプロット
        mean_distance = np.mean(mean_distance_by_origin, axis=0)
        plt.figure(figsize=(6, 5))
        plt.imshow(mean_distance, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Mean Distance')
        plt.title('Mean Distance Heatmap Between Agents (Averaged)')
        plt.xlabel('Agent')
        plt.ylabel('Agent')
        plt.xticks(range(agents_count))
        plt.yticks(range(agents_count))
        plt.tight_layout()
        if args.save_fig:
            plt.savefig(os.path.join(outdir, 'heatmap_mean_distance.png'))
            plt.close()
        else:
            plt.show()

        # mean_distance_by_originをoriginで和をとったものをヒートマップでプロット
        sum_distance = np.sum(mean_distance_by_origin, axis=0)
        plt.figure(figsize=(6, 5))
        plt.imshow(sum_distance, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Sum Distance (summed over origin)')
        plt.title('Sum of Mean Distance by Origin (Heatmap)')
        plt.xlabel('Agent')
        plt.ylabel('Agent')
        plt.xticks(range(agents_count))
        plt.yticks(range(agents_count))
        plt.tight_layout()
        if args.save_fig:
            plt.savefig(os.path.join(outdir, 'heatmap_sum_distance_by_origin.png'))
            plt.close()
        else:
            plt.show()
        # mean_distance_by_originの0からの距離だけ抜き出して棒グラフにする
        distances_from_0 = sum_distance[0, :]  # 0番originから各エージェントへの距離
        plt.figure(figsize=(8, 5))
        plt.bar(range(agents_count), distances_from_0)
        plt.xlabel('Agent')
        plt.ylabel('Mean Distance from Origin 0')
        plt.title('Mean Distance from Agent 0 to Others (by origin)')
        plt.xticks(range(agents_count))
        plt.tight_layout()
        if args.save_fig:
            plt.savefig(os.path.join(outdir, 'bar_mean_distance_from_0.png'))
            plt.close()
        else:
            plt.show()
        def plot_line_by_origin(matrix_by_origin, view_agent, title, ylabel, fname):
            plt.figure(figsize=(8, 5))
            for origin in range(matrix_by_origin.shape[0]):
                plt.plot(range(matrix_by_origin.shape[1]), matrix_by_origin[origin, view_agent], label=f'Origin {origin}')
            plt.xlabel('Agent')
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            plt.tight_layout()
            if args.save_fig:
                plt.savefig(os.path.join(outdir, fname))
                plt.close()
            else:
                plt.show()
        view_agent = 0
        plot_line_by_origin(mean_distance_by_origin, view_agent, f'Mean Distance from Agent {view_agent} to Others (by origin)', f'Mean Distance from Agent {view_agent}', f'line_mean_distance_agent{view_agent}.png')
        plot_line_by_origin(mean_distance_by_origin, view_agent+1, f'Mean Distance from Agent {view_agent+1} to Others (by origin)', f'Mean Distance from Agent {view_agent+1}', f'line_mean_distance_agent{view_agent+1}.png')


