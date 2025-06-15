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
    parser.add_argument('--mode', choices=['save', 'plot', 'print_simulation', 'resimulate'], required=True, help='save: simulate and save, plot: load and plot, print_simulation: print details of a specific simulation, resimulate: rerun a specific simulation with the same seed')
    parser.add_argument('--outdir', type=str, default='data/sim_results', help='directory to save/load npy files')
    parser.add_argument('--simulation_count', type=int, default=1000, help='number of simulations per run (save mode)')
    parser.add_argument('--save_fig', action='store_true', help='Save figures instead of showing them (plot mode only)')
    parser.add_argument('--simulation_index', type=int, help='Index of simulation to print/resimulate (required for print_simulation and resimulate modes)')
    args = parser.parse_args()

    if args.mode == 'print_simulation' and args.simulation_index is None:
        parser.error("--simulation_index is required for print_simulation mode")

    skip_large_values = True  # 大きな値をスキップするかどうか
    agents_count = 15
    alpha_per_data = 0.001
    fr = 0.01
    simulation_count = args.simulation_count
    sample_size = 100
    alpha = np.ones(agents_count) * alpha_per_data * sample_size
    mu = np.ones(agents_count) * alpha_per_data / (1 + alpha_per_data)
    network_args = {"outward_flow_rate": fr}
    network_args = {"bidirectional_flow_rate": fr}

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

    elif args.mode == 'print_simulation':
        # ファイルをロード
        dist_files = sorted(glob.glob(os.path.join(outdir, 'all_distance_by_origin_*.npy')))
        if not dist_files:
            print(f'Error: No simulation files found in {outdir}')
            exit(1)
        
        if args.simulation_index >= len(dist_files):
            print(f"Error: simulation_index {args.simulation_index} is out of range (0-{len(dist_files)-1})")
            exit(1)
        
        # 指定されたファイルをロード
        file_path = dist_files[args.simulation_index]
        print(f"\nLoading simulation from: {file_path}")
        
        # ファイル名からタイムスタンプを取得
        timestamp = os.path.basename(file_path).split('_')[-1].split('.')[0]
        
        # 対応するstepsファイルもロード
        steps_file = os.path.join(outdir, f'steps_by_origin_{timestamp}.npy')
        if not os.path.exists(steps_file):
            print(f"Warning: Could not find corresponding steps file: {steps_file}")
            steps = None
        else:
            steps = np.load(steps_file)
        print("steps")
        print("shape", steps.shape)

        
        
        # 距離データをロード
        distance_data = np.load(file_path)
        print("distance_data")
        print("shape", distance_data.shape)
        print("max value",np.max(distance_data))
        print("argmax",np.unravel_index(np.argmax(distance_data), distance_data.shape))
        simulation_index_max = np.unravel_index(np.argmax(distance_data), distance_data.shape)[0]
        agent_index_max = np.unravel_index(np.argmax(distance_data), distance_data.shape)[1]
        
        print(f"\nSimulation Details:")
        print(f"Network parameters: {network_args}")
        print(f"Number of agents: {agents_count}")
        print(f"Sample size: {sample_size}")
        print(f"Alpha: {alpha}")
        print(f"Mutation rates (mu): {mu}")
        
        # if steps is not None:
        #     print("\nSteps until extinction for each origin:")
        #     for origin, step in enumerate(steps):
        #         print(f"Origin {origin}: {step} steps")
        
        print("\nDistance matrix by origin:")
        for origin in range(agents_count):
            print(f"\nOrigin {origin}:")
            print(distance_data[simulation_index_max, origin])  # 最初のシミュレーションのデータを表示
        print("steps:", steps[simulation_index_max] if steps is not None else "N/A")
        import matplotlib.pyplot as plt
        # Calculate total distance for each origin
        total_distance = np.sum(distance_data[0], axis=1)  # Sum across all destinations for each origin
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(total_distance.reshape(1, -1), cmap='viridis', aspect='auto')
        plt.colorbar(label='Total Distance')
        plt.title('Total Distance by Origin')
        plt.xlabel('Origin Agent')
        plt.yticks([])  # Hide y-axis ticks since we only have one row
        
        # Save the plot
        plt.savefig(os.path.join(outdir, 'total_distance_heatmap.png'))
        plt.close()

        plt.figure(figsize=(10, 8))
        plt.imshow(distance_data[simulation_index_max, agent_index_max], cmap='viridis', aspect='auto')
        plt.colorbar(label='Distance')
        plt.title(f'Distance from Origin {agent_index_max} to All Agents (Simulation {simulation_index_max})')
        plt.savefig(os.path.join(outdir, f'distance_from_origin_{agent_index_max}_simulation_{simulation_index_max}.png'))
        plt.close()
        print(f"Saved heatmap to {os.path.join(outdir, 'total_distance_heatmap.png')}")

    elif args.mode == 'resimulate':
        if args.simulation_index is None:
            parser.error("--simulation_index is required for resimulate mode")
        
        # Find the original simulation file
        dist_files = sorted(glob.glob(os.path.join(outdir, 'all_distance_by_origin_*.npy')))
        if not dist_files:
            print(f'Error: No simulation files found in {outdir}')
            exit(1)
        
        if args.simulation_index >= len(dist_files):
            print(f"Error: simulation_index {args.simulation_index} is out of range (0-{len(dist_files)-1})")
            exit(1)
        
        # Get the timestamp from the original file
        file_path = dist_files[args.simulation_index]
        timestamp = os.path.basename(file_path).split('_')[-1].split('.')[0]
        
        # Set the same seed for reproducibility
        np.random.seed(int(timestamp))
        
        # Run the simulation
        print(f"Resimulating with seed {timestamp}...")
        freq_history, steps = simulate_finite_mutation(agents_count, mu, W, sample_size, alpha=alpha)
        
        # Process and save the results
        all_freq_histories = [freq_history]
        all_steps = [steps]
        steps_by_origin = np.array(all_steps)
        
        # Process frequency history
        padded_histories = [pad_freq_history(fh) for fh in all_freq_histories]
        max_steps = max(h.shape[1] for h in padded_histories)
        for i in range(len(padded_histories)):
            if padded_histories[i].shape[1] < max_steps:
                pad = np.zeros((agents_count, max_steps - padded_histories[i].shape[1], agents_count))
                padded_histories[i] = np.concatenate([padded_histories[i], pad], axis=1)
        
        # Calculate distances
        all_distance_by_origin = []
        for fh in padded_histories:
            all_distance_by_origin.append(calc_distance_by_origin(fh))
        all_distance_by_origin = np.array(all_distance_by_origin)
        
        # Save with a new timestamp
        new_timestamp = int(time.time())
        # np.save(os.path.join(outdir, f'steps_by_origin_{new_timestamp}.npy'), steps_by_origin)
        # np.save(os.path.join(outdir, f'all_distance_by_origin_{new_timestamp}.npy'), all_distance_by_origin)
        print(f'Resimulation saved to {outdir} (new timestamp: {new_timestamp})')


