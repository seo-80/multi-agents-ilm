import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import glob
import os
import time
import pickle
import ilm
from scipy import stats
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

def calc_distance_by_origin(freq_history_pad):
    """
    各originからの距離を計算する関数
    freq_history_pad: shape (agents_count, max_len, agents_count) の配列
    """
    agents_count = freq_history_pad.shape[0]
    distance_by_origin = np.zeros((agents_count, agents_count, agents_count))
    for origin in range(agents_count):
        for i in range(agents_count):
            for j in range(agents_count):
                distance_by_origin[origin, i, j] = np.sum(np.abs(freq_history_pad[origin, :, i] - freq_history_pad[origin, :, j]))
    return distance_by_origin

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
    return False

# 使い方例:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['save', 'plot', 'print_simulation', 'resimulate', 'check_concentric'], required=True, help='save: simulate and save, plot: load and plot, print_simulation: print details of a specific simulation, resimulate: rerun a specific simulation with the same seed, check_concentric: check if distribution is concentric')
    parser.add_argument('--outdir', type=str, default='data/raw', help='directory to save/load npy files')
    parser.add_argument('--figdir', type=str, default='data/fig', help='directory to save figures')
    parser.add_argument('--simulation_count', type=int, default=1000, help='number of simulations per run (save mode)')
    parser.add_argument('--sample_size', type=int, default=100, help='sample size for each agent (default: 100)')
    parser.add_argument('--agents_count', type=int, default=15, help='number of agents in the simulation (default: 15)')
    parser.add_argument('--save_fig', action='store_true', help='Save figures instead of showing them (plot mode only)')
    parser.add_argument('--simulation_index', type=int, help='Index of simulation to print/resimulate (required for print_simulation and resimulate modes)')
    parser.add_argument('--outward_flow', action='store_true', help='Use outward flow rate (default is bidirectional flow rate)')
    args = parser.parse_args()

    if args.mode == 'print_simulation' and args.simulation_index is None:
        parser.error("--simulation_index is required for print_simulation mode")

    skip_large_values = True  # 大きな値をスキップするかどうか
    agents_count = args.agents_count
    alpha_per_data = 0.001
    fr = 0.01
    simulation_count = args.simulation_count
    sample_size = args.sample_size
    alpha = np.ones(agents_count) * alpha_per_data * sample_size
    print(alpha)
    mu = np.ones(agents_count) * alpha_per_data / (1 + alpha_per_data)
    if args.outward_flow:
        network_args = {"outward_flow_rate": fr}
    else:
        network_args = {"bidirectional_flow_rate": fr}
    def network_args_to_dirname(network_args):
        return "_".join(f"{k}_{v}" for k, v in network_args.items())

    subdir = network_args_to_dirname(network_args)
    if sample_size != 100:
        print(f"Warning: sample_size is set to {sample_size}, which may affect results.")
        subdir += f"_sample_size_{sample_size}"
    if agents_count != 15:
        print(f"Warning: agents_count is set to {agents_count}, which may affect results.")
        subdir += f"_agents_{agents_count}"
    outdir = os.path.join(args.outdir, subdir)
    figdir = os.path.join(args.figdir, subdir)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)
    W = ilm.networks.network(agents_count, network_args)

    if args.mode == 'save':
        all_distance_by_origin = []
        all_steps = []
        timestamp = int(time.time())  # タイムスタンプをここで生成
        np.random.seed(timestamp)     # ここでseedを設定
        
        for i in tqdm(range(simulation_count), desc="Simulating"):
            # シミュレーション実行
            freq_history, steps = simulate_finite_mutation(agents_count, mu, W, sample_size, alpha=alpha)
            all_steps.append(steps)
            
            # 履歴の整形と距離計算を即座に行う
            freq_history_np = [np.array(h) for h in freq_history]
            max_len = max(h.shape[0] for h in freq_history_np)
            freq_history_pad = np.zeros((agents_count, max_len, agents_count))
            for j, h in enumerate(freq_history_np):
                freq_history_pad[j, :h.shape[0], :] = h
                
            # 距離計算
            distance_by_origin = calc_distance_by_origin(freq_history_pad)
            all_distance_by_origin.append(distance_by_origin)
                
        # 完了時に保存
        steps_by_origin = np.array(all_steps)
        all_distance_by_origin = np.array(all_distance_by_origin)
        
        np.save(os.path.join(outdir, f'all_distance_by_origin_{timestamp}.npy'), all_distance_by_origin)
        np.save(os.path.join(outdir, f'steps_by_origin_{timestamp}.npy'), steps_by_origin)

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
        mean_distance_by_origin *= mu * sample_size  # Multiply by mu along origin axis
        
        print(f'Calculated mean_distance_by_origin with {count} samples')

        # 以降はmean_distance_by_originのみを使ってプロット
        
        mean_distance = np.sum(mean_distance_by_origin, axis=0)
        print(mean_distance)
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
            plt.savefig(os.path.join(figdir, 'heatmap_mean_distance.png'))
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
            plt.savefig(os.path.join(figdir, 'heatmap_sum_distance_by_origin.png'))
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
            plt.savefig(os.path.join(figdir, 'bar_mean_distance_from_0.png'))
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
                plt.savefig(os.path.join(figdir, fname))
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
        plt.savefig(os.path.join(figdir, 'total_distance_heatmap.png'))
        plt.close()

        plt.figure(figsize=(10, 8))
        plt.imshow(distance_data[simulation_index_max, agent_index_max], cmap='viridis', aspect='auto')
        plt.colorbar(label='Distance')
        plt.title(f'Distance from Origin {agent_index_max} to All Agents (Simulation {simulation_index_max})')
        plt.savefig(os.path.join(figdir, f'distance_from_origin_{agent_index_max}_simulation_{simulation_index_max}.png'))
        plt.close()
        print(f"Saved heatmap to {os.path.join(figdir, 'total_distance_heatmap.png')}")

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

    elif args.mode == 'check_concentric':
        # Find all simulation files
        dist_files = sorted(glob.glob(os.path.join(outdir, 'all_distance_by_origin_*.npy')))
        if not dist_files:
            print(f'Error: No simulation files found in {outdir}')
            exit(1)
        
        total_positive = 0
        total_files = len(dist_files)
        
        # Initialize array to store all mean distances
        all_mean_distances = []
        
        for file_path in tqdm(dist_files):
            # Load the distance data
            distance_data = np.load(file_path)
            timestamp = os.path.basename(file_path).split('_')[-1].split('.')[0]
            
            # Calculate mean distance across all simulations and origins in this file
            mean_distance = np.mean(distance_data, axis=(0, 1))  # Average over simulations and origins
            mean_distance *= mu
            all_mean_distances.append(mean_distance)
            
            # Check if it's concentric
            is_concentric = is_concentric_distribution(mean_distance)
            
            if is_concentric:
                total_positive += 1
        
        # Calculate overall mean distance
        overall_mean_distance = np.mean(all_mean_distances, axis=0)
        
        # Check if overall mean distance is concentric
        is_overall_concentric = is_concentric_distribution(overall_mean_distance)
        
        # Print overall results
        print("\nOverall Results:")
        print(f"Total Positive/Total: {total_positive}/{total_files}")
        print(f"Overall Percentage: {total_positive/total_files*100:.2f}%")
        print(f"\nOverall Mean Distance Matrix is concentric: {is_overall_concentric}")

        # Plot histograms of distances between specified agents
        import matplotlib.pyplot as plt
        
        # Example: Plot distances between agents 0-7 and 0-10
        agent_pairs = [(0, 7), (0, 10)]  # You can modify these pairs as needed
        
        plt.figure(figsize=(10, 6))
        colors = ['blue', 'red']  # 各エージェントペアの色
        
        # データを収集
        distances_0_7 = np.array([dist[0, 7] for dist in all_mean_distances])
        distances_0_10 = np.array([dist[0, 10] for dist in all_mean_distances])
        
        # 2項検定の準備
        over_count_0_7 = np.sum(distances_0_7 - distances_0_10 > 0)
        under_count_0_7 = len(distances_0_7) - over_count_0_7
        
        total = len(distances_0_7)
        p_value = stats.binomtest(over_count_0_7, n=total, p=0.5, alternative='two-sided').pvalue
        
        # プロット
        for i, (agent1, agent2) in enumerate(agent_pairs):
            distances = [dist[agent1, agent2] for dist in all_mean_distances]
            mean_dist = np.mean(distances)
            plt.hist(distances, bins=30, alpha=0.5, label=f'Agent {agent1}-{agent2}', color=colors[i])
            plt.axvline(mean_dist, color=colors[i], linestyle='--', 
                       label=f'Mean {agent1}-{agent2}: {mean_dist:.2f}')
        
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Distances Between Agent Pairs\nBinomial test p-value: {p_value:.3e}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if args.save_fig:
            plt.savefig(os.path.join(figdir, 'agent_pair_distances_histogram.png'))
        else:
            plt.show()
        plt.close()
        
        # 検定結果を表示
        print(f"\nBinomial test results:")
        print(f"Number of times 0-7 > 0-10: {over_count_0_7}/{total}")
        print(f"Number of times 0-7 <= 0-10: {under_count_0_7}/{total}")
        print(f"p-value: {p_value:.3e}")


        


