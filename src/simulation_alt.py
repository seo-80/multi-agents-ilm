import numpy as np
import os
import time
from data_manager import save_obj
from ilm.networks import network
from tqdm import tqdm


def finite_mutation_simulation(agents_count, mu, W, sample_size=100, max_steps=100, alpha=None, seed=None):
    """
    Simulate finite mutation process with two dictionaries:
    1. index_to_mutation_info: index -> mutation info (tuple)
    2. index_to_count: index -> array of counts (per agent)
    Returns: history of both dictionaries (list of dicts)
    """
    if seed is not None:
        np.random.seed(seed)
    if alpha is None:
        alpha = np.zeros(agents_count)

    # Initialize
    index_to_mutation_info = {0: (0, -1, 0)}  # index 0: (timestep, agent, nth new mutation)
    index_to_count = {0: np.ones(agents_count, dtype=int) * sample_size}  # all agents start with index 0
    history_mutation_info = []
    history_count = []
    mutation_index_counter = 1  # next available index for new mutation

    for t in range(max_steps):
        # Save current state
        history_mutation_info.append(index_to_mutation_info.copy())
        history_count.append({k: v.copy() for k, v in index_to_count.items()})

        # Prepare next state
        next_index_to_count = {k: np.zeros(agents_count, dtype=int) for k in index_to_count}
        new_mutations = []  # (agent, how many new mutations)
        for agent in range(agents_count):
            probs = []
            indices = []
            for k in index_to_count:
                # Probability to inherit this mutation from network (using producer's mutation rate)
                p = 0.0
                for j in range(agents_count):
                    # Use mu[j] (producer's mutation rate) and the formula for inheritance
                    inherit_prob = W[agent, j] * (index_to_count[k][j] / (sample_size + alpha[j])) * (1 - mu[j])
                    p += inherit_prob
                probs.append(p)
                indices.append(k)
            # Probability for new mutation: sum_j W_ij * mu[j]
            p_new = 0.0
            for j in range(agents_count):
                p_new += W[agent, j] * mu[j]
            probs.append(p_new)
            indices.append(-1)  # -1 means new mutation
            # Normalize
            probs = np.array(probs)
            probs = probs / probs.sum()
            # Multinomial draw
            counts = np.random.multinomial(sample_size, probs)
            for idx, k in enumerate(indices):
                if k == -1:
                    if counts[idx] > 0:
                        new_mutations.append((agent, counts[idx]))
                else:
                    next_index_to_count[k][agent] += counts[idx]
        # Handle new mutations
        for agent, n_new in new_mutations:
            for i in range(n_new):
                new_idx = mutation_index_counter
                index_to_mutation_info[new_idx] = (t, agent, i)
                next_index_to_count[new_idx] = np.zeros(agents_count, dtype=int)
                next_index_to_count[new_idx][agent] = 1
                mutation_index_counter += 1
        # Remove extinct mutations
        next_index_to_count = {k: v for k, v in next_index_to_count.items() if np.any(v > 0)}
        index_to_mutation_info = {k: v for k, v in index_to_mutation_info.items() if k in next_index_to_count}
        index_to_count = next_index_to_count
        if len(index_to_count) == 0:
            break
    return history_mutation_info, history_count


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['save', 'plot'], required=True, help='save: simulate and save, plot: load and plot')
    parser.add_argument('--outdir', type=str, default='data/finite_mutation_results', help='directory to save results')
    parser.add_argument('--agents_count', type=int, default=15)
    parser.add_argument('--alpha_per_data', type=float, default=0.001)
    parser.add_argument('--fr', type=float, default=0.001)
    parser.add_argument('--sample_size', type=int, default=100)
    parser.add_argument('--max_steps', type=int, default=100)
    parser.add_argument('--simulation_count', type=int, default=10000)
    parser.add_argument('--save_fig', action='store_true', help='Save figures instead of showing them (plot mode only)')
    args = parser.parse_args()

    agents_count = args.agents_count
    alpha_per_data = args.alpha_per_data
    fr = args.fr
    sample_size = args.sample_size
    max_steps = args.max_steps
    simulation_count = args.simulation_count
    alpha = np.ones(agents_count) * alpha_per_data * sample_size
    mu = np.ones(agents_count) * alpha_per_data / (1 + alpha_per_data)
    network_args = {"outward_flow_rate": fr}
    W = network(agents_count, network_args)
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    timestamp = int(time.time())
    if args.mode == 'save':
        import pickle  # 追加
        progress_path = os.path.join(outdir, 'progress_finite_mutation.pkl')
        all_hist_mutinfo = []
        all_hist_count = []
        start_idx = 0
        # 進捗ファイルがあれば再開
        if os.path.exists(progress_path):
            with open(progress_path, 'rb') as f:
                progress = pickle.load(f)
            all_hist_mutinfo = progress['all_hist_mutinfo']
            all_hist_count = progress['all_hist_count']
            start_idx = progress['current_idx']
            print(f"Resuming from {start_idx}...")
        else:
            print("Starting new simulation...")
        save_interval = 1000  # 100回ごとに保存
        if save_interval > simulation_count:
            save_interval = simulation_count
        for i in tqdm(range(start_idx, simulation_count)):
            hist_mutinfo, hist_count = finite_mutation_simulation(
                agents_count, mu, W, sample_size, max_steps, alpha=alpha
            )
            all_hist_mutinfo.append(hist_mutinfo)
            all_hist_count.append(hist_count)
            # 途中保存
            if (i + 1) % save_interval == 0 or (i + 1) == simulation_count:
                with open(progress_path, 'wb') as f:
                    pickle.dump({
                        'all_hist_mutinfo': all_hist_mutinfo,
                        'all_hist_count': all_hist_count,
                        'current_idx': i + 1
                    }, f)
                print(f"Progress saved at {i + 1}/{simulation_count}")
        # 完了時に本保存
        save_obj({
            'mutation_info_history': all_hist_mutinfo,
            'count_history': all_hist_count
        }, os.path.join(outdir, f'finite_mutation_history_{timestamp}'), style='pkl')
        print(f'Saved simulation histories to {outdir} (timestamp: {timestamp})')
        # 進捗ファイル削除
        if os.path.exists(progress_path):
            os.remove(progress_path)
    # --- 距離計算とプロット ---
    elif args.mode == 'plot':
        import os
        import pickle
        import glob
        import matplotlib.pyplot as plt
        # 最新のpklファイルを探す
        pkl_files = sorted(glob.glob(os.path.join(outdir, 'finite_mutation_history_*.pkl')))
        if not pkl_files:
            print('No result file found for plotting.')
            exit()
        with open(pkl_files[-1], 'rb') as f:
            result = pickle.load(f)
        all_hist_count = result['count_history']
        # 距離計算: 各シミュレーション・各ステップでのエージェント間距離
        def calc_distance_by_origin(hist_count):
            # hist_count: list of dicts (stepごと), dict: index->count[agents]
            # 各stepで全indexのcountを合計してエージェントごとの配列に
            agents_count = list(hist_count[0].values())[0].shape[0]
            steps = len(hist_count)
            freq = np.zeros((steps, agents_count))
            for t, d in enumerate(hist_count):
                for v in d.values():
                    freq[t] += v
            # 距離行列
            dist = np.zeros((agents_count, agents_count))
            for i in range(agents_count):
                for j in range(agents_count):
                    dist[i, j] = np.sum(np.abs(freq[:, i] - freq[:, j]))
            return dist

        # 各シミュレーションの距離を平均
        dists = [calc_distance_by_origin(hist) for hist in all_hist_count]
        mean_dist = np.mean(dists, axis=0)
        print(dists)
        print(mean_dist)
        # プロット
        plt.figure(figsize=(6, 5))
        plt.imshow(mean_dist, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Mean Distance')
        plt.title('Mean Distance Heatmap Between Agents (Averaged)')
        plt.xlabel('Agent')
        plt.ylabel('Agent')
        plt.xticks(range(agents_count))
        plt.yticks(range(agents_count))
        plt.tight_layout()
        if args.save_fig:
            
            plt.savefig(os.path.join(outdir, f'heatmap_mean_distance_{timestamp}.png'))
            plt.close()
            print(f'Saved heatmap to {os.path.join(outdir, f"heatmap_mean_distance_{timestamp}.png")}')
        else:
            plt.show()
            print('Displayed heatmap plot.')
