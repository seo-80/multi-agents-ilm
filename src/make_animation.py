import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create animation from simulation state files')

    # Basic parameters
    parser.add_argument('--nonzero_alpha', '-a', type=str, default='center',
                       choices=['evenly', 'center'],
                       help='nonzero_alpha: "evenly" or "center"')
    parser.add_argument('--flow_type', '-f', type=str, default='bidirectional',
                       choices=['bidirectional', 'outward'],
                       help='flow_type: "bidirectional" or "outward"')
    parser.add_argument('--agents_count', '-m', type=int, default=7,
                       help='Number of agents (default: 7)')
    parser.add_argument('--N_i', '-n', type=int, default=100,
                       help='Number of data per subpopulation (default: 100)')
    parser.add_argument('--coupling_strength', '-c', type=float, default=0.01,
                       help='Coupling strength (default: 0.01)')
    parser.add_argument('--alpha_per_data', type=float, default=0.001,
                       help='New word generation bias (default: 0.001)')
    parser.add_argument('--start_idx', type=int, default=900,
                       help='Start state file index (default: 900)')
    parser.add_argument('--end_idx', type=int, default=1000,
                       help='End state file index (default: 1000)')
    parser.add_argument('--cmap_type', type=str, default='discrete',
                       choices=['continuous', 'discrete'],
                       help='Colormap type: "continuous" or "discrete" (default: discrete)')
    parser.add_argument('--interval', type=int, default=200,
                       help='Animation interval in milliseconds (default: 200)')
    parser.add_argument('--animation_type', type=str, default='both',
                       choices=['variant', 'agent', 'both'],
                       help='Animation type: "variant" colors by mutation, "agent" colors by originating agent, "both" creates both types (default: both)')

    return parser.parse_args()


def get_directory_paths(flow_type, nonzero_alpha, coupling_strength, agents_count, N_i, alpha):
    """Get state and save directory paths."""
    if flow_type == 'bidirectional':
        flow_str = 'bidirectional_flow-'
    elif flow_type == 'outward':
        flow_str = 'outward_flow-'
    else:
        raise ValueError(f"Unknown flow_type: {flow_type}")

    subdir = f"{flow_str}nonzero_alpha_{nonzero_alpha}_fr_{coupling_strength}_agents_{agents_count}_N_i_{N_i}_alpha_{alpha}"
    state_dir = f"data/naive_simulation/raw/{subdir}"
    save_dir = f"data/naive_simulation/fig/{subdir}"

    return state_dir, save_dir


def create_animation(args, animation_type, state_dir, inverse_indices, unique_variants):
    """Create a single animation of the specified type."""
    # Get save directory
    _, save_dir_base = get_directory_paths(
        args.flow_type, args.nonzero_alpha, args.coupling_strength,
        args.agents_count, args.N_i, args.alpha_per_data
    )

    os.makedirs(save_dir_base, exist_ok=True)
    save_path = os.path.join(save_dir_base, f"{animation_type}.gif")

    print(f"\n{'='*60}")
    print(f"Creating {animation_type} animation...")
    print(f"Save path: {save_path}")
    print(f"{'='*60}")

    # アニメーションの作成
    fig, ax = plt.subplots(figsize=(12, 7), layout='constrained')

    # 各変異に色を割り当てる
    if args.cmap_type == "continuous":
        colors = plt.get_cmap('viridis')
    elif args.cmap_type == "discrete":
        colors = plt.get_cmap('tab20')
    else:
        raise ValueError("cmap_type must be 'continuous' or 'discrete'")

    # 形状を取得
    num_timesteps, num_populations, num_agents = inverse_indices.shape
    num_unique_variants = len(unique_variants)

    # x軸の位置（集団ID）
    x = np.arange(num_populations)

    # エージェントベースの色分け用の準備
    if animation_type == 'agent':
        # 各変異がどのエージェントで生成されたかを取得（第2要素、index=1）
        variant_origins = unique_variants[:, 1].astype(int)
        # エージェント用のカラーマップ
        agent_colors = plt.get_cmap('tab10')  # 最大10エージェント
        print(f"Variant origins (agent IDs): {np.unique(variant_origins)}")

    def update(t):
        """各時間ステップのグラフを更新する関数"""
        ax.clear()

        # 現在の時間のデータを取得
        data_t = inverse_indices[t]

        if animation_type == 'variant':
            # 元のvariant別色分け
            # 各集団における各変異の数をカウント
            counts = np.zeros((num_populations, num_unique_variants))
            for p in range(num_populations):
                variant_ids, variant_counts = np.unique(data_t[p], return_counts=True)
                counts[p, variant_ids] = variant_counts

            # 積み上げ棒グラフを描画
            bottom = np.zeros(num_populations)
            for i in range(num_unique_variants):
                if args.cmap_type == "continuous":
                    color = colors(i / num_unique_variants)
                elif args.cmap_type == "discrete":
                    color = colors(i % colors.N)
                ax.bar(x, counts[:, i], bottom=bottom, label=f'Variant {i}', color=color)
                bottom += counts[:, i]

        elif animation_type == 'agent':
            # エージェント別色分け
            # 各集団で各エージェント由来の変異をカウント
            agent_counts = np.zeros((num_populations, args.agents_count))
            for p in range(num_populations):
                for agent_idx in range(data_t[p].shape[0]):
                    variant_id = data_t[p, agent_idx]
                    origin_agent = variant_origins[variant_id]
                    agent_counts[p, origin_agent] += 1

            # 積み上げ棒グラフを描画
            bottom = np.zeros(num_populations)
            for agent_id in range(args.agents_count):
                color = agent_colors(agent_id % agent_colors.N)
                ax.bar(x, agent_counts[:, agent_id], bottom=bottom,
                       label=f'Agent {agent_id}', color=color)
                bottom += agent_counts[:, agent_id]

        # グラフの体裁を整える
        title = f'Mutation Distribution at Time: {t}'
        if animation_type == 'agent':
            title += ' (colored by originating agent)'
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f'agent {i+1}' for i in range(num_populations)], rotation=45)
        ax.set_ylim(num_agents, 0)  # 引数を逆にすると軸が反転します

    # アニメーションオブジェクトの作成
    # intervalは描画間隔(ミリ秒)
    print(f"Creating animation with {num_timesteps} frames...")
    ani = animation.FuncAnimation(fig, update, frames=num_timesteps, interval=args.interval, repeat=False)

    # アニメーションをgifとして保存
    print(f"Saving animation to: {save_path}")
    ani.save(save_path, writer='pillow')

    print(f"Animation saved successfully!")

    # 閉じる
    plt.close(fig)


def main():
    args = parse_arguments()

    # Determine which animation types to create
    if args.animation_type == 'both':
        animation_types = ['variant', 'agent']
    else:
        animation_types = [args.animation_type]

    # Get state directory
    state_dir, _ = get_directory_paths(
        args.flow_type, args.nonzero_alpha, args.coupling_strength,
        args.agents_count, args.N_i, args.alpha_per_data
    )

    print(f"Loading state files from: {state_dir}")
    print(f"Start index: {args.start_idx}, End index: {args.end_idx}")

    # --- stateファイルを一括load ---
    all_states = []
    file_indices = []
    for idx in range(args.start_idx, args.end_idx + 1):
        path = os.path.join(state_dir, f"state_{idx}.npy")
        if not os.path.exists(path):
            print(f"ファイルが見つかりません: {path}")
            continue
        state = np.load(path)
        all_states.append(state)
        file_indices.append(idx)

    if not all_states:
        raise RuntimeError("指定範囲にstateファイルがありませんでした。")

    print(f"Loaded {len(all_states)} state files")

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

    print("inverse_indices.shape:", inverse_indices.shape)
    print(f"ユニークなvariant数: {len(unique_variants)}")

    # Create animations for each type
    for anim_type in animation_types:
        create_animation(args, anim_type, state_dir, inverse_indices, unique_variants)

    print(f"\n{'='*60}")
    print("All animations created successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()