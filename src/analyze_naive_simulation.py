import numpy as np
import collections
import glob
import os
import sys

def calculate_genetic_similarity(state):
    """
    state配列から、対立遺伝子（ミーム）頻度を用いて
    エージェント間の遺伝学的類似度を計算します。
    """
    # state配列が空、または不正な場合に備える
    if state.size == 0:
        return np.array([]), np.array([])
        
    agents_count, N_i, _ = state.shape
    
    # データ総数N_iが0の場合は計算不可
    if N_i == 0:
        return np.zeros((agents_count, agents_count)), np.zeros((agents_count, agents_count))

    # 1. 各エージェントのミーム頻度を計算
    freq_list = []
    for i in range(agents_count):
        counts = collections.Counter(tuple(row) for row in state[i])
        freq_dict = {meme: count / N_i for meme, count in counts.items()}
        freq_list.append(freq_dict)

    # 2. 類似度行列を初期化
    dot_product_matrix = np.zeros((agents_count, agents_count))
    cosine_similarity_matrix = np.zeros((agents_count, agents_count))

    # 3. 全てのエージェントのペア (i, j) について類似度を計算
    for i in range(agents_count):
        for j in range(i, agents_count):
            freq_i = freq_list[i]
            freq_j = freq_list[j]
            all_memes = set(freq_i.keys()) | set(freq_j.keys())
            
            # 内積 (Genetic Identity)
            dot_product = sum(freq_i.get(meme, 0) * freq_j.get(meme, 0) for meme in all_memes)
            dot_product_matrix[i, j] = dot_product_matrix[j, i] = dot_product
            
            # コサイン類似度 (Nei's Genetic Identity I)
            norm_i = np.sqrt(sum(p**2 for p in freq_i.values()))
            norm_j = np.sqrt(sum(p**2 for p in freq_j.values()))
            
            if norm_i > 0 and norm_j > 0:
                cosine_sim = dot_product / (norm_i * norm_j)
            else:
                cosine_sim = 0.0
            cosine_similarity_matrix[i, j] = cosine_similarity_matrix[j, i] = cosine_sim
            
    return dot_product_matrix, cosine_similarity_matrix

def process_all_states(target_dir):
    """
    指定されたディレクトリ内のすべてのstate_*.npyファイルに対して
    類似度計算を行い、結果を保存します。
    """
    print(f"処理対象ディレクトリ: {target_dir}")

    # 1. ディレクトリ内の全stateファイルを時系列順に取得
    state_files = sorted(
        glob.glob(os.path.join(target_dir, "state_*.npy")),
        key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
    )

    if not state_files:
        print(f"エラー: '{target_dir}' に state_*.npy ファイルが見つかりませんでした。パスを確認してください。")
        return

    print(f"-> {len(state_files)}個のstateファイルを処理します...")

    # 2. 各ファイルをループ処理
    for state_file_path in state_files:
        try:
            base_name = os.path.basename(state_file_path)
            # 'state_0.npy' からインデックス '0' を抽出
            save_idx = base_name.split('_')[1].split('.')[0]

            # 3. 出力ファイル名を決定
            dot_output_path = os.path.join(target_dir, f"similarity_dot_{save_idx}.npy")
            cosine_output_path = os.path.join(target_dir, f"similarity_cosine_{save_idx}.npy")

            # 4. stateを読み込み、類似度を計算
            state = np.load(state_file_path)
            dot_sim, cos_sim = calculate_genetic_similarity(state)

            # 5. 結果を.npyファイルとして保存
            np.save(dot_output_path, dot_sim)
            np.save(cosine_output_path, cos_sim)

        except Exception as e:
            print(f"    ファイル '{state_file_path}' の処理中にエラーが発生しました: {e}", file=sys.stderr)

    print("\n✅ すべてのファイルの処理が完了しました。")
    print(f"結果は '{target_dir}' ディレクトリ内に保存されています。")
    print("  - `similarity_dot_*.npy`: 頻度の内積（遺伝的同一性）")
    print("  - `similarity_cosine_*.npy`: コサイン類似度（Neiの遺伝的同一性係数I）")


# --- メインの実行部分 ---
if __name__ == "__main__":
    # ▼▼▼【重要】ここにシミュレーション結果が保存されているディレクトリのパスを指定してください ▼▼▼
    # (例: "data/naive_simulation/raw/outward_flow-nonzero_alpha_center_fr_0.01_agents_15_N_i_500")
    TARGET_SIMULATION_DIRECTORY = "data/naive_simulation/raw/bidirectional_flow-nonzero_alpha_center_fr_0.01_agents_15_N_i_100"

    # --- 実行 ---
    if TARGET_SIMULATION_DIRECTORY == "YOUR_SIMULATION_DATA_DIRECTORY_HERE" or not os.path.isdir(TARGET_SIMULATION_DIRECTORY):
        print("⚠️ スクリプトを編集して、'TARGET_SIMULATION_DIRECTORY'変数に")
        print("   存在する正しいディレクトリパスを設定してから実行してください。")
    else:
        process_all_states(TARGET_SIMULATION_DIRECTORY)