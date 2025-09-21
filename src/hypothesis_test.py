import numpy as np
from scipy.stats import spearmanr

# expected_distance: shape (agents_count, agents_count)
def test_agent_distance_correlation(expected_distance):
    agents_count = expected_distance.shape[0]
    # 物理的距離（単純なインデックス差）
    physical_dist = np.abs(np.arange(agents_count).reshape(-1,1) - np.arange(agents_count).reshape(1,-1))
    # 上三角成分のみを使う
    idx = np.triu_indices(agents_count, k=1)
    dist_flat = expected_distance[idx]
    phys_flat = physical_dist[idx]
    # Spearman順位相関
    corr, pval = spearmanr(phys_flat, dist_flat)
    print(f"Spearman correlation: {corr:.3f}, p-value: {pval:.3g}")
    if pval < 0.05:
        print("帰無仮説（相関がない）は棄却されます")
    else:
        print("帰無仮説（相関がない）は棄却できません")
    return corr, pval

# 使い方例
if __name__ == "__main__":
    # 例: expected_distanceをロード
    # expected_distance = np.load('expected_distance.npy')
    # 仮の例
    agents_count = 7
    np.random.seed(0)
    expected_distance = np.random.rand(agents_count, agents_count)
    # expected_distance = (expected_distance + expected_distance.T) / 2  # 対称化
    np.fill_diagonal(expected_distance, 0)
    test_agent_distance_correlation(expected_distance)
