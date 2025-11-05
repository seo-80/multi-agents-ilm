# Distance Calculation Discrepancy Analysis

## 概要

`src/naive_simulation.py` の数値シミュレーションと `symbolic_analysis/` の代数的計算において、distanceの計算結果が一致しない問題を調査しました。

## 結論

**両者のモデル実装は本質的に同じですが、距離の定義に違いがあります。**

## 詳細分析

### 1. モデル実装の比較

#### 1.1 ネットワーク行列（W行列）

**✓ 一致しています**

- **bidirectional_flow**: `networks.py` の bidirectional_flow_rate と symbolic_analysis の symmetric bidirectional は同一
- **outward_flow**: `networks.py` の outward_flow_rate と symbolic_analysis の center-prestige は同一
- 端点の調整も両方で正しく実装されている

#### 1.2 突然変異モデル

**✓ 一致しています**

両方とも **parent-side mutation** を実装：
- `naive_simulation.py`: コピー元エージェント `j_idx` の `mu_j` で突然変異判定
- `symbolic_analysis`: `P(i -> C_l) = Σ_{j∈C_l} W_{ij} × (1 - μ_j)`

#### 1.3 突然変異率の計算

**✓ N_i=1 の場合は一致します**

- `naive_simulation.py`: `mu = alpha / (N_i + alpha)`
- `symbolic_analysis`: `μ_i = α_i / (1 + α_i)`  (N_i=1固定)

N_i=1 のとき: `mu = alpha / (1 + alpha)` となり、完全に一致。

#### 1.4 alphaパラメータの設定

**✓ 一致しています**

- `nonzero_alpha="center"` ⟷ `centralized_neologism_creation=True`
- `nonzero_alpha="evenly"` ⟷ `centralized_neologism_creation=False`
- 中心エージェントのインデックスも一致（M=15の場合: 両方とも7）

### 2. 距離計算の違い

#### 2.1 naive_simulation.py のマンハッタン距離

```python
def calc_agent_distance(state):
    counters = [collections.Counter([tuple(d) for d in state[i]]) for i in range(agents_count)]
    for i in range(agents_count):
        for j in range(i, agents_count):
            all_keys = set(counters[i]) | set(counters[j])
            dist_ij = sum(abs(counters[i][k] - counters[j][k]) for k in all_keys)
```

**N_i=1 の場合の具体例:**

- エージェントiとjが**同じデータ**を持つ場合:
  - `counters[i] = {(t, a, b): 1}`
  - `counters[j] = {(t, a, b): 1}`
  - `dist_ij = |1 - 1| = 0` ✓

- エージェントiとjが**異なるデータ**を持つ場合:
  - `counters[i] = {(t1, a1, b1): 1}`
  - `counters[j] = {(t2, a2, b2): 1}`
  - `all_keys = {(t1, a1, b1), (t2, a2, b2)}`
  - `dist_ij = |1 - 0| + |0 - 1| = 2` ⚠️

#### 2.2 symbolic_analysis の距離

```python
def compute_distance_expectations(states, pi, M):
    for state_idx, state in enumerate(states):
        if agents_in_same_block(i, j, state):
            pass  # Distance is 0
        else:
            expected_dist += pi[state_idx]  # Distance is 1
```

- 同じブロック（同じデータ）: `d_ij = 0`
- 異なるブロック（異なるデータ）: `d_ij = 1` ⚠️

### 3. 不一致の原因

**N_i=1 の場合でも、マンハッタン距離は2倍になっています！**

```
naive_simulation の距離 = 2 × symbolic_analysis の距離
```

理由: マンハッタン距離では、異なる要素 `k` について両方向（`counters[i][k]` と `counters[j][k]`）でカウントするため。

### 4. 解決方法

以下のいずれかの修正が必要です：

#### オプション1: naive_simulation.py の距離を2で割る

```python
def calc_agent_distance(state):
    # ... 既存のコード ...
    dist_ij = sum(abs(counters[i][k] - counters[j][k]) for k in all_keys)
    dist_ij = dist_ij // 2  # N_i=1の場合の補正
    dist[i, j] = dist[j, i] = dist_ij
```

#### オプション2: 異なるデータの個数をカウントする

```python
def calc_agent_distance_normalized(state):
    """
    N_i=1用の距離計算: 0 (同じ) または 1 (異なる)
    """
    agents_count, N_i, _ = state.shape
    assert N_i == 1, "This function is only for N_i=1"

    dist = np.zeros((agents_count, agents_count), dtype=int)
    for i in range(agents_count):
        for j in range(i, agents_count):
            # データが同じかチェック
            if np.array_equal(state[i, 0], state[j, 0]):
                dist_ij = 0
            else:
                dist_ij = 1
            dist[i, j] = dist[j, i] = dist_ij
    return dist
```

#### オプション3: symbolic_analysis の定義を変更

symbolic_analysis側で距離を2倍にする（非推奨、理論的に不自然）

## 推奨事項

**オプション1またはオプション2を推奨します。**

- オプション1: 最小限の変更で済む
- オプション2: より明示的で、N_i=1専用として意図が明確

N_i > 1 の一般的な場合は、現在のマンハッタン距離の定義が適切です。N_i=1 の特殊ケースで symbolic_analysis と比較する場合のみ、補正が必要です。

## 検証

修正後、以下を確認してください：

1. N_i=1 でシミュレーションを実行
2. 距離の時間平均を計算
3. symbolic_analysis の期待値と比較
4. 両者が（ほぼ）一致することを確認

## その他の注意点

モデル実装そのものは正しいため、distance以外の統計量（例: ブロックサイズ分布、状態分布など）は、十分長いシミュレーションで一致するはずです。
