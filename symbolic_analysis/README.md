# M-Agent Infinite Alleles Model - Symbolic Analysis

記号計算によるM個エージェントの無限変異モデルの定常状態計算。

## ファイル構成

```
symbolic_analysis/
├── src/
│   └── m_agent_stationary_symbolic.py  # メインプログラム
├── results/                             # 計算結果の出力先
│   ├── M3_case1.pkl                     # Pickle形式（Python再利用可能）
│   ├── M3_case1.md                      # Markdown形式（人間可読）
│   └── ...
├── test_m_agent_stationary.py          # 全4ケースのテストスクリプト
├── test_save_load.py                   # Save/Load機能のテスト
└── example_load_results.py             # 結果のロード・分析例
```

## 4つのモデルケース

| ケース | center_prestige | centralized_neologism_creation | 説明 |
|--------|----------------|-------------------------------|------|
| case1 | False | False | 対称的双方向相互作用 + 均等な革新生成 |
| case2 | True  | False | 中心から外向き相互作用 + 均等な革新生成 |
| case3 | False | True  | 対称的双方向相互作用 + 中心のみ革新生成 |
| case4 | True  | True  | 中心から外向き相互作用 + 中心のみ革新生成 |

## 使い方

### 1. 基本的な計算実行

```bash
# 全4ケースを計算
python symbolic_analysis/test_m_agent_stationary.py

# 特定のケースのみ計算（1-4）
python symbolic_analysis/test_m_agent_stationary.py 1

# カスタムパラメータで計算
python symbolic_analysis/test_m_agent_stationary.py 5 true false  # M=5, case2
```

### 2. Pythonから直接呼び出し

```python
from symbolic_analysis.src.m_agent_stationary_symbolic import compute_stationary_state_symbolic

# Case 1の計算
states, pi, P, W = compute_stationary_state_symbolic(
    M=3,
    center_prestige=False,
    centralized_neologism_creation=False,
    output_dir="results"
)

# 結果の表示
for i, state in enumerate(states):
    print(f"State {i+1}: {partition_to_string(state)}")
    print(f"  Probability: {simplify(pi[i])}")
```

### 3. 計算結果のロード

計算結果はPickle形式 (`.pkl`) で保存されており、後から簡単にロードできます：

```python
from symbolic_analysis.src.m_agent_stationary_symbolic import load_results_by_case

# ケース名でロード
results = load_results_by_case(M=3, case_name="case1", output_dir="results")

# 結果へのアクセス
metadata = results['metadata']        # メタデータ
states = results['states']            # 状態リスト
W = results['W']                      # 重み行列
alpha_vec = results['alpha_vec']      # 革新パラメータベクトル
mu_vec = results['mu_vec']            # 変異率ベクトル
P = results['P']                      # 遷移確率行列
pi = results['pi']                    # 定常分布
```

### 4. 数値評価

記号式に具体的な数値を代入して評価：

```python
from sympy import symbols

# 結果をロード
results = load_results_by_case(3, "case1")

# パラメータ値を設定
m, alpha = symbols('m alpha')
m_val = 0.3
alpha_val = 0.1

# 定常分布の数値評価
for i, pi_expr in enumerate(results['pi']):
    pi_num = float(pi_expr.subs({m: m_val, alpha: alpha_val}))
    print(f"State {i+1}: {pi_num:.6f}")
```

## 出力形式

### Pickle形式 (`.pkl`)

Python辞書として以下の構造で保存：

```python
{
    'metadata': {
        'M': int,                              # エージェント数
        'center_prestige': bool,               # 中心prestige条件
        'centralized_neologism_creation': bool,# 中心集中的新変異生成
        'case_name': str,                      # ケース名 ("case1"-"case4")
        'timestamp': str,                      # 計算日時
    },
    'states': List[FrozenSet[FrozenSet[int]]],  # 状態リスト
    'W': sympy.Matrix,                          # 重み行列（記号式）
    'alpha_vec': List[sympy.Expr],              # 革新パラメータベクトル
    'mu_vec': List[sympy.Expr],                 # 変異率ベクトル
    'P': sympy.Matrix,                          # 遷移確率行列（記号式）
    'pi': sympy.Matrix,                         # 定常分布（記号式）
}
```

### Markdown形式 (`.md`)

人間が読みやすい形式で以下の情報を含む：
- パラメータ一覧
- 記号変数の説明
- 重み行列W
- 革新パラメータと変異率
- 状態空間
- 遷移確率行列P
- 定常分布（各状態の確率）

## 関数リファレンス

### メイン関数

#### `compute_stationary_state_symbolic(M, center_prestige, centralized_neologism_creation, output_dir)`

定常状態を記号計算。

**引数:**
- `M` (int): エージェント数（奇数、デフォルト: 3）
- `center_prestige` (bool): 中心prestige条件（デフォルト: False）
- `centralized_neologism_creation` (bool): 中心集中的新変異生成（デフォルト: False）
- `output_dir` (str): 出力ディレクトリ（デフォルト: "results"）

**戻り値:**
- `states`: 状態のリスト
- `pi`: 定常分布（記号式）
- `P`: 遷移確率行列（記号式）
- `W`: 重み行列（記号式）

### ロード関数

#### `load_results(filepath)`

Pickleファイルから結果をロード。

**引数:**
- `filepath` (str): Pickleファイルのパス

**戻り値:**
- 結果の辞書

#### `load_results_by_case(M, case_name, output_dir)`

M とケース名から結果をロード。

**引数:**
- `M` (int): エージェント数
- `case_name` (str): ケース名 ("case1", "case2", "case3", "case4")
- `output_dir` (str): 結果ディレクトリ（デフォルト: "results"）

**戻り値:**
- 結果の辞書

### 保存関数

#### `save_results(M, center_prestige, centralized_neologism_creation, states, W, alpha_vec, mu_vec, P, pi, output_dir)`

計算結果をPickle形式で保存。

**注:** 通常は`compute_stationary_state_symbolic()`が自動的に呼び出します。

## 使用例スクリプト

### `example_load_results.py`

5つの分析例を実装：
1. ファイルパスでロード
2. ケース名でロード
3. 数値代入による評価
4. 複数ケースの比較
5. 行列とパラメータへのアクセス

実行方法：
```bash
python symbolic_analysis/example_load_results.py
```

## 分析のための次のステップ

保存された結果を使って、以下のような分析が可能です：

### 1. パラメータ感度分析

```python
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols

# 結果をロード
results = load_results_by_case(3, "case1")
m, alpha = symbols('m alpha')

# パラメータ範囲
m_vals = np.linspace(0.1, 0.9, 20)
alpha_vals = np.linspace(0.1, 2.0, 20)

# 特定状態の確率を計算
state_idx = 0  # 全同期状態など
probs = []
for m_val in m_vals:
    for alpha_val in alpha_vals:
        prob = float(results['pi'][state_idx].subs({m: m_val, alpha: alpha_val}))
        probs.append(prob)

# ヒートマップなどで可視化
```

### 2. ケース間比較

```python
# 複数ケースの結果をロード
cases = ["case1", "case2", "case3", "case4"]
all_results = {c: load_results_by_case(3, c) for c in cases}

# 各ケースで同じパラメータ値での定常分布を比較
m_val, alpha_val = 0.3, 0.1
for case, results in all_results.items():
    print(f"\n{case}:")
    for i, pi_expr in enumerate(results['pi']):
        prob = float(pi_expr.subs({m: m_val, alpha: alpha_val}))
        print(f"  State {i+1}: {prob:.4f}")
```

### 3. 解析的性質の研究

```python
# 特定状態の定常確率の記号式を調査
results = load_results_by_case(3, "case1")
pi_sync = results['pi'][0]  # 全同期状態

# 極限の調査
from sympy import limit, oo
print("alpha -> 0:", limit(pi_sync, alpha, 0))
print("alpha -> ∞:", limit(pi_sync, alpha, oo))
print("m -> 0:", limit(pi_sync, m, 0))
print("m -> 1:", limit(pi_sync, m, 1))
```

## 計算量について

- **状態数**: ベル数B_M（M=3: 5状態、M=5: 52状態、M=7: 877状態）
- **計算時間**: M=3で数分、M=5で数十分以上かかる可能性
- **記号式の複雑さ**: Mが大きいと式が非常に複雑になる

大きなMでは計算に時間がかかるため、バックグラウンドで実行することを推奨。

## トラブルシューティング

### 計算が遅い

記号計算の固有値問題は計算量が大きいです。M=3でも数分かかります。

### メモリ不足

M が大きい場合、メモリ不足になる可能性があります。小さいMから始めてください。

### ファイルが見つからない

結果をロードする前に、まず計算を実行してください：
```bash
python symbolic_analysis/test_m_agent_stationary.py
```

## 参考

モデルの詳細は仕様書を参照してください。
