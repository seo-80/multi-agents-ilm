# IBD Analysis - F-Matrix Symbolic Computation

このディレクトリは、IBD（Identity By Descent）モデルにおけるF行列の定常解を**記号的に**求めるコード群です。

## 📐 数理モデル

### F行列の更新式

定常状態 $F(t+1) = F(t) = F$ において：

**非対角要素** ($i \neq j$):
$$F_{ij} = \sum_{k}\sum_{l}W_{ik}W_{jl}(1-\mu_k)(1-\mu_l)F_{kl}$$

**対角要素** ($i = j$):
$$F_{ii} = \frac{1}{N}+\left(1-\frac{1}{N}\right)\sum_{k}\sum_{l}W_{ik}W_{il}(1-\mu_k)(1-\mu_l)F_{kl}$$

### パラメータ

- **$N$**: 人口サイズ（各エージェント共通）
- **$m$**: coupling strength（移住率）
- **$\alpha$**: innovation parameter
- **$\mu_i$**: 突然変異率 = $\alpha_i / (N + \alpha_i)$
- **$W$**: 移住行列（mating matrix）

## 🎯 実装した4つのケース

| Case | center_prestige | centralized_neologism_creation | W の構造 | α の分布 |
|------|----------------|-------------------------------|---------|---------|
| **case1** | False | False | 対称bidirectional | 全員α (evenly) |
| **case2** | True | False | 中心→外 非対称 | 全員α (evenly) |
| **case3** | False | True | 対称bidirectional | 中心のみα (center) |
| **case4** | True | True | 中心→外 非対称 | 中心のみα (center) |

## 📁 ディレクトリ構成

```
IBD_analysis/
├── src/
│   ├── __init__.py
│   └── f_matrix_symbolic.py          # メインモジュール
├── results/
│   ├── M3_case1.pkl                   # 計算結果（M=3, case1）
│   ├── M3_case1.md                    # 結果の可読形式
│   ├── M3_case2.pkl/md
│   ├── M3_case3.pkl/md
│   └── M3_case4.pkl/md
├── verify_f_matrix_numerically.py     # 数値検証スクリプト
└── README.md                          # このファイル
```

## 🚀 使い方

### 1. F行列の記号解を計算

```bash
# M=3で4ケース全て計算
python -m IBD_analysis.src.f_matrix_symbolic --M 3 --cases case1 case2 case3 case4

# M=3のcase1のみ計算
python -m IBD_analysis.src.f_matrix_symbolic --M 3 --cases case1
```

**出力:**
- `results/M3_case1.pkl`: 記号的F行列と関連情報（pickle形式）
- `results/M3_case1.md`: 可読形式の結果（Markdown）

### 2. 記号解の数値検証

記号解が正しいか、数値計算と比較して検証します：

```bash
# M=3の全ケースを検証
python IBD_analysis/verify_f_matrix_numerically.py --M 3 --cases case1 case2 case3 case4

# パラメータを指定して検証
python IBD_analysis/verify_f_matrix_numerically.py \
    --M 3 \
    --cases case1 \
    --N 100 \
    --m 0.01 \
    --alpha 0.001
```

**検証結果（M=3全ケース）:**
```
M3_case1  ✓ PASS  (max diff 4.49e-08)
M3_case2  ✓ PASS  (max diff 9.80e-09)
M3_case3  ✓ PASS  (max diff 4.52e-08)
M3_case4  ✓ PASS  (max diff 9.81e-09)
```

全て成功！記号解と数値解の差は $10^{-8}$ オーダーで一致。

### 3. 結果の読み込みと利用

```python
from IBD_analysis.src.f_matrix_symbolic import load_results_by_case
from sympy import symbols, lambdify
import numpy as np

# M=3, case1の結果を読み込み
results = load_results_by_case(M=3, case_name='case1')
F_symbolic = results['F_matrix']

# 記号変数
N, m, alpha = symbols('N m alpha')

# 具体的な値で評価
N_val, m_val, alpha_val = 100, 0.01, 0.001

# F[0,0]を評価
f_00_expr = F_symbolic[0, 0]
f_00_func = lambdify((N, m, alpha), f_00_expr, 'numpy')
f_00_value = f_00_func(N_val, m_val, alpha_val)

print(f"F[0,0] = {f_00_value}")  # 出力: F[0,0] = 0.9945042769...
```

## 📊 計算結果の例

### M=3, case1の例

**W行列:**
```
W = | 1-m/2   m/2     0    |
    | m/2     1-m     m/2  |
    | 0       m/2     1-m/2|
```

**F行列:**

各要素 $F_{ij}$ は $N, m, \alpha$ の有理式として得られます：

```
F_{1,1} = (9 N^8 m^5 - 48 N^8 m^4 + 48 N^8 m^3 + ...) / (N * (...))
F_{1,2} = (N m (9 N^6 m^4 - 48 N^6 m^3 + ...)) / (...)
...
```

詳細は `results/M3_case1.md` を参照。

### 4. 結果のキャッシング

同じファイルを複数回ロードする際、自動的にメモリにキャッシュされ、ファイルI/Oを削減します：

```python
from IBD_analysis.src.f_matrix_symbolic import load_results_by_case, clear_results_cache

# 初回ロード - ファイルから読み込み
results1 = load_results_by_case(M=3, case_name='case1')
# 出力: "Results loaded from: .../M3_case1.pkl"

# 2回目 - キャッシュから取得（高速）
results2 = load_results_by_case(M=3, case_name='case1')
# 出力: "Results loaded from cache: M=3, case=case1"

# パラメータ掃引の例（効率的！）
for N in [10, 50, 100, 200, 500]:
    results = load_results_by_case(M=3, case_name='case1')  # 1回だけファイルI/O
    F = results['F_matrix']
    # N の値で評価...

# メモリを解放する必要がある場合
clear_results_cache()
```

**キャッシュの効果:**
- ファイルI/O回数を大幅削減（例: 20回 → 4回、80%削減）
- 大きなファイル（M=5以上）で特に効果的
- 同一プロセス内での複数回ロードを自動最適化

## 🔬 実装の詳細

### 対称性の利用

計算量削減のため、F行列の対称性を利用：

- **M=3, case1（対称）**: 9要素 → 4独立変数に削減
- **M=3, case2（非対称）**: 9要素 → 9独立変数（対称性なし）

### 計算量

- **M=3**: 全4ケースで数秒〜数十秒で完了
- **M=5**: SymPyの連立方程式求解に10分以上かかる（タイムアウト）

→ より大きな$M$には数値的手法を推奨

## ✅ 検証

`verify_f_matrix_numerically.py` で以下を確認：

1. **記号解の評価**: $F(N, m, \alpha)$ に具体値を代入
2. **数値解の計算**: 反復法で定常状態を数値計算
3. **比較**: 両者の差が $10^{-6}$ 以下なら成功

全ケースで検証成功（差は $10^{-8}$ オーダー）。

## 🆚 既存実装との違い

| 項目 | `probability_of_identity.py` | `IBD_analysis/` |
|------|------------------------------|----------------|
| **手法** | 数値反復 | **記号計算** |
| **出力** | 数値の行列 | **$F_{ij}(N, m, \alpha)$ の陽な式** |
| **対角要素** | ドリフト項あり | **新しい定義式** |
| **$N$の扱い** | ベクトル $N_i$ | スカラー $N$（全員同一） |

## 📝 結果ファイルの構造

### Pickle (.pkl)

```python
{
    'metadata': {
        'M': int,
        'center_prestige': bool,
        'centralized_neologism_creation': bool,
        'case_name': str,
        'W': sympy.Matrix,           # 移住行列（記号）
        'alpha_vec': list,           # αベクトル（記号）
        'mu_vec': list,              # μベクトル（記号）
        'timestamp': str,
    },
    'F_matrix': sympy.Matrix         # F行列（記号）
}
```

### Markdown (.md)

- パラメータ説明
- ケース説明
- W行列（LaTeX）
- F行列（LaTeX）
- 各要素の陽な式

## 🔗 関連ファイル

- `../symbolic_analysis/`: 既存の記号計算フレームワーク（定常分布など）
- `../src/probability_of_identity.py`: 既存の数値計算実装
- `../src/ilm/networks.py`: W行列の生成

## 📚 参考

この実装は以下の論文・モデルに基づいています：

- IBD (Identity By Descent) モデル
- 有限島モデル（finite island model）
- Moran model with migration

## 🎓 使用例

### 例1: F[0,1]のα依存性を可視化

```python
from IBD_analysis.src.f_matrix_symbolic import load_results_by_case
from sympy import symbols, lambdify
import numpy as np
import matplotlib.pyplot as plt

results = load_results_by_case(M=3, case_name='case1')
F = results['F_matrix']

N, m, alpha = symbols('N m alpha')
f_01_expr = F[0, 1]
f_01_func = lambdify((N, m, alpha), f_01_expr, 'numpy')

N_val, m_val = 100, 0.01
alpha_vals = np.logspace(-4, -1, 100)
f_01_vals = [f_01_func(N_val, m_val, a) for a in alpha_vals]

plt.semilogx(alpha_vals, f_01_vals)
plt.xlabel('α')
plt.ylabel('F[0,1]')
plt.title('F[0,1] vs α (N=100, m=0.01)')
plt.show()
```

### 例2: ケース間の比較

```python
for case_name in ['case1', 'case2', 'case3', 'case4']:
    results = load_results_by_case(M=3, case_name=case_name)
    F = results['F_matrix']
    # 具体値で評価して比較...
```

## 🐛 トラブルシューティング

### Q: "No module named 'sympy'" エラー

```bash
pip install sympy numpy matplotlib
```

### Q: M=5以上でタイムアウト

→ 変数が増えるとSymPyの求解に時間がかかります。M=3までの使用を推奨。

### Q: 検証が失敗する

→ 数値計算の収束判定を確認してください（`tol=1e-10`, `max_iter=10000`）。

## 📄 ライセンス

プロジェクトのライセンスに従います。

## 👤 作成者

Claude Code による自動実装（2025-11-06）
