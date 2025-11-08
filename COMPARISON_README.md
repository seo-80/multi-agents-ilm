# 記号解と乱数シミュレーションの比較ガイド

## 概要

`compare_symbolic_simulation.py`は、IBD解析の記号解（理論値）と乱数シミュレーション（モンテカルロ法）の結果を比較するためのスクリプトです。

## ファイルと生成物の対応

### 記号解（理論値）
- **実行コマンド**:
  ```bash
  python IBD_analysis/analyze_concentric_parameter_space.py \
    --M 5 --cases case1 case2 case3 case4 --precompute-symbolic --no-plots
  ```
- **生成ファイル**:
  - `IBD_analysis/results/M{M}_{case}.pkl` (例: `M5_case1.pkl`)
  - 平衡状態のF行列（理論的な定常分布）

### 乱数シミュレーション
- **実行コマンド**:
  ```bash
  # 1. シミュレーション実行
  ./run_naive_simulations_parallel.sh -p Ni -m 10000000

  # 2. プロット生成（平均F行列を計算）
  ./run_naive_simulations_parallel.sh -P
  ```
- **生成ファイル**:
  - 時系列データ: `{DATA_RAW_DIR}/{subdir}/f_matrix_{save_idx}.npy`
  - **平均F行列**: `{DATA_RAW_DIR}/{subdir}/mean_f_matrix.npy`
  - その他: 距離行列、類似度行列、統計情報など

## Case定義とパラメータの対応

| Case | center_prestige | centralized_neologism | flow_type | nonzero_alpha | 説明 |
|------|----------------|----------------------|-----------|---------------|------|
| case1 | True | True | outward | center | 中心から外向き + 中心のみ新語生成 |
| case2 | True | False | outward | evenly | 中心から外向き + 全員が新語生成 |
| case3 | False | True | bidirectional | center | 双方向 + 中心のみ新語生成 |
| case4 | False | False | bidirectional | evenly | 双方向 + 全員が新語生成 |

## 使用方法

### 基本的な使い方

```bash
# 単一caseの比較
python compare_symbolic_simulation.py --M 5 --cases case1

# 複数caseの比較
python compare_symbolic_simulation.py --M 5 --cases case1 case2 case3 case4

# カスタムパラメータ
python compare_symbolic_simulation.py --M 7 --cases case1 \
  --m 0.01 --alpha 0.001 --N 99
```

### コマンドライン引数

#### 必須引数
- `--M`: エージェント数（奇数、記号解と一致する必要あり）
- `--cases`: 比較するcase（case1, case2, case3, case4から選択）

#### パラメータ引数（デフォルト値あり）
- `--m`, `--coupling-strength`: 結合強度（デフォルト: 0.01）
- `--alpha`, `--alpha-per-data`: データあたりのalpha（デフォルト: 0.001）
- `--N`, `--N-i`: エージェントあたりのデータ数（デフォルト: 99）

#### ディレクトリ引数
- `--symbolic-dir`: 記号解の結果ディレクトリ（デフォルト: `IBD_analysis/results`）
- `--data-dir`: シミュレーションデータディレクトリ（デフォルト: `config.py`の設定）
- `--output-dir`: 比較結果の出力ディレクトリ（デフォルト: `comparison_results`）

#### その他
- `--verbose`: 詳細な出力

### ヘルプ表示

```bash
python compare_symbolic_simulation.py --help
```

## 出力ファイル

### 1. 個別比較プロット
`comparison_results/comparison_M{M}_{case}.png`

各caseごとに8つのサブプロットを含む総合的な比較図：
1. **記号解のF行列ヒートマップ**
2. **シミュレーションのF行列ヒートマップ**
3. **絶対誤差ヒートマップ**: |Sim - Symbolic|
4. **相対誤差ヒートマップ**: |Sim - Symbolic| / Symbolic
5. **散布図**: シミュレーション vs 記号解（相関係数付き）
6. **誤差分布ヒストグラム**
7. **中心エージェントからの距離**: 記号解とシミュレーションの比較
8. **統計サマリー**: MAE, RMSE, 相関係数など

### 2. サマリーテーブル
`comparison_results/comparison_summary.csv`

全caseの比較統計を含むCSVファイル：
- MAE (Mean Absolute Error): 平均絶対誤差
- Max AE (Maximum Absolute Error): 最大絶対誤差
- RMSE (Root Mean Square Error): 二乗平均平方根誤差
- Mean/Max Relative Error: 平均/最大相対誤差
- Correlation: 相関係数

## ワークフロー例

### 例1: M=5, 全caseの完全な比較

```bash
# ステップ1: 記号解の計算（まだ実行していない場合）
python IBD_analysis/analyze_concentric_parameter_space.py \
  --M 5 \
  --cases case1 case2 case3 case4 \
  --precompute-symbolic \
  --no-plots \
  --verbose

# ステップ2: シミュレーションの実行（長時間かかる場合あり）
# デフォルトパラメータ: M=7, N_i=99, m=0.01, alpha_per_data=0.001
# run_naive_simulations_parallel.shを編集してAGENTS_COUNT=5に変更

# ステップ3: シミュレーション結果から平均F行列を計算
./run_naive_simulations_parallel.sh -P

# ステップ4: 比較実行
python compare_symbolic_simulation.py \
  --M 5 \
  --cases case1 case2 case3 case4 \
  --m 0.01 \
  --alpha 0.001 \
  --N 99 \
  --verbose
```

### 例2: 既存のシミュレーション結果との比較

```bash
# シミュレーションデータが既に存在する場合
python compare_symbolic_simulation.py \
  --M 7 \
  --cases case1 \
  --m 0.01 \
  --alpha 0.001 \
  --N 99 \
  --data-dir data/raw \
  --output-dir my_comparison
```

## 注意事項

### 1. パラメータの一致
記号解とシミュレーションで以下のパラメータが一致している必要があります：
- M (エージェント数)
- m (結合強度)
- alpha (新語生成率)
- N (データ数)
- case設定（flow_typeとnonzero_alphaの組み合わせ）

### 2. シミュレーションの平衡状態
シミュレーションが十分に長い時間実行され、平衡状態に達している必要があります。
- 推奨: `--max_t 10000000`以上
- `mean_f_matrix.npy`が時系列の平均として計算されている

### 3. Alpha値の解釈
- **centralized (center)**: 記号解では中心エージェントの総alpha = `alpha_per_data * N_i`
- **evenly distributed**: 記号解では各エージェントのalpha = `alpha_per_data * N_i`

スクリプトは自動的にこの変換を行います。

### 4. ファイルパスの問題
シミュレーションディレクトリ名は以下の形式である必要があります：
```
{flow_type}_flow-nonzero_alpha_{nonzero_alpha}_fr_{m}_agents_{M}_N_i_{N}_alpha_{alpha_per_data}
```

例:
```
outward_flow-nonzero_alpha_center_fr_0.01_agents_5_N_i_99_alpha_0.001
```

## トラブルシューティング

### エラー: Symbolic solution not found

**原因**: 記号解がまだ計算されていない

**解決策**:
```bash
python IBD_analysis/analyze_concentric_parameter_space.py \
  --M <M値> --cases <case> --precompute-symbolic --no-plots
```

### エラー: Simulation F-matrix not found

**原因1**: シミュレーションが実行されていない

**解決策**:
```bash
./run_naive_simulations_parallel.sh -p Ni -m 10000000
```

**原因2**: `mean_f_matrix.npy`が生成されていない

**解決策**:
```bash
./run_naive_simulations_parallel.sh -P
```

または、個別のシミュレーションディレクトリで：
```bash
python src/naive_simulation.py --recompute_distance \
  --agents_count 5 --N_i 99 --coupling_strength 0.01 \
  --alpha_per_data 0.001 --flow_type outward --nonzero_alpha center
```

### エラー: Dimension mismatch

**原因**: 記号解とシミュレーションのMが一致していない

**解決策**: `--M`引数とシミュレーションの`agents_count`を一致させる

## 期待される結果

良好な一致を示す指標：
- **相関係数 > 0.99**: 記号解とシミュレーションの強い相関
- **平均相対誤差 < 5%**: 十分な精度
- **RMSE < 0.01**: F行列の要素が[0,1]の範囲であることを考慮

誤差の要因：
1. シミュレーションの有限時間効果（平衡に未到達）
2. モンテカルロノイズ
3. 数値計算の精度

## スクリプトの仕組み

1. **パラメータマッピング**: case名を(flow_type, nonzero_alpha)に変換
2. **記号解の読み込み**: pickle fileから記号行列を読み込み、数値化
3. **シミュレーション結果の読み込み**: `mean_f_matrix.npy`を読み込み
4. **比較**: 要素ごとの誤差、統計量を計算
5. **可視化**: 包括的な比較プロットを生成
6. **サマリー**: 全caseの比較表を作成

## 関連ファイル

- `compare_symbolic_simulation.py`: 比較スクリプト本体
- `IBD_analysis/src/f_matrix_symbolic.py`: 記号解の計算とロード
- `src/naive_simulation.py`: モンテカルロシミュレーション
- `src/plotnaive_simulation.py`: 平均F行列の計算
- `run_naive_simulations_parallel.sh`: 並列シミュレーション実行スクリプト
- `config.py`: データディレクトリの設定
