# 言語進化シミュレーションワークフロー仕様書

## 概要

この仕様書は、言語進化のナイーブシミュレーション（naive simulation）から図表生成まで、一連のワークフローを包括的に記述します。ワークフローは3つの主要段階で構成されます：
1. シミュレーション実行
2. 図表作成
3. 図表のまとめ

## 1. シミュレーション実行段階

### 1.1 実行スクリプト: `run_naive_simulations_parallel.sh`

**目的**: 複数パラメータの組み合わせで並列シミュレーションを実行

**主要機能**:
- Task Spooler (tsp) を使用した並列ジョブ管理
- パラメータスイープ（coupling_strength、alpha_per_data、N_i）
- 既存結果の再開機能（--resume）
- プロット専用モード（-P）

**主要パラメータ**:
```bash
# デフォルト設定
AGENTS_COUNT=7                    # エージェント数
MAX_T=10000000                   # 最大タイムステップ
COUPLING_STRENGTHS=(0.01)        # 結合強度
ALPHA_PER_DATA_LIST=(0.001)      # 新語生成バイアス
N_I_LIST=(100)                   # サブ集団サイズ
FLOW_TYPES=("outward" "bidirectional")  # フロータイプ
NONZERO_ALPHAS=("center" "evenly")      # バイアス分布
```

**パラメータスイープ設定**:
- `strength`: [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
- `alpha`: [0.00025, 0.0005, 0.001, 0.002, 0.004]
- `Ni`: [1, 25, 50, 100, 200, 400, 800]

**使用例**:
```bash
# 全パラメータスイープ実行
./run_naive_simulations_parallel.sh

# 特定パラメータのみ
./run_naive_simulations_parallel.sh -p strength

# プロットモード
./run_naive_simulations_parallel.sh -P
```

### 1.2 シミュレーション本体: `src/naive_simulation.py`

**目的**: 言語進化の確率的シミュレーション実行

**主要アルゴリズム**:
1. **ネットワーク生成**: 指定されたフロータイプ（双方向/外向き）でエージェント間結合を作成
2. **初期状態設定**: 各エージェントにN_i個のデータ（ミーム）を初期化
3. **突然変異率計算**: `mu = alphas / (N_i + alphas)`
4. **メインループ**: タイムステップごとにデータ伝播と突然変異を実行

**状態データ構造**:
```python
state[i, j] = [タイムステップ, 発生エージェント番号, データ番号]
# 形状: (agents_count, N_i, 3)
```

**主要処理**:
- **データフロー生成**: `networks.generate_data_flow_count()`で各エージェント間の伝播数を決定
- **突然変異判定**: 各受信データに対して確率的に突然変異を実行
- **距離計算**: エージェント間のデータ多重集合距離を計算
- **保存処理**: 定期的にstate、distance、random_stateを保存

**出力ファイル**:
- `state_{idx}.npy`: 各タイムステップでのエージェント状態
- `distance_{idx}.npy`: エージェント間距離行列
- `random_state_{idx}.pkl`: 乱数状態（再現性のため）
- `save_idx_t_map.csv`: 保存インデックスと実時間の対応表

**ディレクトリ構造**:
```
data/naive_simulation/raw/{flow_prefix}nonzero_alpha_{na}_fr_{strength}_agents_{agents}_N_i_{Ni}_alpha_{alpha}/
```

## 2. 図表作成段階

### 2.1 プロットスクリプト: `src/plotnaive_simulation.py`

**目的**: シミュレーション結果から多様な分析図表を生成

**主要解析機能**:

#### 2.1.1 距離解析（--plot_distance）
- **平均距離ヒートマップ**: エージェント間の平均距離を色分け表示
- **距離ランク行列**: 各エージェントから見た他エージェントとの距離順位
- **同心円分布検出**: 中心エージェントを基準とした同心円状分布の検出
- **特定ペア比較**: エージェント0と中心/対照エージェント間の距離分布

#### 2.1.2 年齢解析（--plot_age）
- **単語年齢統計**: 各エージェントが保持する単語の平均年齢
- **年齢分散**: エージェント間での年齢ばらつき分析

#### 2.1.3 類似度解析（--plot_similarity）
- **ドット積類似度**: ミーム頻度ベクトルの内積
- **コサイン類似度**: 正規化されたミーム頻度ベクトル間の角度
- **類似度ヒートマップ**: エージェント間類似度の可視化

#### 2.1.4 統計検定
- **二項検定**: 中心vs対照エージェントとの距離/類似度比較
- **ロジスティック回帰**: パラメータが同心円分布に与える影響分析
- **ペアワイズ回帰**: 全エージェントペア組み合わせでの回帰分析

**主要出力図表**:
- `mean_distance_heatmap_Blues.png`: 平均距離ヒートマップ
- `distance_rank_matrix_heatmap_Blues_with_border.png`: 距離ランク行列
- `agent_pair_distances_histogram.png`: ペア距離分布
- `mean_dot_similarity_heatmap.png`: ドット積類似度ヒートマップ
- `age_mean_timeavg_per_agent.png`: エージェント別平均年齢

#### 2.1.5 コマンドライン引数例
```bash
# 距離解析実行
python src/plotnaive_simulation.py --plot_distance --skip 1000

# 全解析実行
python src/plotnaive_simulation.py --plot_distance --plot_age --plot_similarity --check_concentric

# 特定条件のみ
python src/plotnaive_simulation.py --nonzero_alpha center --flow_type outward --plot_distance
```

## 3. 図表まとめ段階

### 3.1 図表結合スクリプト: `src/stitch_figures.py`

**目的**: パラメータスイープ結果を単一の比較画像に結合

**主要機能**:
- **パラメータ軸設定**: strength/alpha/Ni のいずれかを横軸に設定
- **条件軸設定**: (nonzero_alpha, flow_type) の組み合わせを縦軸に配置
- **グリッドレイアウト**: 4行×N列の比較グリッド生成
- **ラベル表示**: 各セルにパラメータ値や条件を表示

**グリッド構造**:
```
行順序（固定）:
1. (evenly, bidirectional)
2. (evenly, outward)
3. (center, bidirectional)
4. (center, outward)

列: 選択されたパラメータの値（strength/alpha/Ni）
```

**主要パラメータ**:
- `--param_to_change`: 横軸にするパラメータ (strength/alpha/Ni/flow_center)
- `--figure_name`: 結合対象の図表ファイル名
- `--hide_outer_labels`: 外側ラベルの非表示
- `--overlay_on_tile`: 各セルにテキストオーバーレイ

**使用例**:
```bash
# 結合強度スイープの結合
python src/stitch_figures.py --param_to_change strength --figure_name distance_rank_matrix_heatmap_Blues_with_border.png

# ラベル非表示版
python src/stitch_figures.py --figure_name distance_rank_matrix_heatmap_Blues_with_border.png --hide_outer_labels

# 4パネル比較
python src/stitch_figures.py --param_to_change flow_center
```

**出力構造**:
```
data/naive_simulation/fig/montage/
├── param_sweeps/
│   └── fixed_agents_7_Ni_100_alpha_0.001_fr_0.01/
│       └── distance_rank_matrix_heatmap_Blues_with_border/
│           ├── strength.png
│           ├── alpha.png
│           └── N_i.png
└── comparisons/
    └── fixed_agents_7_Ni_100_alpha_0.001_fr_0.01/
        └── distance_rank_matrix_heatmap_Blues_with_border/
            └── flow_center.png
```

## 4. ワークフロー実行手順

### 4.1 標準実行手順

```bash
# 1. シミュレーション実行（全パラメータスイープ）
./run_naive_simulations_parallel.sh

# 2. 図表生成（全条件の距離解析）
./run_naive_simulations_parallel.sh -P

# 3. 図表結合（例：距離ランク行列）
python src/stitch_figures.py --figure_name distance_rank_matrix_heatmap_Blues_with_border.png --hide_outer_labels
```

### 4.2 特定パラメータのみ実行

```bash
# 1. 結合強度のみのシミュレーション
./run_naive_simulations_parallel.sh -p strength

# 2. 対応する図表生成
./run_naive_simulations_parallel.sh -p strength -P

# 3. 結合強度スイープの結合
python src/stitch_figures.py --param_to_change strength --figure_name distance_rank_matrix_heatmap_Blues_with_border.png
```

## 5. データ管理

### 5.1 ディレクトリ構造
```
data/naive_simulation/
├── raw/                    # シミュレーション生データ
│   └── {条件別サブディレクトリ}/
│       ├── state_*.npy
│       ├── distance_*.npy
│       ├── random_state_*.pkl
│       └── save_idx_t_map.csv
├── fig/                    # 個別図表
│   ├── {条件別サブディレクトリ}/
│   │   ├── mean_distance_heatmap_*.png
│   │   ├── *_similarity_*.png
│   │   └── age_*.png
│   └── montage/           # 結合図表
│       ├── param_sweeps/
│       └── comparisons/
```

### 5.2 ファイル命名規則
```
# シミュレーション条件サブディレクトリ
{flow_prefix}nonzero_alpha_{na}_fr_{strength}_agents_{agents}_N_i_{Ni}_alpha_{alpha}

# 例
bidirectional_flow-nonzero_alpha_center_fr_0.01_agents_7_N_i_100_alpha_0.001
outward_flow-nonzero_alpha_evenly_fr_0.05_agents_7_N_i_200_alpha_0.002
```

## 6. 主要出力と解釈

### 6.1 距離ランク行列の解釈
- **同心円分布**: 中心エージェントからの距離が対称的な場合
- **非対称分布**: 特定方向に言語変化が偏る場合
- **枠線色分け**:
  - 黒枠: 中心エージェントとのペア
  - 赤枠: 同心円分布を崩すペア

### 6.2 統計検定結果
- **二項検定**: 中心vs対照との距離差の有意性
- **ロジスティック回帰**: パラメータが分布パターンに与える影響
- **p値ヒートマップ**: エージェントペア別の統計的有意性

### 6.3 パラメータ効果
- **結合強度**: エージェント間相互作用の強さ
- **新語生成率**: 言語革新の頻度
- **集団サイズ**: ドリフト効果の大きさ
- **フロータイプ**: 情報伝播の方向性（双方向 vs 外向き）
- **バイアス分布**: 革新源の空間分布（中心集中 vs 均等分散）

## 7. 技術仕様

### 7.1 依存関係
- **Python**: numpy, matplotlib, pandas, scikit-learn, statsmodels, tqdm, PIL
- **システム**: Task Spooler (tsp), bash
- **カスタムモジュール**: ilm.networks

### 7.2 計算効率化
- **並列処理**: Task Spoolerによるジョブ並列化
- **メモリ効率**: 大規模データの段階的読み込み
- **キャッシュ機能**: 類似度計算結果の保存・再利用
- **再開機能**: 既存結果からの継続実行

### 7.3 カスタマイズ可能要素
- **図表スタイル**: カラーマップ、サイズ、レイアウト
- **統計解析**: 検定手法、回帰モデル
- **パラメータ範囲**: スイープ対象値の設定
- **出力フォーマット**: PNG、PDF、解像度設定