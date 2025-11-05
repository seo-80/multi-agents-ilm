# バグ分析サマリー

## 状況

**シミュレーション結果** (M=3, m=0.01, alpha=0.001, outward flow, center-only):
```
E[d_12] = 0.193 > E[d_13] = 0.190
```

**symbolic_analysis結果** (ユーザー報告):
```
E[d_12] < E[d_13] (パラメータ空間の1/176でのみ逆転)
```

## 数学的分析の結論

M=3の場合、以下の関係式が成立：
```
E[d_12] - E[d_13] = π₃ - π₂
```

where:
- State 2: {{1,2},{3}} = 左端と中心が同じ、右端が異なる
- State 3: {{1,3},{2}} = 左端と右端が同じ、中心が異なる

## 理論的予測（手動計算）

center-only innovation + outward flowの場合：

### State 1からの遷移
```
P(State 1 → State 3) ≈ 0.001  (中心が突然変異)
P(State 1 → State 2) ≈ 0.00001 (周辺が突然変異、ほぼ不可能)
```

**State 3への流入が圧倒的に大きい！**

### 理由
1. 中心エージェント（agent 2）のみが革新を生成（μ₂ ≈ 0.001, μ₁ = μ₃ = 0）
2. 中心が突然変異すると → State 3（左端と右端は古いデータを保持）
3. 周辺は突然変異しないので → State 2になりにくい

### 結論
```
π₃ > π₂  ⟹  E[d_12] > E[d_13]
```

**シミュレーション結果と一致！**

## symbolic_analysisの検証

### W行列の検証 ✓
```
       j=1      j=2      j=3
i=1  [  1-m      m       0    ]
i=2  [   0       1       0    ]
i=3  [   0       m      1-m   ]
```
**正しい！**

### 遷移確率ロジックの検証 ✓
- `enumerate_valid_mappings`: 正しく実装されている
- `prob_copy_block`: 正しく実装されている
- `prob_receive_mutation_from`: 正しく実装されている

State 1 → State 3の計算を手動で追跡：
```
{1,3} → {1,2,3}, {2} → mutation
P ≈ (1-0.001m)² × 0.001 ≈ 0.001
```
**計算ロジックは正しい！**

## 必要な確認事項

symbolic_analysisの実際の計算結果を確認してください：

1. **M=3, case4, m=0.01, alpha=0.001での数値結果:**
   ```
   E_symbolic[d_12] = ?
   E_symbolic[d_13] = ?
   ```

2. **定常分布:**
   ```
   π₁ ({{1,2,3}}) = ?
   π₂ ({{1,2},{3}}) = ?
   π₃ ({{1,3},{2}}) = ?
   π₄ ({{1},{2,3}}) = ?
   π₅ ({{1},{2},{3}}) = ?
   ```

3. **主要な遷移確率:**
   ```
   P(State 1 → State 2) = ?
   P(State 1 → State 3) = ?
   P(State 5 → State 2) = ?
   P(State 5 → State 3) = ?
   ```

## 可能性のあるバグ箇所

コードレビューでは明らかなバグが見つかりませんでしたが、以下の可能性があります：

### 仮説1: インデックスの混乱
- symbolic_analysisが1-indexed、simulationが0-indexedで混乱？
- しかし、確認した限り正しくマッピングされている

### 仮説2: case4の定義が違う
- simulationの"outward + center-only"
- symbolic_analysisの"center_prestige + centralized_neologism_creation"
- これらが本当に同じモデルか再確認が必要

### 仮説3: 数値計算の精度問題
- sympyのsolveで定常分布を計算する際の数値誤差？
- π₂とπ₃は非常に小さい値なので、誤差が影響している可能性

### 仮説4: パラメータ代入のタイミング
- symbolic_analysisは記号計算で一般解を求める
- 具体的なm=0.01, alpha=0.001を代入するタイミングで問題？

## 推奨される次のステップ

1. **symbolic_analysisを実際に実行**
   ```bash
   # sympyをインストール
   pip install sympy

   # M=3, case4を計算
   python symbolic_analysis/test_m_agent_stationary.py 3 true true

   # 距離の期待値を計算
   python symbolic_analysis/src/analyze_distances.py 3 case4

   # m=0.01, alpha=0.001を代入して数値評価
   ```

2. **詳細な比較**
   - 全ての遷移確率を出力
   - 定常分布を出力
   - 距離の期待値を出力

3. **デバッグ出力の追加**
   - `transition_probability`関数に詳細なログを追加
   - 各マッピングの確率を出力
   - 定常分布の計算過程を出力

## 現時点での結論

**理論的には、シミュレーション結果（E[d_12] > E[d_13]）が正しいはずです。**

もしsymbolic_analysisが逆の結果を出しているなら、以下のいずれかです：

1. **symbolic_analysisに未発見のバグがある**
2. **symbolic_analysisとsimulationで微妙にモデル定義が違う**
3. **ユーザーが提供した情報の解釈に誤解がある**

コードレビューだけでは限界があるため、**実際の計算結果の確認が必須**です。
