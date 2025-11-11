# Nei距離を文化進化モデルに使用する理論的根拠

## 概要

このプロジェクトでは、集団間の文化的分化を測定するために**Nei標準遺伝距離**（Nei 1972）を使用しています。本文書では、その理論的妥当性と注意点を説明します。

## Nei距離の定義

Nei標準遺伝距離は以下の式で定義されます：

```
D_ij = -ln(F_ij / sqrt(F_ii * F_jj))
```

ここで：
- `F_ij`: 集団iと集団jの間のIBD（Identity By Descent）確率
- `F_ii`, `F_jj`: 各集団内のIBD確率（同質性の指標）

実装: `IBD_analysis/src/distance_metrics.py:12-49`

## なぜNei距離が適切か

### 1. モデルの前提条件が一致

このプロジェクトのモデルは以下の前提に基づいています：

| 条件 | 本モデル | Nei距離の前提 | 整合性 |
|------|----------|--------------|--------|
| 突然変異モデル | 無限対立遺伝子 (`BayesianInfiniteVariantsAgent`) | 無限対立遺伝子 | ✅ |
| 進化モデル | 中立進化 | 中立進化 | ✅ |
| 形質の離散性 | 離散的変異体 | 離散的対立遺伝子 | ✅ |
| IBDの仮定 | 同一変異体=共通祖先由来 | 同一対立遺伝子=共通祖先由来 | ✅ |

### 2. F行列の理論的妥当性

`IBD_analysis/src/f_matrix_symbolic.py`で計算されるF行列は、標準的な集団遺伝学のIBDモデルに基づいています：

**異なるエージェント間** (i ≠ j):
```
F_ij = Σ_k Σ_l W_ik W_jl (1-μ_k)(1-μ_l) F_kl
```

**同一エージェント内** (i = j):
```
F_ii = 1/N + (1-1/N) Σ_k Σ_l W_ik W_il (1-μ_k)(1-μ_l) F_kl
```

ここで：
- `W_ik`: エージェントiがエージェントkから学習する重み
- `μ_i = α_i / (N + α_i)`: エージェントiの突然変異率
- `N`: エージェントあたりのデータサイズ

この定式化は、**文化伝達を遺伝的継承のアナロジー**として扱っており、Nei距離の理論的基礎と整合します。

## 文化進化における解釈

### IBD確率の意味

**遺伝学での解釈**:
- F_ij = エージェントiとjがランダムに選んだ対立遺伝子が共通祖先由来である確率

**文化進化での解釈**:
- F_ij = エージェントiとjがランダムに選んだ文化形質（データ点）が共通の起源（同じエージェントの過去の生成）から派生している確率

### 世代の概念

文化進化では生物学的世代が存在しませんが、本モデルでは：
- **離散時間ステップ** = 世代に相当
- 各ステップ: 学習（copying）→ 生成（production）→ 革新（innovation）
- `agents.py:85`: `self.__generation += 1`

これは遺伝の世代交代に類似しており、Nei距離の時間的解釈が可能です。

## 代替距離指標との比較

| 距離指標 | 特徴 | 本モデルでの適合性 |
|---------|------|------------------|
| **Nei距離** | IBD確率ベース、対数変換、無限対立遺伝子に最適 | ✅ **最適** |
| 1-F | シンプルな補数、線形距離 | △ 加法性なし |
| -log(F) | 加法的、F_iiを考慮しない | △ 集団内異質性を無視 |
| sqrt(1-F) | 中間的変換 | △ 理論的根拠が弱い |
| FST | 集団分化の標準指標 | △ 有限対立遺伝子向け |

実装: `IBD_analysis/src/distance_metrics.py:52-103`（複数の距離指標を実装済み）

## 使用上の注意点

### 1. F_iiの対角要素の解釈

F_iiは「エージェント内の異なるデータ点間のIBD確率」を表します。これは：
- **遺伝学**: 同一個体の2つの遺伝子座（通常は1）
- **文化**: 同一エージェントの異なる文化形質（N個のデータ点から2つ選ぶ）

このため、F_iiはエージェントの「内的同質性」を測ります。Nei距離の分母`sqrt(F_ii * F_jj)`は、この同質性で正規化します。

### 2. 有限サンプルサイズ効果

F_iiの`1/N`項（`f_matrix_symbolic.py:19`）は：
- N個のデータ点のうち、同じデータ点を2回選ぶ確率
- Nが大きいほど、この項は無視できる（無限集団の近似）

### 3. 選択圧がある場合の拡張

現在のモデルは中立進化を仮定していますが、将来的に選択圧（例: prestige bias）を導入する場合：
- Nei距離は系統距離を**過小評価**する可能性があります
- 平行進化（convergent evolution）により、独立した起源の形質が同一になる
- この場合、**Fst**や**selection-aware距離**の検討が必要

ただし、現在のコードには`center_prestige`パラメータがありますが（`f_matrix_symbolic.py:34`）、これは**情報の流れの非対称性**を表し、選択圧とは異なります：
- `center_prestige=True`: 中心から周辺への一方向的な影響（情報構造）
- 選択圧: 特定の形質が生存・伝播で有利（適応度の差）

本モデルでは学習重み`W`が固定されており、形質の内容には依存しないため、**中立進化の仮定は維持**されています。

## 実装の検証

### 数値シミュレーションとの比較

`compare_symbolic_simulation.py`を使用して、記号解（理論的F行列）とモンテカルロシミュレーションを比較できます：

```bash
python compare_symbolic_simulation.py --M 5 --cases case1 case2 case3 case4
```

これにより：
1. 記号解から計算したNei距離
2. シミュレーションから推定したF行列→Nei距離

の一致度を確認できます。

### 同心円分布の検出

`IBD_analysis/src/concentric_analysis.py:23-50`では、Nei距離行列を使って同心円分布を検出しています：

```python
def is_concentric_distribution(distance_matrix):
    """
    Check if the distance matrix shows a concentric distribution.

    A distribution is concentric if there exists a base agent (not center)
    that is closer to an agent on the opposite side than to the center.
    """
```

これは、距離指標が**空間的なパターン**を捉えられることを示しています。

## 参考文献

1. **Nei, M. (1972)**. Genetic distance between populations. *The American Naturalist*, 106(949), 283-292.
   - Nei距離の原論文

2. **Cavalli-Sforza, L. L., & Feldman, M. W. (1981)**. *Cultural transmission and evolution: A quantitative approach*. Princeton University Press.
   - 文化進化における集団遺伝学モデルの適用

3. **Boyd, R., & Richerson, P. J. (1985)**. *Culture and the evolutionary process*. University of Chicago Press.
   - 文化進化理論の基礎

4. **Ewens, W. J. (2004)**. *Mathematical Population Genetics*. Springer.
   - 無限対立遺伝子モデルとIBD理論

## 結論

本モデルにおいて**Nei距離は理論的に適切**です：

✅ 無限対立遺伝子モデル
✅ 中立進化の仮定
✅ IBD確率ベースのF行列
✅ 正しい実装

文化進化のコンテキストでは、「形質の共通起源」をIBDとして解釈することで、遺伝学のアナロジーが成立します。

ただし、将来的に**内容依存的な選択圧**を導入する場合は、距離指標の再検討が必要です。
