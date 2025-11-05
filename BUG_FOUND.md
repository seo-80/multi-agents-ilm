# バグ特定：alpha範囲の問題

## 発見したバグ

`symbolic_analysis/src/solve_distance_inequality.py` の `analyze_inequality_numerically` 関数：

```python
def analyze_inequality_numerically(expr1: sympy.Expr, expr2: sympy.Expr,
                                   m_range: Tuple[float, float] = (0.01, 0.99),
                                   alpha_range: Tuple[float, float] = (0.01, 100.0),  # ← BUG!
                                   grid_size: int = 100) -> Dict:
```

**問題:**
- デフォルトの `alpha_range = (0.01, 100.0)`
- シミュレーションで使用している `alpha = 0.001` は**範囲外**
- したがって、数値グリッド評価でalpha=0.001の点が含まれていない

## 影響

1. **数値評価が不正確**
   - 「Fraction: 1/176 (0.006)」はalpha ≥ 0.01の範囲のみ
   - alpha=0.001での挙動が評価されていない

2. **プロットが誤解を招く**
   - alpha=0.001の点がプロット範囲外
   - 実際に使用しているパラメータでの結果が見えない

## 修正方法

### オプション1: デフォルト範囲を拡大

```python
alpha_range: Tuple[float, float] = (0.0001, 100.0),  # 0.001を含む
```

### オプション2: 対数スケールでサンプリング

```python
# Linear scale for m
m_vals = np.linspace(m_range[0], m_range[1], grid_size)

# Log scale for alpha (covers wide range better)
alpha_vals = np.logspace(np.log10(alpha_range[0]), np.log10(alpha_range[1]), grid_size)
```

これにより、alpha=0.001のような小さい値もカバーできます。

## 理論的予測の確認

State順序（実際の実装）:
```
State 1: {{1,2,3}}      d_12=0, d_13=0
State 2: {{1},{2,3}}    d_12=1, d_13=1
State 3: {{1,3},{2}}    d_12=1, d_13=0  ← 左端と右端が同じ
State 4: {{1,2},{3}}    d_12=0, d_13=1  ← 左端と中心が同じ
State 5: {{1},{2},{3}}  d_12=1, d_13=1
```

公式:
```
E[d_12] - E[d_13] = π_3 - π_4
```

**理論的予測 (m=0.01, alpha=0.001, outward + center-only):**
- center-only innovationなので、中心のみが突然変異
- State 1 {{1,2,3}} → State 3 {{1,3},{2}}: 中心が突然変異（確率 ≈ 0.001）
- State 1 {{1,2,3}} → State 4 {{1,2},{3}}: 右端が突然変異（確率 ≈ 0、不可能！）
- したがって **π_3 > π_4**
- よって **E[d_12] > E[d_13]** ✓

**シミュレーション結果:**
```
E[d_12] = 0.193 > E[d_13] = 0.190
```

**一致！**

## 結論

symbolic_analysisの**記号計算自体は正しい**可能性が高いです。

問題は**数値評価の範囲設定**にあります。

修正すれば、symbolic_analysisもシミュレーションと同じ結果を示すはずです。

## 検証手順

1. `alpha_range`を修正
2. M=3, case4を再計算
3. m=0.01, alpha=0.001での数値を確認
4. シミュレーション結果と比較
