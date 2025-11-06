# CASE 4 Comparison: Simulation vs Symbolic Analysis

## Parameters (Confirmed Matching)

**Both use:**
- M = 3
- m = 0.01
- alpha = 0.001
- Flow type: outward (center-prestige)
- Innovation: center-only

**Network matrix:**
```
[[0.99  0.01  0.00]
 [0.00  1.00  0.00]
 [0.00  0.01  0.99]]
```

**Alphas:** `[0, 0.001, 0]`
**Mu:** `[0, 0.000999, 0]`

✓ **Models match perfectly**

---

## Results Comparison

### Symbolic Analysis

**Stationary distribution:**
```
π_1 = 0.864  ({{1,2,3}} - all same)
π_2 = 0.044  ({{1},{2,3}} - left alone, center&right same)
π_3 = 0.043  ({{2},{1,3}} - center alone, left&right same)
π_4 = 0.044  ({{3},{1,2}} - right alone, left&center same)
π_5 = 0.005  ({{1},{2},{3}} - all different)
```

**Distances (0/1 metric):**
```
E[d_12] = 0.092
E[d_13] = 0.092
E[d_12] - E[d_13] = -0.00045 < 0
```

**Conclusion:** E[d_12] < E[d_13]

---

### Simulation

**Distances (Manhattan metric for N_i=1):**
```
E[d_12] = 0.193
E[d_13] = 0.190
E[d_12] - E[d_13] = 0.003 > 0
```

**Conclusion:** E[d_12] > E[d_13]

---

## Analysis

### Issue 1: Magnitude Difference (Factor of 2)

Simulation distances are approximately **2× symbolic distances**:

```
Simulation / Symbolic:
0.193 / 0.092 = 2.10
0.190 / 0.092 = 2.07
```

**Cause:** Manhattan distance for N_i=1
- Different data: distance = 2 (each unique datum counts separately)
- Symbolic uses 0/1 metric

**This is expected and not a bug.**

---

### Issue 2: Sign Reversal (CRITICAL BUG)

Converting symbolic to simulation metric (multiply by 2):

```
Symbolic (×2):
E[d_12] = 0.092 × 2 = 0.184
E[d_13] = 0.092 × 2 = 0.184
Difference = -0.00045 × 2 = -0.0009

Simulation:
E[d_12] = 0.193
E[d_13] = 0.190
Difference = +0.003
```

**The SIGN is opposite!**
- Symbolic predicts: E[d_12] < E[d_13] (by -0.0009 when scaled)
- Simulation shows: E[d_12] > E[d_13] (by +0.003)

**This indicates a real bug, not just a scaling issue.**

---

## Theoretical Expectation

With **center-only innovation** and **outward flow**:

**State transitions from State 1 {{1,2,3}}:**

To **State 2** {{1},{2,3}} (left alone):
- Agent 1 must get new data
- Only via mutation from center: W[1,2] × mu_2 = 0.01 × 0.001 = 0.00001
- **Very rare**

To **State 3** {{2},{1,3}} (center alone):
- Agent 2 (center) must mutate
- Probability: W[2,2] × mu_2 = 1 × 0.001 = 0.001
- **100× more likely than State 2!**

To **State 4** {{3},{1,2}} (right alone):
- Agent 3 must get new data
- Only via mutation from center: W[3,2] × mu_2 = 0.01 × 0.001 = 0.00001
- **Very rare**

**Expected:** π_3 > π_2 = π_4 (by symmetry)

**Symbolic result:** π_2 = π_4 = 0.044 > π_3 = 0.043 ❌

**This contradicts theory!**

---

## Where is the Bug?

The symbolic_analysis computes **π_3 < π_2**, which implies:
- E[d_12] < E[d_13]

But theory and simulation both suggest:
- π_3 > π_2
- E[d_12] > E[d_13]

**Suspect:** Bug in `transition_probability` or `enumerate_valid_mappings`

Specifically, the transitions FROM State 1 to States 2, 3, 4 may be incorrectly computed.

---

## Next Steps

1. **Manually verify transition probabilities:**
   - P(State 1 → State 2)
   - P(State 1 → State 3)
   - P(State 1 → State 4)

2. **Check `enumerate_valid_mappings` for State 1 → State 3**

3. **Add debug output to transition probability calculation**

4. **Compare with simulation transition counts** (if available)
