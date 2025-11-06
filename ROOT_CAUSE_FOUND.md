# 🔴 **ROOT CAUSE FOUND!**

## The Models Being Compared Are DIFFERENT!

### Simulation Parameters (Your Command)
```bash
--flow_type bidirectional
--nonzero_alpha evenly
```

**Network matrix:**
```
[[0.995 0.005 0.000]
 [0.005 0.990 0.005]
 [0.000 0.005 0.995]]
```

**Alphas:** `[0.001, 0.001, 0.001]`
- ALL agents innovate equally

**This is CASE 1:**
- `center_prestige = False` (bidirectional = symmetric)
- `centralized_neologism_creation = False` (evenly = all innovate)

---

### Symbolic Analysis (What You Analyzed)
```bash
python symbolic_analysis/test_inequality.py case4
```

**Case 4 parameters:**
- `center_prestige = True` (outward flow)
- `centralized_neologism_creation = True` (center-only)

**Network matrix (Case 4):**
```
[[0.99  0.01  0.00]
 [0.00  1.00  0.00]
 [0.00  0.01  0.99]]
```

**Alphas:** `[0, 0.001, 0]`
- ONLY center innovates

---

## Conclusion

**You compared two completely different models!**

- **Simulation**: bidirectional + evenly (CASE 1)
- **Symbolic**: outward + center-only (CASE 4)

The discrepancy is NOT a bug—it's expected because they're different models!

---

## How to Fix

### Option 1: Match Simulation to Case 4

Run simulation with:
```bash
python src/naive_simulation.py \
  --coupling_strength 0.01 \
  --alpha_per_data 0.001 \
  --flow_type outward \
  --nonzero_alpha center \
  --N_i 1 \
  --agents_count 3
```

### Option 2: Match Symbolic to Case 1

Run symbolic analysis for case1:
```bash
python symbolic_analysis/test_m_agent_stationary.py 1
python symbolic_analysis/test_distance_analysis.py case1
python symbolic_analysis/test_inequality.py case1
```

Then update verification script to load case1 instead of case4.

---

## Original Data You Showed

```
Processing: center + bidirectional
Processing: center + outward
```

This suggests you have BOTH:
- `center + bidirectional` → **CASE 3**
- `center + outward` → **CASE 4**

But your simulation command uses `evenly + bidirectional` → **CASE 1**

Please clarify which case you actually want to compare!
