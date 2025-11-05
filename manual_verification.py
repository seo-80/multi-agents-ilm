"""
Manual verification of symbolic analysis for M=3, m=0.01, alpha=0.001
to compare with simulation results.

Parameters:
- M = 3 (agents: 1, 2, 3 in 1-indexed, or 0, 1, 2 in 0-indexed)
- m = 0.01 (very small coupling)
- alpha = 0.001 (very small innovation, only at center agent 2)
- N_i = 1

Simulation results (1-indexed):
- Bidirectional: E[d_12] = 0.193 > E[d_13] = 0.190
- Outward: E[d_12] = 0.193 > E[d_13] = 0.190

Key question: Does symbolic_analysis predict E[d_12] > E[d_13] or E[d_12] < E[d_13]?
"""

import sys
import os
sys.path.insert(0, 'symbolic_analysis/src')

# We need to check what symbolic_analysis actually computes
# Let's trace through the logic manually

print("="*80)
print("MANUAL VERIFICATION OF SYMBOLIC ANALYSIS")
print("="*80)
print("\nParameters: M=3, m=0.01, alpha=0.001")
print("Center agent: 2 (1-indexed)")
print("\nSimulation results:")
print("  Bidirectional: E[d_12] = 0.193 > E[d_13] = 0.190")
print("  Outward: E[d_12] = 0.193 > E[d_13] = 0.190")

print("\n" + "-"*80)
print("State space for M=3:")
print("-"*80)

states = [
    "{{1,2,3}}",    # All same
    "{{1,2},{3}}",  # 1,2 same; 3 different
    "{{1,3},{2}}",  # 1,3 same; 2 different
    "{{1},{2,3}}",  # 1 different; 2,3 same
    "{{1},{2},{3}}" # All different
]

for i, s in enumerate(states, 1):
    print(f"{i}. {s}")

print("\n" + "-"*80)
print("Distance matrix for each state:")
print("-"*80)

# For each state, compute d_12 and d_13
distances = []

for i, s in enumerate(states, 1):
    if s == "{{1,2,3}}":
        d_12, d_13 = 0, 0
        desc = "all same"
    elif s == "{{1,2},{3}}":
        d_12, d_13 = 0, 1
        desc = "1&2 same, 3 diff"
    elif s == "{{1,3},{2}}":
        d_12, d_13 = 1, 0
        desc = "1&3 same, 2 diff"
    elif s == "{{1},{2,3}}":
        d_12, d_13 = 1, 1
        desc = "1 alone, 2&3 same"
    else:  # "{{1},{2},{3}}"
        d_12, d_13 = 1, 1
        desc = "all different"

    distances.append((d_12, d_13))
    print(f"State {i}: {s:20} → d_12={d_12}, d_13={d_13}  ({desc})")

print("\n" + "-"*80)
print("Key observation:")
print("-"*80)
print("\nFor E[d_12] > E[d_13] to hold, we need:")
print("  π(state 2) > π(state 3)")
print("\nBecause:")
print("  E[d_12] - E[d_13] = π(state 2) + π(state 4) + π(state 5)")
print("                     - [π(state 3) + π(state 4) + π(state 5)]")
print("                    = π(state 2) - π(state 3)")
print()
print("State 2: {{1,2},{3}} - agents 1 and 2 share data, 3 different")
print("State 3: {{1,3},{2}} - agents 1 and 3 share data, 2 different")

print("\n" + "-"*80)
print("WHY MIGHT π(state 2) > π(state 3)?")
print("-"*80)
print("\nHypothesis: When only agent 2 creates innovations (center-only):")
print("  - Agent 2 constantly creates new data")
print("  - This data spreads to agents 1 and 3")
print("  - But agent 2 quickly creates NEW data, leaving 1 and 3 with OLD data")
print("  - So agents 1 and 3 are more likely to share OLD data from agent 2")
print("  - While agent 2 has NEW data, different from 1 and 3")
print("  → State 3 {{1,3},{2}} is MORE likely")
print("  → π(state 3) > π(state 2)")
print("  → E[d_12] < E[d_13]")

print("\n" + "-"*80)
print("BUT SIMULATION SHOWS E[d_12] > E[d_13]!")
print("-"*80)
print("\nThis suggests:")
print("  1. The symbolic analysis is wrong, OR")
print("  2. The simulation implementation differs from symbolic model, OR")
print("  3. My reasoning above is incorrect")

print("\n" + "-"*80)
print("ALTERNATIVE HYPOTHESIS:")
print("-"*80)
print("\nLet's think about the DYNAMICS more carefully...")
print("\nWith center-only innovation and very small m=0.01:")
print("  - Most of the time, agents keep their own data (W[i,i] ≈ 0.99)")
print("  - Rarely, they copy from neighbors")
print("  - Only agent 2 creates new innovations")
print()
print("Key insight: With m=0.01, agents are VERY isolated.")
print("  - Agent 1 copies from agent 2 with prob ≈ 0.01")
print("  - Agent 3 copies from agent 2 with prob ≈ 0.01")
print("  - Agent 2 keeps its data with prob ≈ 0.99 (bidirectional)")
print()
print("So most of the time, all three agents have DIFFERENT data.")
print("State 5 {{1},{2},{3}} dominates the stationary distribution.")
print()
print("The SMALL differences between E[d_12] and E[d_13] come from")
print("RARE events where agents temporarily share data.")

print("\n" + "-"*80)
print("NEED TO CHECK:")
print("-"*80)
print("1. What does symbolic_analysis actually predict for these parameters?")
print("2. Are the W matrices identical between the two implementations?")
print("3. Is there a bug in one of the implementations?")

print("\n" + "="*80)
print("RECOMMENDED NEXT STEP:")
print("="*80)
print("\nWe need the ACTUAL symbolic analysis results for:")
print("  M=3, m=0.01, alpha=0.001, case3 and case4")
print("\nPlease provide:")
print("  1. E_symbolic[d_12] = ?")
print("  2. E_symbolic[d_13] = ?")
print("  3. The stationary distribution π for all 5 states")
