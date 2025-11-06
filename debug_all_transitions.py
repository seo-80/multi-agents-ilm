"""
Debug ALL transition probabilities in the transition matrix

This script examines all transitions to find where the bug occurs.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'symbolic_analysis/src'))

from m_agent_stationary_symbolic import (
    load_results_by_case, transition_probability
)
from sympy import symbols
import numpy as np

# Load results
print("Loading symbolic_analysis results for M=3, case4...")
results = load_results_by_case(3, "case4")
states = results['states']
W = results['W']
mu_vec = results['mu_vec']
P = results['P']
pi = results['pi']

# Find symbols
m_symbol = [s for s in W[0,0].free_symbols if str(s) == 'm'][0]
alpha_symbol = [s for s in mu_vec[1].free_symbols if str(s) == 'alpha'][0]

# Parameters
m_val = 0.01
alpha_val = 0.001

print(f"\nParameters: m={m_val}, alpha={alpha_val}")
print("="*80)

# Display states
print("\nStates:")
for i, state in enumerate(states):
    state_str = str(state).replace('frozenset', '').replace('(', '').replace(')', '')
    print(f"  State {i+1}: {state_str}")

# Evaluate full transition matrix numerically
print("\n" + "="*80)
print("FULL TRANSITION MATRIX P")
print("="*80)

P_numeric = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        P_expr = P[i, j].subs({m_symbol: m_val, alpha_symbol: alpha_val}).evalf()
        P_numeric[i, j] = float(P_expr) if P_expr.is_real else float(P_expr.as_real_imag()[0])

print("\nTransition matrix P (rows = from, cols = to):")
print("       ", end="")
for j in range(5):
    print(f"  S{j+1}      ", end="")
print()
for i in range(5):
    print(f"  S{i+1}: ", end="")
    for j in range(5):
        print(f"{P_numeric[i, j]:.6f}  ", end="")
    print()

# Check row sums (should be 1)
print("\nRow sums (should all be 1.0):")
for i in range(5):
    row_sum = sum(P_numeric[i, :])
    print(f"  State {i+1}: {row_sum:.10f}")

# Evaluate stationary distribution
print("\n" + "="*80)
print("STATIONARY DISTRIBUTION")
print("="*80)

pi_numeric = []
for i in range(5):
    pi_expr = pi[i].subs({m_symbol: m_val, alpha_symbol: alpha_val}).evalf()
    pi_val = float(pi_expr) if pi_expr.is_real else float(pi_expr.as_real_imag()[0])
    pi_numeric.append(pi_val)
    print(f"  π_{i+1} = {pi_val:.10f}")

print(f"\nSum: {sum(pi_numeric):.10f}")

# Verify stationary distribution: π = πP
print("\n" + "="*80)
print("VERIFICATION: π = πP")
print("="*80)

pi_vec = np.array(pi_numeric)
pi_P = pi_vec @ P_numeric

print("\nLeft side (π):")
for i in range(5):
    print(f"  π_{i+1} = {pi_numeric[i]:.10f}")

print("\nRight side (πP):")
for i in range(5):
    print(f"  (πP)_{i+1} = {pi_P[i]:.10f}")

print("\nDifference (π - πP):")
max_diff = 0
for i in range(5):
    diff = pi_numeric[i] - pi_P[i]
    print(f"  State {i+1}: {diff:.2e}")
    max_diff = max(max_diff, abs(diff))

print(f"\nMaximum difference: {max_diff:.2e}")
if max_diff < 1e-9:
    print("✓ Stationary distribution is correct!")
else:
    print("✗ Stationary distribution does NOT satisfy πP = π")

# Focus on transitions involving States 2 and 3
print("\n" + "="*80)
print("FOCUS: TRANSITIONS TO STATES 2 AND 3")
print("="*80)

print("\nTransitions TO State 2 {{1},{2,3}}:")
for i in range(5):
    if P_numeric[i, 1] > 1e-10:
        state_str = str(states[i]).replace('frozenset', '').replace('(', '').replace(')', '')
        print(f"  P({i+1}→2) = {P_numeric[i, 1]:.10f}  from {state_str}")

print("\nTransitions TO State 3 {{2},{1,3}}:")
for i in range(5):
    if P_numeric[i, 2] > 1e-10:
        state_str = str(states[i]).replace('frozenset', '').replace('(', '').replace(')', '')
        print(f"  P({i+1}→3) = {P_numeric[i, 2]:.10f}  from {state_str}")

# Compute contributions to π₂ and π₃
print("\n" + "="*80)
print("CONTRIBUTIONS TO STATIONARY DISTRIBUTION")
print("="*80)

print("\nContributions to π₂:")
total_2 = 0
for i in range(5):
    contrib = pi_numeric[i] * P_numeric[i, 1]
    if contrib > 1e-12:
        print(f"  π_{i+1} × P({i+1}→2) = {pi_numeric[i]:.6f} × {P_numeric[i, 1]:.6f} = {contrib:.10f}")
        total_2 += contrib
print(f"  Total π₂ = {total_2:.10f}")

print("\nContributions to π₃:")
total_3 = 0
for i in range(5):
    contrib = pi_numeric[i] * P_numeric[i, 2]
    if contrib > 1e-12:
        print(f"  π_{i+1} × P({i+1}→3) = {pi_numeric[i]:.6f} × {P_numeric[i, 2]:.6f} = {contrib:.10f}")
        total_3 += contrib
print(f"  Total π₃ = {total_3:.10f}")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

print(f"\nπ₂ = {pi_numeric[1]:.10f}")
print(f"π₃ = {pi_numeric[2]:.10f}")
print(f"π₃ - π₂ = {pi_numeric[2] - pi_numeric[1]:.10f}")

if pi_numeric[2] < pi_numeric[1]:
    print("\n✗ BUG: π₃ < π₂ but should be π₃ > π₂")
    print("\nExpected: π₃ > π₂ because:")
    print("  - Only center (agent 2) mutates")
    print("  - State 1 → State 3 requires center mutation (likely)")
    print("  - State 1 → State 2 requires endpoint mutation (unlikely)")
else:
    print("\n✓ π₃ > π₂ as expected")
