"""
Debug transition probabilities for State 1 to States 2, 3, 4

This manually calculates the transition probabilities to verify
the implementation is correct.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'symbolic_analysis/src'))

from m_agent_stationary_symbolic import (
    load_results_by_case, transition_probability,
    prob_copy_block, prob_receive_mutation_from
)
from sympy import symbols

# Load results
print("Loading symbolic_analysis results for M=3, case4...")
results = load_results_by_case(3, "case4")
states = results['states']
W = results['W']
mu_vec = results['mu_vec']
P = results['P']

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

print("\n" + "="*80)
print("TRANSITION PROBABILITIES FROM STATE 1")
print("="*80)

# State 1 is states[0]
S = states[0]  # {{1,2,3}}

print(f"\nFrom State 1: {str(S).replace('frozenset', '').replace('(', '').replace(')', '')}")
print()

# Compute transitions to each state
for i, S_prime in enumerate(states):
    if i == 0:
        continue  # Skip self-transition for now

    state_str = str(S_prime).replace('frozenset', '').replace('(', '').replace(')', '')

    # Compute transition probability symbolically
    prob_symbolic = transition_probability(S, S_prime, W, mu_vec, 3)

    # Evaluate numerically
    prob_numeric = prob_symbolic.subs({m_symbol: m_val, alpha_symbol: alpha_val}).evalf()
    prob_val = float(prob_numeric) if prob_numeric.is_real else float(prob_numeric.as_real_imag()[0])

    print(f"To State {i+1}: {state_str}")
    print(f"  P(1 → {i+1}) = {prob_val:.10f}")
    print()

print("="*80)
print("MANUAL CALCULATION VERIFICATION")
print("="*80)

# Manual calculation for State 1 → State 3
print("\nManual: State 1 → State 3 {{2},{1,3}}")
print("  Agents 1 and 3 must copy from {1,2,3}")
print("  Agent 2 must receive mutation")
print()

# W matrix values
W_vals = W.subs({m_symbol: m_val})
mu_vals = [mu_vec[i].subs({alpha_symbol: alpha_val}) for i in range(3)]

print("W matrix:")
for i in range(3):
    row = [float(W_vals[i,j]) for j in range(3)]
    print(f"  {row}")
print()
print(f"mu = [{float(mu_vals[0])}, {float(mu_vals[1])}, {float(mu_vals[2])}]")
print()

# Probability for agent 1 to copy from {1,2,3}
prob_1 = sum(float(W_vals[0,j]) * (1 - float(mu_vals[j])) for j in range(3))
print(f"prob_copy_block(1, {{1,2,3}}) = {prob_1:.10f}")

# Probability for agent 3 to copy from {1,2,3}
prob_3 = sum(float(W_vals[2,j]) * (1 - float(mu_vals[j])) for j in range(3))
print(f"prob_copy_block(3, {{1,2,3}}) = {prob_3:.10f}")

# Probability for agent 2 to receive mutation
prob_2_mut = sum(float(W_vals[1,j]) * float(mu_vals[j]) for j in range(3))
print(f"prob_mutation(2) = {prob_2_mut:.10f}")

total_manual_3 = prob_1 * prob_3 * prob_2_mut
print(f"\nTotal P(1 → 3) = {total_manual_3:.10f}")

print("\n" + "-"*80)

# Manual calculation for State 1 → State 2
print("\nManual: State 1 → State 2 {{1},{2,3}}")
print("  Agents 2 and 3 must copy from {1,2,3}")
print("  Agent 1 must receive mutation")
print()

# Probability for agent 2 to copy from {1,2,3}
prob_2 = sum(float(W_vals[1,j]) * (1 - float(mu_vals[j])) for j in range(3))
print(f"prob_copy_block(2, {{1,2,3}}) = {prob_2:.10f}")

# Probability for agent 1 to receive mutation
prob_1_mut = sum(float(W_vals[0,j]) * float(mu_vals[j]) for j in range(3))
print(f"prob_mutation(1) = {prob_1_mut:.10f}")

total_manual_2 = prob_2 * prob_3 * prob_1_mut
print(f"\nTotal P(1 → 2) = {total_manual_2:.10f}")

print("\n" + "-"*80)

# Manual calculation for State 1 → State 4
print("\nManual: State 1 → State 4 {{3},{1,2}}")
print("  Agents 1 and 2 must copy from {1,2,3}")
print("  Agent 3 must receive mutation")
print()

# Probability for agent 3 to receive mutation
prob_3_mut = sum(float(W_vals[2,j]) * float(mu_vals[j]) for j in range(3))
print(f"prob_mutation(3) = {prob_3_mut:.10f}")

total_manual_4 = prob_1 * prob_2 * prob_3_mut
print(f"\nTotal P(1 → 4) = {total_manual_4:.10f}")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)

print(f"\nP(1 → 2) / P(1 → 3) = {total_manual_2 / total_manual_3:.6f}")
print(f"P(1 → 4) / P(1 → 3) = {total_manual_4 / total_manual_3:.6f}")

print("\nExpected:")
print("  P(1 → 2) ≈ P(1 → 4) (by symmetry)")
print("  P(1 → 3) >> P(1 → 2) ≈ P(1 → 4)")
print("  Because only center mutates (mu_1 = mu_3 = 0, mu_2 > 0)")

if total_manual_3 > total_manual_2:
    print("\n✓ Transition probabilities are correct!")
    print("  P(1 → 3) > P(1 → 2) as expected")
else:
    print("\n✗ BUG FOUND in transition probabilities!")
    print(f"  P(1 → 3) = {total_manual_3} is NOT > P(1 → 2) = {total_manual_2}")
