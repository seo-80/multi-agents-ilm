"""
Debug the reverse transitions between States 2 and 3

State 2: {{1},{2,3}}  - agents 2 and 3 share data, agent 1 different
State 3: {{2},{1,3}}  - agents 1 and 3 share data, agent 2 different

Key question: Why is P(3→2) >> P(2→3)?
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'symbolic_analysis/src'))

from m_agent_stationary_symbolic import (
    load_results_by_case, transition_probability,
    prob_copy_block, prob_receive_mutation_from
)

# Load results
print("Loading symbolic_analysis results for M=3, case4...")
results = load_results_by_case(3, "case4")
states = results['states']
W = results['W']
mu_vec = results['mu_vec']

# Find symbols
m_symbol = [s for s in W[0,0].free_symbols if str(s) == 'm'][0]
alpha_symbol = [s for s in mu_vec[1].free_symbols if str(s) == 'alpha'][0]

# Parameters
m_val = 0.01
alpha_val = 0.001

print(f"\nParameters: m={m_val}, alpha={alpha_val}")
print("="*80)

# States
S2 = states[1]  # {{1},{2,3}}
S3 = states[2]  # {{2},{1,3}}

print("\nState 2:", str(S2).replace('frozenset', '').replace('(', '').replace(')', ''))
print("State 3:", str(S3).replace('frozenset', '').replace('(', '').replace(')', ''))

# W matrix
W_vals = W.subs({m_symbol: m_val})
mu_vals = [mu_vec[i].subs({alpha_symbol: alpha_val}) for i in range(3)]

print("\nW matrix (outward flow):")
for i in range(3):
    row = [float(W_vals[i,j]) for j in range(3)]
    print(f"  Agent {i+1}: {row}")

print(f"\nmu = [{float(mu_vals[0]):.10f}, {float(mu_vals[1]):.10f}, {float(mu_vals[2]):.10f}]")
print("  (Only center mutates)")

print("\n" + "="*80)
print("TRANSITION: State 2 → State 3")
print("  From {{1},{2,3}} to {{2},{1,3}}")
print("="*80)

print("\nWhat needs to happen:")
print("  Initial: {1} | {2,3}  (agent 1 different)")
print("  Final:   {2} | {1,3}  (agent 2 different)")
print()
print("  Agents 1 and 3 must end up in same block")
print("  Agent 2 must end up alone")

# This is tricky. Let me think about all possible ways this can happen:
print("\nPossible scenarios:")
print("  1. Agent 1 copies from {2,3}, Agent 2 mutates, Agent 3 copies from {1,2,3}")
print("  2. Agent 1 copies from {1}, Agent 2 mutates, Agent 3 copies from {1}")
print("  3. Others...")

# Actually, let me just compute it directly
prob_2to3_symbolic = transition_probability(S2, S3, W, mu_vec, 3)
prob_2to3 = prob_2to3_symbolic.subs({m_symbol: m_val, alpha_symbol: alpha_val}).evalf()
prob_2to3_val = float(prob_2to3) if prob_2to3.is_real else float(prob_2to3.as_real_imag()[0])

print(f"\nComputed P(2→3) = {prob_2to3_val:.10f}")

print("\n" + "="*80)
print("TRANSITION: State 3 → State 2")
print("  From {{2},{1,3}} to {{1},{2,3}}")
print("="*80)

print("\nWhat needs to happen:")
print("  Initial: {2} | {1,3}  (agent 2 different)")
print("  Final:   {1} | {2,3}  (agent 1 different)")
print()
print("  Agents 2 and 3 must end up in same block")
print("  Agent 1 must end up alone")

print("\nPossible scenarios:")
print("  1. Agent 1 mutates (via copying from center with mutation)")
print("  2. Agent 2 copies from {1,3}, Agent 3 stays same")
print("  3. Others...")

# Compute directly
prob_3to2_symbolic = transition_probability(S3, S2, W, mu_vec, 3)
prob_3to2 = prob_3to2_symbolic.subs({m_symbol: m_val, alpha_symbol: alpha_val}).evalf()
prob_3to2_val = float(prob_3to2) if prob_3to2.is_real else float(prob_3to2.as_real_imag()[0])

print(f"\nComputed P(3→2) = {prob_3to2_val:.10f}")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)

print(f"\nP(2→3) = {prob_2to3_val:.10f}")
print(f"P(3→2) = {prob_3to2_val:.10f}")
print(f"P(3→2) / P(2→3) = {prob_3to2_val / prob_2to3_val:.2f}")

print("\n" + "="*80)
print("MANUAL ANALYSIS")
print("="*80)

print("\nState 2 → State 3: {{1},{2,3}} → {{2},{1,3}}")
print("  Current situation:")
print("    - Agent 1 is alone (has unique data)")
print("    - Agents 2 and 3 share data")
print()
print("  Network (outward flow):")
print("    - Agent 1 can only copy from: agent 1 (w=0.99) or agent 2 (w=0.01)")
print("    - Agent 2 can only copy from: agent 2 (w=1.0)")
print("    - Agent 3 can only copy from: agent 3 (w=0.99) or agent 2 (w=0.01)")
print()
print("  For State 2 → State 3:")
print("    - Need agent 1 to join {1,3} block")
print("    - Need agent 2 to become isolated")
print()
print("  This requires:")
print("    - Agent 2 MUST mutate (only way to become different)")
print("    - But agent 2 always copies from itself! (w=1.0)")
print("    - Mutation probability: w[2,2] × mu[2] = 1.0 × 0.001 = 0.001")
print("    - So agent 1 must somehow get agent 3's data...")
print("    - But agent 1 can only copy from {1,2}, not from 3!")
print("    - In State 2, agent 3 has same data as agent 2")
print("    - So agent 1 copying from 2 with mutation is the only way")
print("    - P ≈ w[1,2] × mu[2] = 0.01 × 0.001 = 0.00001")

print("\n" + "-"*80)

print("\nState 3 → State 2: {{2},{1,3}} → {{1},{2,3}}")
print("  Current situation:")
print("    - Agent 2 is alone (has unique data)")
print("    - Agents 1 and 3 share data")
print()
print("  Network (outward flow):")
print("    - Agent 1 can copy from: agent 1 (w=0.99) or agent 2 (w=0.01)")
print("    - Agent 2 can copy from: agent 2 (w=1.0)")
print("    - Agent 3 can copy from: agent 3 (w=0.99) or agent 2 (w=0.01)")
print()
print("  For State 3 → State 2:")
print("    - Need agent 1 to become isolated")
print("    - Need agent 2 to join {2,3} block")
print()
print("  This requires:")
print("    - Agent 3 copies from agent 2: w[3,2] × (1-mu[2]) = 0.01 × 0.999 = 0.00999")
print("    - Agent 2 stays same (always true, w[2,2]=1.0)")
print("    - Agent 1 stays same (keeps its own data)")
print("    - OR agent 1 receives mutation from copying agent 2")
print("    - Main pathway: agent 3 copies from center")
print("    - P ≈ 0.01 × 0.999 = 0.00999")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

print("\nThe asymmetry comes from the network structure!")
print("  - In State 3, agent 3 can easily copy from center (agent 2)")
print("  - This moves State 3 → State 2 with probability ≈ 0.01")
print()
print("  - In State 2, agent 1 can copy from center (agent 2)")
print("  - But this MAINTAINS State 2 (they already share data with agent 3)")
print("  - To go State 2 → State 3 requires mutation")
print("  - This has probability ≈ 0.01 × 0.001 = 0.00001")
print()
print(f"Ratio: {prob_3to2_val / prob_2to3_val:.0f}x difference")

print("\n" + "="*80)
print("IS THIS CORRECT?")
print("="*80)

print("\nThe question is: does this asymmetry make physical sense?")
print()
print("State 2: {{1},{2,3}} - endpoints share data (via center)")
print("State 3: {{2},{1,3}} - endpoints share data (NOT via center)")
print()
print("In an OUTWARD flow network:")
print("  - Information flows from center to periphery")
print("  - Endpoints cannot directly communicate")
print("  - Endpoints can only match via center")
print()
print("So State 3 {{2},{1,3}} is UNSTABLE:")
print("  - Endpoints share data, but center has different data")
print("  - This configuration is unlikely to persist")
print("  - Endpoints will quickly adopt center's data")
print("  - This moves to State 2 {{1},{2,3}} or State 4 {{3},{1,2}}")
print()
print("✓ The asymmetry P(3→2) >> P(2→3) makes sense!")
print("✓ This explains why π₂ > π₃ despite P(1→3) > P(1→2)")
