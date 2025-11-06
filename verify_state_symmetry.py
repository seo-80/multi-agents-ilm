"""
Verify that State 2 and State 4 have equal stationary probabilities in symbolic analysis

State 2: {{1},{2,3}} - Agent 0 (left) isolated, Agents 1,2 (center, right) share
State 4: {{3},{1,2}} - Agent 2 (right) isolated, Agents 0,1 (left, center) share

For CASE 4 (outward + center-only), these should be symmetric.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'symbolic_analysis/src'))

from m_agent_stationary_symbolic import load_results_by_case

print("="*80)
print("Verifying State 2 vs State 4 symmetry")
print("="*80)

# Load CASE 4 results
results = load_results_by_case(3, "case4")
states = results['states']
pi = results['pi']
W = results['W']
mu_vec = results['mu_vec']

print("\nStates:")
for i, state in enumerate(states):
    state_str = str(state).replace('frozenset', '').replace('(', '').replace(')', '')
    print(f"  State {i+1}: {state_str}")

# Extract symbols
m_symbol = [s for s in pi[0].free_symbols if str(s) == 'm'][0]
alpha_symbol = [s for s in pi[0].free_symbols if str(s) == 'alpha'][0]

# Evaluate at m=0.01, alpha=0.001
m_val = 0.01
alpha_val = 0.001

print(f"\nEvaluating at m={m_val}, alpha={alpha_val}")

pi_numeric = []
for i in range(len(pi)):
    pi_expr = pi[i].subs({m_symbol: m_val, alpha_symbol: alpha_val}).evalf()
    pi_val = float(pi_expr) if pi_expr.is_real else float(pi_expr.as_real_imag()[0])
    pi_numeric.append(pi_val)

print("\nStationary distribution:")
for i, p in enumerate(pi_numeric):
    state_str = str(states[i]).replace('frozenset', '').replace('(', '').replace(')', '')
    print(f"  π_{i+1} = {p:.10f}  {state_str}")

print("\n" + "="*80)
print("Symmetry check:")
print("="*80)

# State 2 vs State 4
pi_2 = pi_numeric[1]  # State 2 (0-indexed as 1)
pi_4 = pi_numeric[3]  # State 4 (0-indexed as 3)

print(f"\nπ_2 ({{1}},{{2,3}}): {pi_2:.10f}")
print(f"π_4 ({{3}},{{1,2}}): {pi_4:.10f}")
print(f"Difference: {abs(pi_2 - pi_4):.2e}")

if abs(pi_2 - pi_4) < 1e-10:
    print("✓ π_2 = π_4 (symmetric)")
else:
    print(f"✗ π_2 ≠ π_4 (NOT symmetric!)")

# Check symbolic expressions directly
print("\n" + "="*80)
print("Symbolic expressions:")
print("="*80)

print(f"\nπ_2 (symbolic):")
print(f"  {pi[1]}")

print(f"\nπ_4 (symbolic):")
print(f"  {pi[3]}")

# Check if they're equal symbolically
from sympy import simplify
diff = simplify(pi[1] - pi[3])

print(f"\nπ_2 - π_4 (simplified):")
print(f"  {diff}")

if diff == 0:
    print("  ✓ Symbolically equal")
else:
    print("  ⚠ NOT symbolically equal (but may be numerically close)")

print("\n" + "="*80)
print("Network structure analysis:")
print("="*80)

print("\nW matrix (symbolic):")
for i in range(3):
    print(f"  Agent {i}: {[str(W[i,j]) for j in range(3)]}")

print("\nW matrix (numeric):")
W_numeric = W.subs({m_symbol: m_val})
for i in range(3):
    print(f"  Agent {i}: {[f'{float(W_numeric[i,j]):.4f}' for j in range(3)]}")

print("\nAgent roles:")
print("  Agent 0: Left endpoint")
print("  Agent 1: Center")
print("  Agent 2: Right endpoint")

print("\nMutation rates:")
mu_numeric = [mu_vec[i].subs({alpha_symbol: alpha_val}) for i in range(3)]
for i in range(3):
    print(f"  Agent {i}: μ = {float(mu_numeric[i]):.6f}")

print("\nState 2 {{1},{2,3}}:")
print("  - Agent 0 (left) is isolated")
print("  - Agents 1,2 (center, right) share data")
print("  - To exit: Agent 0 must adopt center's data (prob ≈ 0.01)")
print("           OR center must mutate while Agent 0 copies (prob ≈ 0.01 × 0.001)")

print("\nState 4 {{3},{1,2}}:")
print("  - Agent 2 (right) is isolated")
print("  - Agents 0,1 (left, center) share data")
print("  - To exit: Agent 2 must adopt center's data (prob ≈ 0.01)")
print("           OR center must mutate while Agent 2 copies (prob ≈ 0.01 × 0.001)")

print("\n✓ By symmetry, π_2 should equal π_4")
