"""
Verify symbolic_analysis numerical results at m=0.01, alpha=0.001

This script loads the symbolic results and evaluates them numerically
to compare with simulation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'symbolic_analysis/src'))

from m_agent_stationary_symbolic import load_results_by_case
from analyze_distances import compute_distance_expectations
from sympy import symbols

# Load results
print("Loading symbolic_analysis results for M=3, case4...")
results = load_results_by_case(3, "case4")
states = results['states']
pi = results['pi']

# Parameters
m, alpha = symbols('m alpha')
m_val = 0.01
alpha_val = 0.001

print(f"\nEvaluating at m={m_val}, alpha={alpha_val}")
print("="*80)

# Evaluate stationary distribution
print("\nStationary distribution (numerical):")
pi_numerical = []
for i, state in enumerate(states):
    state_str = str(state).replace('frozenset', '').replace('(', '').replace(')', '')
    pi_val = float(pi[i].subs({m: m_val, alpha: alpha_val}).n())
    pi_numerical.append(pi_val)
    print(f"  π_{i+1} (State {i+1}): {pi_val:.10f}")

print(f"\nSum of probabilities: {sum(pi_numerical):.10f}")

# Compute expected distances
print("\n" + "="*80)
print("Computing expected distances...")
expected_distances = compute_distance_expectations(states, pi, 3)

# Evaluate numerically
E_d12 = float(expected_distances[(1, 2)].subs({m: m_val, alpha: alpha_val}).n())
E_d13 = float(expected_distances[(1, 3)].subs({m: m_val, alpha: alpha_val}).n())
E_d23 = float(expected_distances[(2, 3)].subs({m: m_val, alpha: alpha_val}).n())

print(f"\nE[d_12] = {E_d12:.10f}")
print(f"E[d_13] = {E_d13:.10f}")
print(f"E[d_23] = {E_d23:.10f}")

print(f"\nE[d_12] - E[d_13] = {E_d12 - E_d13:.10f}")

if E_d12 > E_d13:
    print("  => E[d_12] > E[d_13]")
elif E_d12 < E_d13:
    print("  => E[d_12] < E[d_13]")
else:
    print("  => E[d_12] = E[d_13]")

print("\n" + "="*80)
print("Simulation results:")
print("  E[d_12] ≈ 0.193")
print("  E[d_13] ≈ 0.190")
print("  E[d_12] - E[d_13] ≈ 0.003")
print("  => E[d_12] > E[d_13]")

print("\n" + "="*80)
if E_d12 > E_d13:
    print("✓ MATCH: Symbolic analysis agrees with simulation!")
else:
    print("✗ MISMATCH: Symbolic analysis contradicts simulation!")
    print("\n** BUG CONFIRMED **")

print("\n" + "="*80)
print("Manual verification of E[d_12]:")
print("  E[d_12] = π_2 + π_3 + π_5")
print(f"         = {pi_numerical[1]:.10f} + {pi_numerical[2]:.10f} + {pi_numerical[4]:.10f}")
print(f"         = {pi_numerical[1] + pi_numerical[2] + pi_numerical[4]:.10f}")

print("\nManual verification of E[d_13]:")
print("  E[d_13] = π_2 + π_4 + π_5")
print(f"         = {pi_numerical[1]:.10f} + {pi_numerical[3]:.10f} + {pi_numerical[4]:.10f}")
print(f"         = {pi_numerical[1] + pi_numerical[3] + pi_numerical[4]:.10f}")

print("\nManual verification of difference:")
print(f"  E[d_12] - E[d_13] = π_3 - π_4")
print(f"                    = {pi_numerical[2]:.10f} - {pi_numerical[3]:.10f}")
print(f"                    = {pi_numerical[2] - pi_numerical[3]:.10f}")
print()
print(f"  (Note: π_2 = π_4 due to symmetry)")
print(f"  Therefore: E[d_12] - E[d_13] = π_3 - π_2")
print(f"                               = {pi_numerical[2]:.10f} - {pi_numerical[1]:.10f}")
print(f"                               = {pi_numerical[2] - pi_numerical[1]:.10f}")
