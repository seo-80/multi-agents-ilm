"""
Compare simulation and symbolic analysis for all cases

This script analyzes simulation results for different cases and compares
with symbolic analysis predictions.
"""

import numpy as np
import glob
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'symbolic_analysis/src'))

from m_agent_stationary_symbolic import load_results_by_case

def analyze_simulation(sim_dir, case_name):
    """Analyze simulation results from a directory"""

    if not os.path.exists(sim_dir):
        print(f"  ✗ Directory not found: {sim_dir}")
        return None

    distance_files = sorted(
        glob.glob(os.path.join(sim_dir, "distance_*.npy")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1])
    )

    if len(distance_files) == 0:
        print(f"  ✗ No distance files found in {sim_dir}")
        return None

    # Load all distances
    all_distances = []
    for f in distance_files:
        dist = np.load(f)
        all_distances.append(dist)

    all_distances = np.array(all_distances)

    # Average distances over time (excluding initial transient)
    n_samples = all_distances.shape[0]
    burn_in = max(1, n_samples // 5)  # At least skip first sample

    avg_distances = np.mean(all_distances[burn_in:], axis=0)

    d_12 = avg_distances[0, 1]
    d_13 = avg_distances[0, 2]
    d_23 = avg_distances[1, 2]

    return {
        'n_samples': n_samples,
        'burn_in': burn_in,
        'd_12': d_12,
        'd_13': d_13,
        'd_23': d_23,
        'sign': 'E[d_12] < E[d_13]' if d_12 < d_13 else 'E[d_12] > E[d_13]' if d_12 > d_13 else 'E[d_12] = E[d_13]'
    }

def analyze_symbolic(case_num, m_val=0.01, alpha_val=0.001):
    """Analyze symbolic results for a case"""

    case_name = f"case{case_num}"

    try:
        results = load_results_by_case(3, case_name)
    except FileNotFoundError:
        print(f"  ✗ Symbolic results not found for {case_name}")
        return None

    states = results['states']
    pi = results['pi']

    # Find symbols
    m_symbol = None
    alpha_symbol = None
    for s in pi[0].free_symbols:
        if str(s) == 'm':
            m_symbol = s
        elif str(s) == 'alpha':
            alpha_symbol = s

    if m_symbol is None or alpha_symbol is None:
        print(f"  ✗ Could not find symbols in {case_name}")
        return None

    # Evaluate pi numerically
    pi_numeric = []
    for i in range(len(pi)):
        pi_expr = pi[i].subs({m_symbol: m_val, alpha_symbol: alpha_val}).evalf()
        pi_val = float(pi_expr) if pi_expr.is_real else float(pi_expr.as_real_imag()[0])
        pi_numeric.append(pi_val)

    # Compute expected distances
    # State 1: {{1,2,3}} - all same
    # State 2: {{1},{2,3}} - agents 2,3 same
    # State 3: {{2},{1,3}} - agents 1,3 same
    # State 4: {{3},{1,2}} - agents 1,2 same
    # State 5: {{1},{2},{3}} - all different

    # E[d_12] = P(agents 1,2 different) = π_2 + π_3 + π_5
    # E[d_13] = P(agents 1,3 different) = π_2 + π_4 + π_5
    # E[d_23] = P(agents 2,3 different) = π_3 + π_4 + π_5

    d_12 = pi_numeric[1] + pi_numeric[2] + pi_numeric[4]
    d_13 = pi_numeric[1] + pi_numeric[3] + pi_numeric[4]
    d_23 = pi_numeric[2] + pi_numeric[3] + pi_numeric[4]

    return {
        'pi': pi_numeric,
        'd_12': d_12,
        'd_13': d_13,
        'd_23': d_23,
        'sign': 'E[d_12] < E[d_13]' if d_12 < d_13 else 'E[d_12] > E[d_13]' if d_12 > d_13 else 'E[d_12] = E[d_13]'
    }

# Main comparison
print("="*80)
print("COMPARING SIMULATION AND SYMBOLIC ANALYSIS FOR ALL CASES")
print("="*80)

cases = [
    {
        'num': 3,
        'name': 'CASE 3: Bidirectional + Center-only',
        'sim_dir': 'data/naive_simulation/raw/bidirectional_flow-nonzero_alpha_center_fr_0.01_agents_3_N_i_1_alpha_0.001'
    },
    {
        'num': 4,
        'name': 'CASE 4: Outward + Center-only',
        'sim_dir': 'data/naive_simulation/raw/outward_flow-nonzero_alpha_center_fr_0.01_agents_3_N_i_1_alpha_0.001'
    }
]

for case in cases:
    print(f"\n{case['name']}")
    print("-"*80)

    # Analyze simulation
    print("\nSimulation:")
    sim_result = analyze_simulation(case['sim_dir'], case['name'])

    if sim_result:
        print(f"  Samples: {sim_result['n_samples']} (burn-in: {sim_result['burn_in']})")
        print(f"  E[d_12] = {sim_result['d_12']:.6f}")
        print(f"  E[d_13] = {sim_result['d_13']:.6f}")
        print(f"  E[d_23] = {sim_result['d_23']:.6f}")
        print(f"  => {sim_result['sign']}")

    # Analyze symbolic
    print("\nSymbolic Analysis:")
    sym_result = analyze_symbolic(case['num'])

    if sym_result:
        print(f"  E[d_12] = {sym_result['d_12']:.6f}")
        print(f"  E[d_13] = {sym_result['d_13']:.6f}")
        print(f"  E[d_23] = {sym_result['d_23']:.6f}")
        print(f"  => {sym_result['sign']}")

    # Compare
    if sim_result and sym_result:
        print("\nComparison:")
        if sim_result['sign'] == sym_result['sign']:
            print(f"  ✓ SIGNS MATCH: {sim_result['sign']}")
        else:
            print(f"  ✗ SIGNS DIFFER:")
            print(f"    Simulation: {sim_result['sign']}")
            print(f"    Symbolic:   {sym_result['sign']}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("\nKey finding:")
print("  - CASE 3 (bidirectional + center): Symmetric network")
print("    → State 3 {{2},{1,3}} is equally stable as State 2 {{1},{2,3}}")
print("    → Expect E[d_12] ≈ E[d_13] (or possibly E[d_12] > E[d_13])")
print()
print("  - CASE 4 (outward + center): Asymmetric network")
print("    → State 3 {{2},{1,3}} is UNSTABLE (endpoints share non-center data)")
print("    → Expect E[d_12] < E[d_13]")
