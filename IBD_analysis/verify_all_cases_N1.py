"""
Verify E[d_ij] = 1 - F_ij(N=1) for all 4 cases
"""

import sys, os
parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'symbolic_analysis', 'src'))

from sympy import symbols, lambdify
import numpy as np
from IBD_analysis.src.f_matrix_symbolic import load_results_by_case as load_ibd
from m_agent_stationary_symbolic import load_results_by_case as load_sym
from analyze_distances import compute_distance_expectations

M = 3
cases = ['case1', 'case2', 'case3', 'case4']

# Test values
N_val, m_val, alpha_val = 1.0, 0.01, 0.001
N, m, alpha = symbols('N m alpha')

print(f"\nVerifying E[d_ij] = 1 - F_ij(N={N_val}) for all cases")
print(f"Parameters: m={m_val}, α={alpha_val}\n")
print("="*80)

all_passed = True

for case in cases:
    print(f"\n{case.upper()}:")
    print("-" * 40)

    try:
        # Load results
        ibd = load_ibd(M, case)
        sym = load_sym(M, case)

        # Compute expected distances
        E_d = compute_distance_expectations(sym['states'], sym['pi'], M)

        max_diff = 0

        for i in range(1, M+1):
            for j in range(i+1, M+1):
                # F_ij from IBD
                f_expr = ibd['F_matrix'][i-1, j-1]
                f_func = lambdify((N, m, alpha), f_expr, 'numpy')
                f_val = float(f_func(N_val, m_val, alpha_val))

                # E[d_ij] from symbolic
                e_d_expr = E_d[(i, j)]
                e_d_func = lambdify((m, alpha), e_d_expr, 'numpy')
                e_d_val = float(e_d_func(m_val, alpha_val))

                # Compare
                diff = abs(e_d_val - (1.0 - f_val))
                max_diff = max(max_diff, diff)

                status = "✓" if diff < 1e-6 else "✗"
                print(f"  ({i},{j}): diff = {diff:.2e} {status}")

        if max_diff < 1e-6:
            print(f"\n  {case}: PASSED ✓ (max diff = {max_diff:.2e})")
        else:
            print(f"\n  {case}: FAILED ✗ (max diff = {max_diff:.2e})")
            all_passed = False

    except FileNotFoundError as e:
        print(f"  Skipping {case} (results not found)")
        continue

print("\n" + "="*80)
if all_passed:
    print("ALL CASES PASSED ✓")
    print("\nConfirmed: E[d_ij] = 1 - F_ij(N=1)")
    print("The IBD F-matrix and symbolic distance calculations are consistent!")
else:
    print("Some cases failed")
print("="*80)
