"""
Quick numerical comparison of IBD F-matrix with symbolic distances at N=1
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

# Load results
M, case = 3, 'case1'
ibd = load_ibd(M, case)
sym = load_sym(M, case)

# Compute expected distances
E_d = compute_distance_expectations(sym['states'], sym['pi'], M)

# Symbolic variables
N, m, alpha = symbols('N m alpha')

# Test values
N_val, m_val, alpha_val = 1.0, 0.01, 0.001

print(f"\nTesting: N={N_val}, m={m_val}, α={alpha_val}\n")
print(f"Relationship: E[d_ij] should equal 1 - F_ij(N=1)\n")
print("="*70)

for i in range(1, M+1):
    for j in range(i+1, M+1):
        # F_ij from IBD_analysis
        f_expr = ibd['F_matrix'][i-1, j-1]
        f_func = lambdify((N, m, alpha), f_expr, 'numpy')
        f_val = float(f_func(N_val, m_val, alpha_val))

        # E[d_ij] from symbolic_analysis
        e_d_expr = E_d[(i, j)]
        e_d_func = lambdify((m, alpha), e_d_expr, 'numpy')
        e_d_val = float(e_d_func(m_val, alpha_val))

        # Compare
        one_minus_f = 1.0 - f_val
        diff = abs(e_d_val - one_minus_f)

        status = "✓" if diff < 1e-6 else "✗"
        print(f"  ({i},{j}): E[d] = {e_d_val:.8f}, 1-F(N=1) = {one_minus_f:.8f}, diff = {diff:.2e} {status}")

print("="*70)
