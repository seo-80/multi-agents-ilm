#!/usr/bin/env python3
"""
Debug script to understand transition probability calculation
"""

import sympy
from sympy import symbols, simplify
from m_agent_stationary_symbolic import (
    get_all_partitions,
    build_W_matrix,
    build_alpha_vector,
    compute_mutation_rates,
    transition_probability,
    partition_to_string,
)


def debug_single_transition():
    """
    Debug a single state's transitions in detail
    """
    M = 3
    m, alpha = symbols('m alpha', real=True, positive=True)
    W = build_W_matrix(M, False, m)
    alpha_vec = build_alpha_vector(M, False, alpha)
    mu_vec = compute_mutation_rates(alpha_vec)
    states = get_all_partitions(M)

    print("="*80)
    print("DEBUGGING TRANSITION PROBABILITIES")
    print("="*80)

    # Focus on S = {{1,2,3}}
    S = frozenset([frozenset([1,2,3])])
    print(f"\nSource state S = {partition_to_string(S)}")
    print("\nCalculating transition probabilities to all states:")
    print("-"*80)

    total = sympy.Integer(0)
    for i, S_prime in enumerate(states):
        prob = transition_probability(S, S_prime, W, mu_vec, M)
        prob_simp = simplify(prob)
        total += prob_simp

        print(f"\n{i+1}. S' = {partition_to_string(S_prime)}")
        print(f"   P(S'|S) = {prob_simp}")

    print("\n" + "="*80)
    print(f"TOTAL = {simplify(total)}")
    print("="*80)

    # Try specific numerical values
    print("\nNumerical check with m=0.3, alpha=0.5:")
    subs_dict = {m: 0.3, alpha: 0.5}
    total_num = float(total.subs(subs_dict))
    print(f"Total = {total_num}")

    print("\nIndividual probabilities:")
    for i, S_prime in enumerate(states):
        prob = transition_probability(S, S_prime, W, mu_vec, M)
        prob_num = float(prob.subs(subs_dict))
        print(f"  {i+1}. {partition_to_string(S_prime)}: {prob_num:.6f}")


if __name__ == '__main__':
    debug_single_transition()
