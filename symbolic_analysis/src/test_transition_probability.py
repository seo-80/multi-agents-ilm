#!/usr/bin/env python3
"""
Test transition_probability function to verify correctness.

This script tests:
1. Sum of transition probabilities from each state equals 1
2. Specific transition probability calculations
3. Edge cases
"""

import sympy
from sympy import symbols, simplify, N as numerical_eval
from m_agent_stationary_symbolic import (
    get_all_partitions,
    build_W_matrix,
    build_alpha_vector,
    compute_mutation_rates,
    transition_probability,
    partition_to_string,
    prob_copy_block,
    prob_receive_mutation_from,
)


def test_probability_sum(M=3, center_prestige=False, centralized_neologism_creation=False):
    """
    Test that sum of transition probabilities from each state equals 1.
    """
    print("="*80)
    print(f"TEST 1: Probability Sum Test (M={M})")
    print("="*80)

    # Setup
    m, alpha = symbols('m alpha', real=True, positive=True)
    W = build_W_matrix(M, center_prestige, m)
    alpha_vec = build_alpha_vector(M, centralized_neologism_creation, alpha)
    mu_vec = compute_mutation_rates(alpha_vec)
    states = get_all_partitions(M)

    print(f"\nNumber of states: {len(states)}")
    print("\nStates:")
    for i, state in enumerate(states):
        print(f"  {i+1}. {partition_to_string(state)}")

    # Test each state
    print("\n" + "-"*80)
    print("Checking transition probability sums...")
    print("-"*80)

    all_pass = True
    for i, S in enumerate(states):
        print(f"\nState {i+1}: {partition_to_string(S)}")

        # Calculate sum of all transition probabilities from this state
        total_prob = sympy.Integer(0)
        for j, S_prime in enumerate(states):
            prob = transition_probability(S, S_prime, W, mu_vec, M)
            total_prob += prob

        # Simplify
        total_prob = simplify(total_prob)

        print(f"  Sum of P(S'|S) = {total_prob}")

        # Check if equals 1
        if total_prob != 1:
            print(f"  ❌ FAILED: Sum is not 1!")
            all_pass = False
        else:
            print(f"  ✓ PASSED")

    print("\n" + "="*80)
    if all_pass:
        print("✓ ALL TESTS PASSED: All probability sums equal 1")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*80)

    return all_pass


def test_specific_transitions(M=3):
    """
    Test specific transition probabilities with known results.
    """
    print("\n" + "="*80)
    print(f"TEST 2: Specific Transition Probabilities (M={M})")
    print("="*80)

    # Setup
    m, alpha = symbols('m alpha', real=True, positive=True)
    W = build_W_matrix(M, False, m)  # Symmetric
    alpha_vec = build_alpha_vector(M, False, alpha)  # Uniform
    mu_vec = compute_mutation_rates(alpha_vec)

    print("\nW matrix:")
    for i in range(M):
        row = [str(W[i, j]) for j in range(M)]
        print(f"  {' '.join(row)}")

    print("\nMutation rates:")
    for i, mu in enumerate(mu_vec, 1):
        print(f"  μ_{i} = {mu}")

    # Test case 1: S = {{1,2,3}} -> S' = {{1,2,3}}
    # All agents copy from {1,2,3}
    print("\n" + "-"*80)
    print("Test Case 1: S = {{1,2,3}} -> S' = {{1,2,3}}")
    print("-"*80)

    S = frozenset([frozenset([1,2,3])])
    S_prime = frozenset([frozenset([1,2,3])])

    prob = transition_probability(S, S_prime, W, mu_vec, M)
    prob_simplified = simplify(prob)

    print(f"P(S'|S) = {prob_simplified}")

    # Manual calculation
    # Each agent must copy from {1,2,3} without mutation
    # Agent i: Σ_{j∈{1,2,3}} W[i,j] * (1-μ_j)
    manual_prob = sympy.Integer(1)
    for i in range(1, M+1):
        agent_prob = prob_copy_block(i, frozenset([1,2,3]), W, mu_vec)
        manual_prob *= agent_prob
    manual_prob = simplify(manual_prob)

    print(f"Manual calculation: {manual_prob}")
    print(f"Match: {simplify(prob_simplified - manual_prob) == 0}")

    # Test case 2: S = {{1,2,3}} -> S' = {{1},{2,3}}
    print("\n" + "-"*80)
    print("Test Case 2: S = {{1,2,3}} -> S' = {{1},{2,3}}")
    print("-"*80)

    S = frozenset([frozenset([1,2,3])])
    S_prime = frozenset([frozenset([1]), frozenset([2,3])])

    prob = transition_probability(S, S_prime, W, mu_vec, M)
    prob_simplified = simplify(prob)

    print(f"P(S'|S) = {prob_simplified}")

    # Manual calculation:
    # {2,3} must copy from {1,2,3} (no mutation)
    # {1} must come from mutation (since {1,2,3} is used by {2,3})
    manual_prob = sympy.Integer(1)

    # {2,3} copies from {1,2,3}
    for agent in [2, 3]:
        manual_prob *= prob_copy_block(agent, frozenset([1,2,3]), W, mu_vec)

    # {1} receives mutation
    mutation_prob = sum(
        prob_receive_mutation_from(1, j, W, mu_vec)
        for j in range(1, M+1)
    )
    manual_prob *= mutation_prob
    manual_prob = simplify(manual_prob)

    print(f"Manual calculation: {manual_prob}")
    print(f"Match: {simplify(prob_simplified - manual_prob) == 0}")

    # Test case 3: S = {{1},{2,3}} -> S' = {{1},{2,3}}
    print("\n" + "-"*80)
    print("Test Case 3: S = {{1},{2,3}} -> S' = {{1},{2,3}}")
    print("-"*80)

    S = frozenset([frozenset([1]), frozenset([2,3])])
    S_prime = frozenset([frozenset([1]), frozenset([2,3])])

    prob = transition_probability(S, S_prime, W, mu_vec, M)
    prob_simplified = simplify(prob)

    print(f"P(S'|S) = {prob_simplified}")
    print("This should be sum of multiple mappings:")
    print("  - {2,3}' <- {2,3}, {1}' <- {1}")
    print("  - {2,3}' <- {2,3}, {1}' <- mutation")
    print("  - {2,3}' <- {1}, {1}' <- {2,3} (but {1} is singleton, can't produce {2,3})")
    print("  - {1}' <- {2,3}, {2,3}' <- {1} (symmetric)")

    # Test case 4: S = {{1},{2},{3}} -> S' = {{1,2,3}}
    print("\n" + "-"*80)
    print("Test Case 4: S = {{1},{2},{3}} -> S' = {{1,2,3}}")
    print("-"*80)

    S = frozenset([frozenset([1]), frozenset([2]), frozenset([3])])
    S_prime = frozenset([frozenset([1,2,3])])

    prob = transition_probability(S, S_prime, W, mu_vec, M)
    prob_simplified = simplify(prob)

    print(f"P(S'|S) = {prob_simplified}")
    print("All agents must copy from the same singleton block")


def test_numerical_values():
    """
    Test with specific numerical values to verify probabilities are valid.
    """
    print("\n" + "="*80)
    print("TEST 3: Numerical Value Test")
    print("="*80)

    M = 3
    m_val = 0.3
    alpha_val = 0.5

    print(f"\nParameters: M={M}, m={m_val}, α={alpha_val}")

    # Setup
    m, alpha = symbols('m alpha', real=True, positive=True)
    W = build_W_matrix(M, False, m)
    alpha_vec = build_alpha_vector(M, False, alpha)
    mu_vec = compute_mutation_rates(alpha_vec)
    states = get_all_partitions(M)

    print(f"Number of states: {len(states)}")

    # Substitute numerical values
    subs_dict = {m: m_val, alpha: alpha_val}

    print("\n" + "-"*80)
    print("Checking numerical probabilities...")
    print("-"*80)

    all_pass = True
    for i, S in enumerate(states):
        print(f"\nState {i+1}: {partition_to_string(S)}")

        total_prob = 0.0
        for j, S_prime in enumerate(states):
            prob = transition_probability(S, S_prime, W, mu_vec, M)
            prob_num = float(numerical_eval(prob.subs(subs_dict)))

            # Check if probability is valid (between 0 and 1)
            if prob_num < -1e-10 or prob_num > 1 + 1e-10:
                print(f"  ❌ Invalid probability to state {j+1}: {prob_num}")
                all_pass = False

            total_prob += prob_num

        print(f"  Sum = {total_prob:.10f}")

        # Check if sum is close to 1
        if abs(total_prob - 1.0) > 1e-8:
            print(f"  ❌ FAILED: Sum deviates from 1 by {abs(total_prob - 1.0)}")
            all_pass = False
        else:
            print(f"  ✓ PASSED")

    print("\n" + "="*80)
    if all_pass:
        print("✓ ALL NUMERICAL TESTS PASSED")
    else:
        print("❌ SOME NUMERICAL TESTS FAILED")
    print("="*80)

    return all_pass


def test_edge_cases():
    """
    Test edge cases and boundary conditions.
    """
    print("\n" + "="*80)
    print("TEST 4: Edge Cases")
    print("="*80)

    M = 3
    m, alpha = symbols('m alpha', real=True, positive=True)
    W = build_W_matrix(M, False, m)
    alpha_vec = build_alpha_vector(M, False, alpha)
    mu_vec = compute_mutation_rates(alpha_vec)

    # Test: S = {{1},{2},{3}} -> S' = {{1},{2},{3}}
    print("\n" + "-"*80)
    print("Edge Case 1: All singletons -> All singletons")
    print("-"*80)

    S = frozenset([frozenset([1]), frozenset([2]), frozenset([3])])
    S_prime = frozenset([frozenset([1]), frozenset([2]), frozenset([3])])

    prob = transition_probability(S, S_prime, W, mu_vec, M)
    prob_simplified = simplify(prob)

    print(f"S = {partition_to_string(S)}")
    print(f"S' = {partition_to_string(S_prime)}")
    print(f"P(S'|S) = {prob_simplified}")
    print("\nEach singleton can:")
    print("  - Copy from any of the 3 singleton blocks")
    print("  - Receive a new mutation")
    print("But mappings must be injective!")


def test_mapping_enumeration():
    """
    Test and visualize the mapping enumeration for various transitions.
    """
    from m_agent_stationary_symbolic import enumerate_valid_mappings

    print("\n" + "="*80)
    print("TEST 5: Mapping Enumeration Visualization")
    print("="*80)

    # Test case 1: S = {{1,2,3}} -> S' = {{1,2,3}}
    print("\n" + "-"*80)
    print("Case 1: S = {{1,2,3}} -> S' = {{1,2,3}}")
    print("-"*80)

    S = frozenset([frozenset([1,2,3])])
    S_prime = frozenset([frozenset([1,2,3])])

    mappings = list(enumerate_valid_mappings(S, S_prime))
    print(f"Number of mappings: {len(mappings)}")

    for i, (non_sing_map, sing_map) in enumerate(mappings, 1):
        print(f"\nMapping {i}:")
        print(f"  Non-singletons: {format_mapping(non_sing_map, S_prime, False)}")
        print(f"  Singletons: {format_mapping(sing_map, S_prime, True)}")

    # Test case 2: S = {{1,2,3}} -> S' = {{1},{2,3}}
    print("\n" + "-"*80)
    print("Case 2: S = {{1,2,3}} -> S' = {{1},{2,3}}")
    print("-"*80)

    S = frozenset([frozenset([1,2,3])])
    S_prime = frozenset([frozenset([1]), frozenset([2,3])])

    mappings = list(enumerate_valid_mappings(S, S_prime))
    print(f"Number of mappings: {len(mappings)}")

    for i, (non_sing_map, sing_map) in enumerate(mappings, 1):
        print(f"\nMapping {i}:")
        print(f"  Non-singletons: {format_mapping(non_sing_map, S_prime, False)}")
        print(f"  Singletons: {format_mapping(sing_map, S_prime, True)}")

    # Test case 3: S = {{1},{2,3}} -> S' = {{1},{2,3}}
    print("\n" + "-"*80)
    print("Case 3: S = {{1},{2,3}} -> S' = {{1},{2,3}}")
    print("-"*80)

    S = frozenset([frozenset([1]), frozenset([2,3])])
    S_prime = frozenset([frozenset([1]), frozenset([2,3])])

    mappings = list(enumerate_valid_mappings(S, S_prime))
    print(f"Number of mappings: {len(mappings)}")

    for i, (non_sing_map, sing_map) in enumerate(mappings, 1):
        print(f"\nMapping {i}:")
        print(f"  Non-singletons: {format_mapping(non_sing_map, S_prime, False)}")
        print(f"  Singletons: {format_mapping(sing_map, S_prime, True)}")

    # Test case 4: S = {{1},{2},{3}} -> S' = {{1,2,3}}
    print("\n" + "-"*80)
    print("Case 4: S = {{1},{2},{3}} -> S' = {{1,2,3}}")
    print("-"*80)

    S = frozenset([frozenset([1]), frozenset([2]), frozenset([3])])
    S_prime = frozenset([frozenset([1,2,3])])

    mappings = list(enumerate_valid_mappings(S, S_prime))
    print(f"Number of mappings: {len(mappings)}")

    for i, (non_sing_map, sing_map) in enumerate(mappings, 1):
        print(f"\nMapping {i}:")
        print(f"  Non-singletons: {format_mapping(non_sing_map, S_prime, False)}")
        print(f"  Singletons: {format_mapping(sing_map, S_prime, True)}")

    # Test case 5: S = {{1},{2},{3}} -> S' = {{1},{2},{3}}
    print("\n" + "-"*80)
    print("Case 5: S = {{1},{2},{3}} -> S' = {{1},{2},{3}}")
    print("-"*80)

    S = frozenset([frozenset([1]), frozenset([2]), frozenset([3])])
    S_prime = frozenset([frozenset([1]), frozenset([2]), frozenset([3])])

    mappings = list(enumerate_valid_mappings(S, S_prime))
    print(f"Number of mappings: {len(mappings)}")
    print("(Showing first 10 mappings only)")

    for i, (non_sing_map, sing_map) in enumerate(mappings[:10], 1):
        print(f"\nMapping {i}:")
        print(f"  Non-singletons: {format_mapping(non_sing_map, S_prime, False)}")
        print(f"  Singletons: {format_mapping(sing_map, S_prime, True)}")

    # Test case 6: S = {{1,2},{3}} -> S' = {{1},{2},{3}}
    print("\n" + "-"*80)
    print("Case 6: S = {{1,2},{3}} -> S' = {{1},{2},{3}}")
    print("-"*80)

    S = frozenset([frozenset([1,2]), frozenset([3])])
    S_prime = frozenset([frozenset([1]), frozenset([2]), frozenset([3])])

    mappings = list(enumerate_valid_mappings(S, S_prime))
    print(f"Number of mappings: {len(mappings)}")

    for i, (non_sing_map, sing_map) in enumerate(mappings, 1):
        print(f"\nMapping {i}:")
        print(f"  Non-singletons: {format_mapping(non_sing_map, S_prime, False)}")
        print(f"  Singletons: {format_mapping(sing_map, S_prime, True)}")


def format_mapping(mapping, S_prime, is_singleton):
    """
    Format a mapping for display.

    Args:
        mapping: Dictionary of block index -> target
        S_prime: State S'
        is_singleton: True if formatting singleton mapping

    Returns:
        Formatted string
    """
    if not mapping:
        return "None"

    # Get blocks in order
    if is_singleton:
        blocks = [b for b in S_prime if len(b) == 1]
    else:
        blocks = [b for b in S_prime if len(b) > 1]

    items = []
    for idx, block in enumerate(blocks):
        if idx in mapping:
            target = mapping[idx]
            if target == 'mutation':
                target_str = "mutation"
            else:
                target_str = partition_to_string(frozenset([target]))
            block_str = partition_to_string(frozenset([block]))
            items.append(f"{block_str} -> {target_str}")

    return ", ".join(items) if items else "None"


def main():
    """
    Run all tests.
    """
    print("\n" + "="*80)
    print("TRANSITION PROBABILITY VERIFICATION TESTS")
    print("="*80)

    # Test 1: Symbolic probability sums
    result1 = test_probability_sum(M=3)

    # Test 2: Specific transitions
    test_specific_transitions(M=3)

    # Test 3: Numerical values
    result3 = test_numerical_values()

    # Test 4: Edge cases
    test_edge_cases()

    # Test 5: Mapping enumeration visualization
    test_mapping_enumeration()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Test 1 (Probability Sum): {'✓ PASSED' if result1 else '❌ FAILED'}")
    print(f"Test 3 (Numerical Values): {'✓ PASSED' if result3 else '❌ FAILED'}")
    print("="*80)


if __name__ == '__main__':
    main()
