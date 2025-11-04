"""
Example script for loading and analyzing saved symbolic computation results.

This demonstrates how to load previously computed stationary state results
and perform analysis on them.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from m_agent_stationary_symbolic import (
    load_results, load_results_by_case, partition_to_string
)
from sympy import symbols, simplify


def example_load_by_filepath():
    """
    Example: Load results by filepath
    """
    print("="*80)
    print("Example 1: Load by filepath")
    print("="*80)

    # Load using the default path (symbolic_analysis/results)
    results = load_results_by_case(3, "case1")

    # Access metadata
    print("\nMetadata:")
    for key, value in results['metadata'].items():
        print(f"  {key}: {value}")

    # Access states
    print(f"\nNumber of states: {len(results['states'])}")
    print("First 3 states:")
    for i, state in enumerate(results['states'][:3], 1):
        print(f"  {i}. {partition_to_string(state)}")

    # Access stationary distribution
    print("\nStationary distribution:")
    for i, (state, prob) in enumerate(zip(results['states'], results['pi']), 1):
        print(f"  State {i} {partition_to_string(state)}: {simplify(prob)}")

    print()


def example_load_by_case():
    """
    Example: Load results by M and case name
    """
    print("="*80)
    print("Example 2: Load by case")
    print("="*80)

    M = 3
    case_name = "case1"

    results = load_results_by_case(M, case_name)

    print(f"\nLoaded results for M={M}, {case_name}")
    print(f"Number of states: {len(results['states'])}")

    print()


def example_numerical_substitution():
    """
    Example: Substitute numerical values for symbolic parameters
    """
    print("="*80)
    print("Example 3: Numerical substitution")
    print("="*80)

    # Load results
    results = load_results_by_case(3, "case1")

    # Define parameter values
    m_val = 0.3
    alpha_val = 0.1

    print(f"\nSubstituting m={m_val}, alpha={alpha_val}")

    # Get symbolic variables
    m, alpha = symbols('m alpha')

    # Substitute into stationary distribution
    print("\nNumerical stationary distribution:")
    for i, (state, pi_expr) in enumerate(zip(results['states'], results['pi']), 1):
        pi_num = pi_expr.subs({m: m_val, alpha: alpha_val})
        pi_num_float = float(pi_num)
        print(f"  State {i} {partition_to_string(state)}: {pi_num_float:.6f}")

    # Verify sum = 1
    total = sum(float(pi_expr.subs({m: m_val, alpha: alpha_val}))
                for pi_expr in results['pi'])
    print(f"\nSum of probabilities: {total:.6f} (should be 1.0)")

    print()


def example_compare_cases():
    """
    Example: Compare stationary distributions across different cases
    """
    print("="*80)
    print("Example 4: Compare cases")
    print("="*80)

    M = 3
    cases = ["case1", "case2", "case3", "case4"]

    # Load all cases
    all_results = {}
    for case in cases:
        try:
            all_results[case] = load_results_by_case(M, case)
            print(f"  Loaded {case}")
        except FileNotFoundError:
            print(f"  {case} not found (not yet computed)")

    if not all_results:
        print("\nNo results found. Please run test_m_agent_stationary.py first.")
        return

    # Compare probability of full synchronization (all agents same state)
    print(f"\nComparing probability of full synchronization {{1,2,3}}:")

    m, alpha = symbols('m alpha')
    m_val = 0.3
    alpha_val = 0.1

    for case, results in all_results.items():
        states = results['states']
        pi = results['pi']

        # Find state where all agents are synchronized
        for i, state in enumerate(states):
            if len(state) == 1:  # Single block = all agents same
                prob_expr = pi[i]
                prob_num = float(prob_expr.subs({m: m_val, alpha: alpha_val}))
                print(f"  {case}: {prob_num:.6f}")
                break

    print()


def example_access_matrices():
    """
    Example: Access and examine W and P matrices
    """
    print("="*80)
    print("Example 5: Access matrices")
    print("="*80)

    results = load_results_by_case(3, "case1")

    # Weight matrix W
    print("\nWeight matrix W:")
    W = results['W']
    M = results['metadata']['M']
    for i in range(M):
        row_str = "  ".join(str(simplify(W[i, j])) for j in range(M))
        print(f"  {row_str}")

    # Transition matrix P (showing just first row)
    print("\nTransition matrix P (first row):")
    P = results['P']
    n_states = len(results['states'])
    row_str = "  ".join(str(simplify(P[0, j])) for j in range(n_states))
    print(f"  {row_str}")

    # Innovation parameters
    print("\nInnovation parameters:")
    for i, alpha_i in enumerate(results['alpha_vec'], 1):
        print(f"  α_{i} = {alpha_i}")

    # Mutation rates
    print("\nMutation rates:")
    for i, mu_i in enumerate(results['mu_vec'], 1):
        print(f"  μ_{i} = {simplify(mu_i)}")

    print()


if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# Loading and Analyzing Symbolic Computation Results")
    print("#"*80 + "\n")

    # Run examples
    try:
        example_load_by_filepath()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run test_m_agent_stationary.py first to generate results.\n")

    try:
        example_load_by_case()
    except FileNotFoundError as e:
        print(f"Error: {e}\n")

    try:
        example_numerical_substitution()
    except FileNotFoundError as e:
        print(f"Error: {e}\n")

    try:
        example_compare_cases()
    except FileNotFoundError as e:
        print(f"Error: {e}\n")

    try:
        example_access_matrices()
    except FileNotFoundError as e:
        print(f"Error: {e}\n")

    print("#"*80)
    print("# Examples complete")
    print("#"*80 + "\n")
