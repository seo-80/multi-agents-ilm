"""
Factorize the difference E[d_12] - E[d_13] for all cases.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sympy import symbols, simplify, factor, latex
from m_agent_stationary_symbolic import load_results_by_case
from analyze_distances import compute_distance_expectations


def factorize_distance_difference(M: int, case_name: str):
    """
    Factorize the difference E[d_12] - E[d_13].

    Args:
        M: Number of agents
        case_name: Case identifier
    """
    print("=" * 80)
    print(f"Factorizing Distance Difference: M={M}, {case_name.upper()}")
    print("=" * 80)

    # Load saved results
    print("\nLoading saved results...")
    results = load_results_by_case(M, case_name)
    states = results['states']
    pi = results['pi']

    # Compute expected distances
    print("Computing expected distances...")
    expected_distances = compute_distance_expectations(states, pi, M)

    # Get expressions for the two pairs
    expr1 = expected_distances[(1, 2)]  # E[d_12]
    expr2 = expected_distances[(1, 3)]  # E[d_13]

    # Compute difference
    print("Computing difference...")
    diff = simplify(expr1 - expr2)

    print("\nOriginal difference (simplified):")
    print(f"E[d_{{1,2}}] - E[d_{{1,3}}] = {diff}")
    print()

    # Factorize
    print("Factorizing...")
    factored = factor(diff)

    print("\n" + "=" * 80)
    print("FACTORIZED FORM")
    print("=" * 80)
    print()
    print(f"E[d_{{1,2}}] - E[d_{{1,3}}] = {factored}")
    print()

    # LaTeX output
    print("=" * 80)
    print("LATEX")
    print("=" * 80)
    print()
    print(f"$$E[d_{{1,2}}] - E[d_{{1,3}}] = {latex(factored)}$$")
    print()

    # Extract numerator and denominator
    print("=" * 80)
    print("NUMERATOR AND DENOMINATOR")
    print("=" * 80)
    print()

    numer, denom = factored.as_numer_denom()

    print("Numerator (factored):")
    print(f"  {numer}")
    print()

    print("Denominator (factored):")
    print(f"  {denom}")
    print()

    print("LaTeX Numerator:")
    print(f"  $${latex(numer)}$$")
    print()

    print("LaTeX Denominator:")
    print(f"  $${latex(denom)}$$")
    print()

    return {
        'diff_original': diff,
        'diff_factored': factored,
        'numerator': numer,
        'denominator': denom
    }


def factorize_all_cases(M: int = 3):
    """
    Factorize for all 4 cases.

    Args:
        M: Number of agents
    """
    print("\n" + "#" * 80)
    print(f"# Factorization Analysis for All Cases (M={M})")
    print("#" * 80)

    cases = ["case1", "case2", "case3", "case4"]
    results = {}

    for case in cases:
        try:
            result = factorize_distance_difference(M, case)
            results[case] = result
            print(f"\n✓ {case.upper()} factorization complete\n")
        except Exception as e:
            print(f"\n✗ {case.upper()} failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "#" * 80)
    print("# SUMMARY")
    print("#" * 80)
    print(f"\nSuccessfully factorized: {len(results)}/4 cases")
    print("\n" + "#" * 80)

    return results


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Default: all cases
        factorize_all_cases(M=3)
    elif len(sys.argv) == 2:
        # Single case
        case = sys.argv[1]
        if case == "all":
            factorize_all_cases(M=3)
        elif case in ["case1", "case2", "case3", "case4"]:
            factorize_distance_difference(M=3, case_name=case)
        else:
            print(f"Error: Unknown case '{case}'")
            print("Valid cases: case1, case2, case3, case4, all")
            sys.exit(1)
    else:
        print("Usage:")
        print("  python factorize_difference.py           # All cases")
        print("  python factorize_difference.py <case>    # Single case")
        print("  python factorize_difference.py all       # All cases")
        sys.exit(1)
