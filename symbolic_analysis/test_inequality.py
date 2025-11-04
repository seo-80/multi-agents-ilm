"""
Test script for distance inequality analysis.

This script analyzes inequalities E[d_ij] > E[d_kl] for different cases.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from solve_distance_inequality import analyze_distance_inequality


def test_case1_d12_vs_d13():
    """
    Test Case 1: Analyze E[d_12] > E[d_13]

    In the symmetric bidirectional model with uniform innovation,
    we compare the expected distance between agents 1-2 (neighbors)
    vs agents 1-3 (non-neighbors).
    """
    print("\n" + "#"*80)
    print("# Case 1: E[d_12] > E[d_13]")
    print("# (neighbor distance vs non-neighbor distance)")
    print("#"*80 + "\n")

    result = analyze_distance_inequality(
        M=3,
        case_name="case1",
        pair1=(1, 2),  # neighbors
        pair2=(1, 3)   # non-neighbors
    )

    return result


def test_case2_d12_vs_d13():
    """
    Test Case 2: Analyze E[d_12] > E[d_13]

    In the center-prestige model (center-outward) with uniform innovation,
    agent 2 is the center. We compare distances:
    - (1,2): edge to center
    - (1,3): edge to edge
    """
    print("\n" + "#"*80)
    print("# Case 2: E[d_12] > E[d_13]")
    print("# (edge-center distance vs edge-edge distance)")
    print("#"*80 + "\n")

    result = analyze_distance_inequality(
        M=3,
        case_name="case2",
        pair1=(1, 2),
        pair2=(1, 3)
    )

    return result


def test_case3_d12_vs_d13():
    """
    Test Case 3: Analyze E[d_12] > E[d_13]

    In the symmetric model with center-only innovation,
    only agent 2 (center) creates innovations.
    """
    print("\n" + "#"*80)
    print("# Case 3: E[d_12] > E[d_13]")
    print("# (symmetric network, center-only innovation)")
    print("#"*80 + "\n")

    result = analyze_distance_inequality(
        M=3,
        case_name="case3",
        pair1=(1, 2),
        pair2=(1, 3)
    )

    return result


def test_case4_d12_vs_d13():
    """
    Test Case 4: Analyze E[d_12] > E[d_13]

    In the center-prestige model with center-only innovation,
    agent 2 is both the prestige center and the sole innovator.
    """
    print("\n" + "#"*80)
    print("# Case 4: E[d_12] > E[d_13]")
    print("# (center-prestige + center-only innovation)")
    print("#"*80 + "\n")

    result = analyze_distance_inequality(
        M=3,
        case_name="case4",
        pair1=(1, 2),
        pair2=(1, 3)
    )

    return result


def test_all_cases():
    """
    Test all 4 cases for the inequality E[d_12] > E[d_13].
    """
    print("\n" + "#"*80)
    print("# Distance Inequality Analysis: All Cases")
    print("# Analyzing: E[d_12] > E[d_13]")
    print("#"*80)

    results = {}

    test_functions = [
        ("case1", test_case1_d12_vs_d13),
        ("case2", test_case2_d12_vs_d13),
        ("case3", test_case3_d12_vs_d13),
        ("case4", test_case4_d12_vs_d13),
    ]

    for case, test_func in test_functions:
        try:
            result = test_func()
            results[case] = result
            print(f"\n✓ {case.upper()} analysis complete")
        except FileNotFoundError:
            print(f"\n✗ {case.upper()} failed: Results file not found")
            print(f"  Please run: python test_m_agent_stationary.py")
        except Exception as e:
            print(f"\n✗ {case.upper()} failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "#"*80)
    print("# SUMMARY")
    print("#"*80)
    print(f"\nSuccessfully analyzed: {len(results)}/4 cases")

    if results:
        print("\nOutput files generated:")
        for case in results.keys():
            print(f"  ✓ symbolic_analysis/results/M3_{case}_inequality_d12_vs_d13.md")
            print(f"  ✓ symbolic_analysis/results/M3_{case}_inequality_d12_vs_d13.png")

    print("\n" + "#"*80)

    return results


def test_custom_inequality(case: str, i: int, j: int, k: int, l: int):
    """
    Test custom inequality E[d_ij] > E[d_kl].

    Args:
        case: Case name (case1, case2, case3, or case4)
        i, j: First agent pair
        k, l: Second agent pair
    """
    print("\n" + "#"*80)
    print(f"# Custom Inequality: E[d_{{{i},{j}}}] > E[d_{{{k},{l}}}]")
    print(f"# Case: {case.upper()}")
    print("#"*80 + "\n")

    # Ensure pairs are in canonical form
    pair1 = (min(i, j), max(i, j))
    pair2 = (min(k, l), max(k, l))

    result = analyze_distance_inequality(
        M=3,
        case_name=case,
        pair1=pair1,
        pair2=pair2
    )

    return result


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Default: test all cases for d_12 > d_13
        test_all_cases()

    elif len(sys.argv) == 2:
        # Single case for d_12 > d_13
        case = sys.argv[1]
        if case == "case1":
            test_case1_d12_vs_d13()
        elif case == "case2":
            test_case2_d12_vs_d13()
        elif case == "case3":
            test_case3_d12_vs_d13()
        elif case == "case4":
            test_case4_d12_vs_d13()
        elif case == "all":
            test_all_cases()
        else:
            print(f"Error: Unknown case '{case}'")
            print("Valid cases: case1, case2, case3, case4, all")
            sys.exit(1)

    elif len(sys.argv) == 6:
        # Custom inequality: case i j k l
        case = sys.argv[1]
        i, j, k, l = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
        test_custom_inequality(case, i, j, k, l)

    else:
        print("Usage:")
        print("  python test_inequality.py                  # Test all cases for E[d_12] > E[d_13]")
        print("  python test_inequality.py <case>           # Test single case for E[d_12] > E[d_13]")
        print("  python test_inequality.py all              # Test all cases")
        print("  python test_inequality.py <case> i j k l   # Custom inequality E[d_ij] > E[d_kl]")
        print("\nExamples:")
        print("  python test_inequality.py case1")
        print("  python test_inequality.py case1 1 2 1 3")
        print("  python test_inequality.py case2 1 2 2 3")
        sys.exit(1)
