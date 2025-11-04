"""
Test script for distance analysis.

This script analyzes expected distances between agents for all 4 cases.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analyze_distances import analyze_distances_from_results


def test_all_cases(M: int = 3):
    """
    Analyze distances for all 4 cases.

    Args:
        M: Number of agents (default: 3)
    """
    print("\n" + "#"*80)
    print(f"# Agent Distance Analysis for All Cases (M={M})")
    print("#"*80)

    cases = ["case1", "case2", "case3", "case4"]
    results = {}

    for case in cases:
        print(f"\n[{case.upper()}] Analyzing distances...")
        try:
            result = analyze_distances_from_results(M, case)
            results[case] = result
            print(f"✓ {case.upper()} analysis complete")
        except FileNotFoundError as e:
            print(f"✗ {case.upper()} failed: Results file not found")
            print(f"  Please run: python test_m_agent_stationary.py {case[-1]}")
        except Exception as e:
            print(f"✗ {case.upper()} failed: {e}")
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
            print(f"  ✓ symbolic_analysis/results/M{M}_{case}_distances.md")

        # Show symbolic results
        print("\n" + "-"*80)
        print("Symbolic Expressions Generated:")
        print("-"*80)

        print("\nAll cases have symbolic expressions for:")
        print("  - Pairwise expected distances E[d_ij]")
        print("  - Average expected distance")
        print("\nSee results/*_distances.md for full symbolic expressions.")

    else:
        print("\nNo cases analyzed. Please compute stationary states first:")
        print("  python test_m_agent_stationary.py")

    print("\n" + "#"*80)


def test_single_case(M: int, case_name: str):
    """
    Analyze distances for a single case.

    Args:
        M: Number of agents
        case_name: Case identifier ("case1", "case2", "case3", or "case4")
    """
    print("\n" + "#"*80)
    print(f"# Agent Distance Analysis: M={M}, {case_name.upper()}")
    print("#"*80 + "\n")

    try:
        result = analyze_distances_from_results(M, case_name)
        print(f"\n✓ Analysis complete!")
        print(f"  Output: symbolic_analysis/results/M{M}_{case_name}_distances.md")

        # Show symbolic expressions
        print("\n" + "-"*80)
        print("Symbolic Expressions")
        print("-"*80)

        expected_distances = result['expected_distances']
        print("\nPairwise expected distances (symbolic):")
        for (i, j) in sorted(expected_distances.keys()):
            print(f"  E[d_{{{i},{j}}}] = (expression in m and α)")

        print(f"\nAverage expected distance: (expression in m and α)")
        print(f"\nSee {case_name}_distances.md for full symbolic expressions.")

    except FileNotFoundError:
        print(f"\n✗ Error: Results file not found for M={M}, {case_name}")
        print(f"  Please run: python test_m_agent_stationary.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) == 1:
        # Default: analyze all cases for M=3
        test_all_cases(M=3)

    elif len(sys.argv) == 2:
        # Single case for M=3
        case_name = sys.argv[1]
        if case_name in ["case1", "case2", "case3", "case4"]:
            test_single_case(M=3, case_name=case_name)
        elif sys.argv[1] == "all":
            test_all_cases(M=3)
        else:
            print(f"Error: Invalid case '{case_name}'")
            print("Valid cases: case1, case2, case3, case4, all")
            sys.exit(1)

    elif len(sys.argv) == 3:
        # Custom M and case
        try:
            M = int(sys.argv[1])
            case_name = sys.argv[2]

            if case_name == "all":
                test_all_cases(M=M)
            elif case_name in ["case1", "case2", "case3", "case4"]:
                test_single_case(M=M, case_name=case_name)
            else:
                print(f"Error: Invalid case '{case_name}'")
                print("Valid cases: case1, case2, case3, case4, all")
                sys.exit(1)
        except ValueError:
            print("Error: First argument must be an integer (M)")
            sys.exit(1)

    else:
        print("Usage:")
        print("  python test_distance_analysis.py              # All cases, M=3")
        print("  python test_distance_analysis.py <case>       # Single case, M=3")
        print("  python test_distance_analysis.py all          # All cases, M=3")
        print("  python test_distance_analysis.py <M> <case>   # Custom M and case")
        print("  python test_distance_analysis.py <M> all      # All cases, custom M")
        print("\nExamples:")
        print("  python test_distance_analysis.py case1")
        print("  python test_distance_analysis.py 3 all")
        print("  python test_distance_analysis.py 5 case2")
        sys.exit(1)
