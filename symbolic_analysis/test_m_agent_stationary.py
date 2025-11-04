"""
Test script for M-Agent Stationary State Symbolic Computation

This script tests all 4 model cases:
    Case 1: center_prestige=False, centralized_neologism_creation=False
    Case 2: center_prestige=True,  centralized_neologism_creation=False
    Case 3: center_prestige=False, centralized_neologism_creation=True
    Case 4: center_prestige=True,  centralized_neologism_creation=True

For each case, it computes the stationary state symbolically for M=3 agents.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from m_agent_stationary_symbolic import compute_stationary_state_symbolic


def test_case1():
    """
    Test Case 1: Symmetric bidirectional + Uniform innovation

    Model characteristics:
    - All agents interact symmetrically with neighbors
    - All agents create innovations equally (α_i = α for all i)
    - Most democratic/egalitarian model
    """
    print("\n" + "="*80)
    print("TESTING CASE 1")
    print("="*80)

    states, pi, P, W = compute_stationary_state_symbolic(
        M=3,
        center_prestige=False,
        centralized_neologism_creation=False
    )

    return states, pi, P, W


def test_case2():
    """
    Test Case 2: Center-outward asymmetric + Uniform innovation

    Model characteristics:
    - Information flows from center to periphery (asymmetric)
    - All agents create innovations equally (α_i = α for all i)
    - Center is conservative (only self-refers), periphery follows center
    """
    print("\n" + "="*80)
    print("TESTING CASE 2")
    print("="*80)

    states, pi, P, W = compute_stationary_state_symbolic(
        M=3,
        center_prestige=True,
        centralized_neologism_creation=False
    )

    return states, pi, P, W


def test_case3():
    """
    Test Case 3: Symmetric bidirectional + Center-only innovation

    Model characteristics:
    - All agents interact symmetrically with neighbors
    - Only center creates innovations (α_c = α, α_i = 0 for i ≠ c)
    - Innovations diffuse symmetrically from center
    """
    print("\n" + "="*80)
    print("TESTING CASE 3")
    print("="*80)

    states, pi, P, W = compute_stationary_state_symbolic(
        M=3,
        center_prestige=False,
        centralized_neologism_creation=True
    )

    return states, pi, P, W


def test_case4():
    """
    Test Case 4: Center-outward asymmetric + Center-only innovation

    Model characteristics:
    - Information flows from center to periphery (asymmetric)
    - Only center creates innovations (α_c = α, α_i = 0 for i ≠ c)
    - Most centralized model: center is the sole cultural/linguistic source
    """
    print("\n" + "="*80)
    print("TESTING CASE 4")
    print("="*80)

    states, pi, P, W = compute_stationary_state_symbolic(
        M=3,
        center_prestige=True,
        centralized_neologism_creation=True
    )

    return states, pi, P, W


def test_all_cases():
    """
    Run all 4 test cases sequentially.
    """
    print("\n" + "#"*80)
    print("# M-Agent Stationary State: Testing All 4 Cases")
    print("#"*80)

    results = {}

    try:
        print("\n[1/4] Running Case 1...")
        results['case1'] = test_case1()
    except Exception as e:
        print(f"ERROR in Case 1: {e}")
        import traceback
        traceback.print_exc()

    try:
        print("\n[2/4] Running Case 2...")
        results['case2'] = test_case2()
    except Exception as e:
        print(f"ERROR in Case 2: {e}")
        import traceback
        traceback.print_exc()

    try:
        print("\n[3/4] Running Case 3...")
        results['case3'] = test_case3()
    except Exception as e:
        print(f"ERROR in Case 3: {e}")
        import traceback
        traceback.print_exc()

    try:
        print("\n[4/4] Running Case 4...")
        results['case4'] = test_case4()
    except Exception as e:
        print(f"ERROR in Case 4: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "#"*80)
    print("# TEST SUMMARY")
    print("#"*80)
    print(f"\nSuccessfully completed: {len(results)}/4 cases")
    print(f"Cases completed: {', '.join(results.keys())}")
    print("\nOutput files generated in: results/")
    for case in ['case1', 'case2', 'case3', 'case4']:
        if case in results:
            print(f"  ✓ M3_{case}.md")
        else:
            print(f"  ✗ M3_{case}.md (failed)")
    print()

    return results


def test_single_case(case_number: int):
    """
    Test a single case by number (1-4).

    Args:
        case_number: Case number (1, 2, 3, or 4)
    """
    if case_number == 1:
        return test_case1()
    elif case_number == 2:
        return test_case2()
    elif case_number == 3:
        return test_case3()
    elif case_number == 4:
        return test_case4()
    else:
        raise ValueError(f"Invalid case number: {case_number}. Must be 1-4.")


def test_custom(M: int, center_prestige: bool, centralized_neologism_creation: bool):
    """
    Test with custom parameters.

    Args:
        M: Number of agents (must be odd)
        center_prestige: Center prestige condition
        centralized_neologism_creation: Centralized neologism creation condition
    """
    print("\n" + "="*80)
    print("CUSTOM TEST")
    print("="*80)
    print(f"M={M}, center_prestige={center_prestige}, "
          f"centralized_neologism_creation={centralized_neologism_creation}")

    states, pi, P, W = compute_stationary_state_symbolic(
        M=M,
        center_prestige=center_prestige,
        centralized_neologism_creation=centralized_neologism_creation
    )

    return states, pi, P, W


if __name__ == "__main__":
    # Default: run all cases
    if len(sys.argv) == 1:
        test_all_cases()

    # Run specific case
    elif len(sys.argv) == 2:
        try:
            case_num = int(sys.argv[1])
            if 1 <= case_num <= 4:
                test_single_case(case_num)
            else:
                print(f"Error: Case number must be 1-4, got {case_num}")
                sys.exit(1)
        except ValueError:
            print(f"Error: Invalid case number '{sys.argv[1]}'. Must be 1-4.")
            sys.exit(1)

    # Custom parameters
    elif len(sys.argv) == 4:
        try:
            M = int(sys.argv[1])
            center_prestige = sys.argv[2].lower() in ['true', '1', 'yes']
            centralized = sys.argv[3].lower() in ['true', '1', 'yes']
            test_custom(M, center_prestige, centralized)
        except Exception as e:
            print(f"Error: {e}")
            print("\nUsage:")
            print("  python test_m_agent_stationary.py              # Run all 4 cases")
            print("  python test_m_agent_stationary.py <case_num>   # Run specific case (1-4)")
            print("  python test_m_agent_stationary.py <M> <cp> <cn>  # Custom parameters")
            print("\nWhere:")
            print("  <M>  = Number of agents (odd)")
            print("  <cp> = center_prestige (true/false)")
            print("  <cn> = centralized_neologism_creation (true/false)")
            sys.exit(1)

    else:
        print("Usage:")
        print("  python test_m_agent_stationary.py              # Run all 4 cases")
        print("  python test_m_agent_stationary.py <case_num>   # Run specific case (1-4)")
        print("  python test_m_agent_stationary.py <M> <cp> <cn>  # Custom parameters")
        print("\nWhere:")
        print("  <M>  = Number of agents (odd)")
        print("  <cp> = center_prestige (true/false)")
        print("  <cn> = centralized_neologism_creation (true/false)")
        sys.exit(1)
