"""
Test parameter dependence for M=5 to understand why heatmap is single color.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'IBD_analysis', 'src'))

from IBD_analysis.src.concentric_analysis import analyze_concentric_for_parameters


def test_extreme_parameters():
    """Test extreme parameter values to find boundaries."""
    M = 5
    case_name = 'case4'

    print("=" * 60)
    print("Testing extreme parameters for M=5 case4")
    print("=" * 60)

    # Test very small and very large values
    test_cases = [
        # (N, m, alpha, description)
        (32, 0.001, 0.00003, "Very small m and alpha"),
        (32, 0.5, 0.00003, "Large m, small alpha"),
        (32, 0.001, 0.01, "Small m, large alpha"),
        (32, 0.5, 0.01, "Large m and alpha"),
        (1024, 0.001, 0.00003, "Large N, small m and alpha"),
        (1024, 0.5, 0.01, "All large"),
    ]

    for N, m, alpha, desc in test_cases:
        result = analyze_concentric_for_parameters(
            N, m, alpha, M,
            center_prestige=True,
            centralized_neologism_creation=True,
            distance_method='nei',
            use_symbolic=False,
            verbose=False
        )

        status = "✓ CONCENTRIC" if result['is_concentric'] else "✗ NOT CONCENTRIC"
        print(f"{status}: {desc}")
        print(f"  N={N}, m={m:.6f}, α={alpha:.8f}")
        print(f"  Distance[0,1]={result['distance_matrix'][0,1]:.6f}, Distance[0,2]={result['distance_matrix'][0,2]:.6f}")
        print()


def test_gradient():
    """Test gradual change in parameters to find transition."""
    M = 5
    N = 64

    print("=" * 60)
    print("Testing gradient: varying m with fixed N, alpha")
    print("=" * 60)

    alpha = 0.001
    m_values = [2**p for p in range(-15, 1)]  # 2^-15 to 2^0

    concentric_results = []
    for m in m_values:
        result = analyze_concentric_for_parameters(
            N, m, alpha, M,
            center_prestige=True,
            centralized_neologism_creation=True,
            distance_method='nei',
            use_symbolic=False,
            verbose=False
        )
        concentric_results.append(result['is_concentric'])
        status = "✓" if result['is_concentric'] else "✗"
        print(f"{status} m=2^{int(np.log2(m))}={m:.8f}: {result['is_concentric']}")

    concentric_rate = sum(concentric_results) / len(concentric_results)
    print(f"\nConcentric rate: {sum(concentric_results)}/{len(concentric_results)} = {100*concentric_rate:.1f}%")

    print("\n" + "=" * 60)
    print("Testing gradient: varying alpha with fixed N, m")
    print("=" * 60)

    m = 0.1
    alpha_values = [2**p for p in range(-20, -4)]  # 2^-20 to 2^-5

    concentric_results = []
    for alpha in alpha_values:
        result = analyze_concentric_for_parameters(
            N, m, alpha, M,
            center_prestige=True,
            centralized_neologism_creation=True,
            distance_method='nei',
            use_symbolic=False,
            verbose=False
        )
        concentric_results.append(result['is_concentric'])
        status = "✓" if result['is_concentric'] else "✗"
        print(f"{status} α=2^{int(np.log2(alpha))}={alpha:.10f}: {result['is_concentric']}")

    concentric_rate = sum(concentric_results) / len(concentric_results)
    print(f"\nConcentric rate: {sum(concentric_results)}/{len(concentric_results)} = {100*concentric_rate:.1f}%")


def test_all_cases_comparison():
    """Compare all cases to see which shows parameter dependence."""
    M = 5
    N = 64
    m = 0.1
    alpha = 0.001

    print("\n" + "=" * 60)
    print(f"Comparing all cases (N={N}, m={m}, α={alpha})")
    print("=" * 60)

    cases = [
        ('case1', False, False),
        ('case2', True, False),
        ('case3', False, True),
        ('case4', True, True),
    ]

    for case_name, center_prestige, centralized_neologism in cases:
        result = analyze_concentric_for_parameters(
            N, m, alpha, M,
            center_prestige=center_prestige,
            centralized_neologism_creation=centralized_neologism,
            distance_method='nei',
            use_symbolic=False,
            verbose=False
        )

        status = "✓ CONCENTRIC" if result['is_concentric'] else "✗ NOT CONCENTRIC"
        print(f"\n{case_name}: {status}")
        print(f"  Distance matrix (row 0):")
        print(f"    {result['distance_matrix'][0]}")


if __name__ == "__main__":
    test_extreme_parameters()
    test_gradient()
    test_all_cases_comparison()
