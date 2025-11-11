"""
Test to identify discrepancy between naive_simulation and IBD_analysis.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'IBD_analysis', 'src'))

from IBD_analysis.src.concentric_analysis import analyze_concentric_for_parameters
from IBD_analysis.src.distance_metrics import compute_f_distance


def test_single_case():
    """Test a single parameter case that should show concentric distribution."""
    # Parameters that should show concentric in naive_simulation
    N = 64
    m = 0.0625  # 2^-4
    alpha = 0.0009765625  # 2^-10
    M = 5

    print("=" * 60)
    print("Testing parameter set:")
    print(f"  N = {N}")
    print(f"  m = {m}")
    print(f"  α = {alpha}")
    print(f"  M = {M}")
    print("=" * 60)

    cases = [
        ('case1', False, False),
        ('case2', True, False),
        ('case3', False, True),
        ('case4', True, True),
    ]

    for case_name, center_prestige, centralized_neologism in cases:
        print(f"\n{case_name}: center_prestige={center_prestige}, centralized_neologism={centralized_neologism}")

        # Test with symbolic (if available)
        try:
            result_sym = analyze_concentric_for_parameters(
                N, m, alpha, M,
                center_prestige=center_prestige,
                centralized_neologism_creation=centralized_neologism,
                distance_method='nei',
                use_symbolic=True,
                verbose=False
            )
            print(f"  Symbolic: is_concentric = {result_sym['is_concentric']}")
            print(f"    F diagonal: {np.diag(result_sym['F_matrix'])}")
            print(f"    F[0,1]={result_sym['F_matrix'][0,1]:.6f}, F[0,2]={result_sym['F_matrix'][0,2]:.6f}")
            print(f"    Distance[0,1]={result_sym['distance_matrix'][0,1]:.6f}, Distance[0,2]={result_sym['distance_matrix'][0,2]:.6f}")
        except FileNotFoundError as e:
            print(f"  Symbolic: Not available ({e})")

        # Test with numerical
        result_num = analyze_concentric_for_parameters(
            N, m, alpha, M,
            center_prestige=center_prestige,
            centralized_neologism_creation=centralized_neologism,
            distance_method='nei',
            use_symbolic=False,
            verbose=False
        )
        print(f"  Numerical: is_concentric = {result_num['is_concentric']}")
        print(f"    F diagonal: {np.diag(result_num['F_matrix'])}")
        print(f"    F[0,1]={result_num['F_matrix'][0,1]:.6f}, F[0,2]={result_num['F_matrix'][0,2]:.6f}")
        print(f"    Distance[0,1]={result_num['distance_matrix'][0,1]:.6f}, Distance[0,2]={result_num['distance_matrix'][0,2]:.6f}")


def test_parameter_range():
    """Test multiple parameters to see if any show concentric."""
    M = 5
    case_name = 'case4'

    N_values = [32, 64, 128]
    m_values = [0.0625, 0.125, 0.25]  # 2^-4, 2^-3, 2^-2
    alpha_values = [0.00048828125, 0.0009765625, 0.001953125]  # 2^-11, 2^-10, 2^-9

    print("\n" + "=" * 60)
    print(f"Testing parameter range for {case_name} (M={M})")
    print("=" * 60)

    concentric_count = 0
    total_count = 0

    for N in N_values:
        for m in m_values:
            for alpha in alpha_values:
                result = analyze_concentric_for_parameters(
                    N, m, alpha, M,
                    center_prestige=True,
                    centralized_neologism_creation=True,
                    distance_method='nei',
                    use_symbolic=False,
                    verbose=False
                )

                total_count += 1
                if result['is_concentric']:
                    concentric_count += 1
                    print(f"✓ CONCENTRIC: N={N}, m={m:.6f}, α={alpha:.9f}")

    print(f"\nConcentric rate: {concentric_count}/{total_count} = {100*concentric_count/total_count:.1f}%")


if __name__ == "__main__":
    test_single_case()
    test_parameter_range()
