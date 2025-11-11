"""
Test concentric detection with M=3 (default).
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'IBD_analysis', 'src'))

from IBD_analysis.src.concentric_analysis import analyze_concentric_for_parameters


def test_m3_parameter_range():
    """Test if M=3 shows concentric distribution."""
    M = 3

    # Test parameters similar to our M=5 test
    N_values = [32, 64, 128]
    m_values = [0.0625, 0.125, 0.25]  # 2^-4, 2^-3, 2^-2
    alpha_values = [0.00048828125, 0.0009765625, 0.001953125]  # 2^-11, 2^-10, 2^-9

    print("=" * 60)
    print(f"Testing M=3 parameter range")
    print("=" * 60)

    for case_name, center_prestige, centralized_neologism in [
        ('case1', False, False),
        ('case2', True, False),
        ('case3', False, True),
        ('case4', True, True),
    ]:
        print(f"\n{case_name}:")
        concentric_count = 0
        total_count = 0

        for N in N_values:
            for m in m_values:
                for alpha in alpha_values:
                    result = analyze_concentric_for_parameters(
                        N, m, alpha, M,
                        center_prestige=center_prestige,
                        centralized_neologism_creation=centralized_neologism,
                        distance_method='nei',
                        use_symbolic=False,
                        verbose=False
                    )

                    total_count += 1
                    if result['is_concentric']:
                        concentric_count += 1

        print(f"  Concentric rate: {concentric_count}/{total_count} = {100*concentric_count/total_count:.1f}%")


def test_wide_parameter_range():
    """Test wider parameter range for M=3 case4."""
    M = 3
    case_name = 'case4'

    # Wider range
    N_values = [2**p for p in range(5, 11)]  # 32, 64, 128, 256, 512, 1024
    m_values = [2**p for p in range(-10, 0)]  # 2^-10 to 2^-1
    alpha_values = [2**p for p in range(-15, -5)]  # 2^-15 to 2^-6

    print("\n" + "=" * 60)
    print(f"Testing wide parameter range for M=3 {case_name}")
    print(f"  N: {len(N_values)} values")
    print(f"  m: {len(m_values)} values")
    print(f"  α: {len(alpha_values)} values")
    print(f"  Total combinations: {len(N_values)*len(m_values)*len(alpha_values)}")
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
                    if concentric_count <= 10:  # Print first 10
                        print(f"✓ CONCENTRIC: N={N}, m={m:.6f}, α={alpha:.9f}")

    print(f"\nTotal concentric rate: {concentric_count}/{total_count} = {100*concentric_count/total_count:.1f}%")


if __name__ == "__main__":
    test_m3_parameter_range()
    test_wide_parameter_range()
