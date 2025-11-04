"""
Analyze expected distances between agents based on stationary distribution.

This module computes the expected distance between each pair of agents,
where distance is defined as:
    d_ij = 0 if agents i and j share the same data (same block)
    d_ij = 1 if agents i and j have different data (different blocks)

The expected distance is:
    E[d_ij] = Σ_s π_s × d_ij(s)

where π_s is the stationary probability of state s, and d_ij(s) is the
distance between agents i and j in state s.
"""

import sympy
from sympy import symbols, simplify, latex, Float
from typing import List, FrozenSet, Dict, Tuple
import numpy as np
import os

from m_agent_stationary_symbolic import (
    load_results_by_case, partition_to_string
)


def agents_in_same_block(agent_i: int, agent_j: int,
                         state: FrozenSet[FrozenSet[int]]) -> bool:
    """
    Check if two agents are in the same block in a given state.

    Args:
        agent_i: First agent ID (1-indexed)
        agent_j: Second agent ID (1-indexed)
        state: State as frozenset of frozensets (partition)

    Returns:
        True if agents are in the same block, False otherwise
    """
    for block in state:
        if agent_i in block and agent_j in block:
            return True
    return False


def compute_distance_expectations(states: List[FrozenSet[FrozenSet[int]]],
                                  pi: sympy.Matrix,
                                  M: int) -> Dict[Tuple[int, int], sympy.Expr]:
    """
    Compute expected distance between all pairs of agents.

    Args:
        states: List of all states (partitions)
        pi: Stationary distribution (symbolic)
        M: Number of agents

    Returns:
        Dictionary mapping (i, j) to expected distance E[d_ij]
        Only includes pairs where i < j (due to symmetry)
    """
    print("Computing expected distances between agents...")

    # Initialize expected distances
    expected_distances = {}

    # Compute for all pairs i < j
    for i in range(1, M + 1):
        for j in range(i + 1, M + 1):
            # E[d_ij] = Σ_s π_s × d_ij(s)
            # where d_ij(s) = 0 if same block, 1 if different blocks
            expected_dist = sympy.Integer(0)

            for state_idx, state in enumerate(states):
                # Check if i and j are in same block
                if agents_in_same_block(i, j, state):
                    # Distance is 0, doesn't contribute
                    pass
                else:
                    # Distance is 1, contributes π_s
                    expected_dist += pi[state_idx]

            # Simplify the expression
            expected_dist = simplify(expected_dist)
            expected_distances[(i, j)] = expected_dist

            print(f"  E[d_{{{i},{j}}}] computed")

    print("Expected distances computed.")
    return expected_distances


def compute_average_distance(expected_distances: Dict[Tuple[int, int], sympy.Expr],
                            M: int) -> sympy.Expr:
    """
    Compute the average expected distance across all agent pairs.

    Args:
        expected_distances: Dictionary of pairwise expected distances
        M: Number of agents

    Returns:
        Average expected distance (symbolic expression)
    """
    total = sum(expected_distances.values())
    num_pairs = M * (M - 1) // 2
    avg_distance = simplify(total / num_pairs)
    return avg_distance


def evaluate_distances_numerically(expected_distances: Dict[Tuple[int, int], sympy.Expr],
                                   m_val: float, alpha_val: float) -> Dict[Tuple[int, int], float]:
    """
    Evaluate expected distances numerically for given parameter values.

    Args:
        expected_distances: Dictionary of symbolic expected distances
        m_val: Value for m parameter
        alpha_val: Value for alpha parameter

    Returns:
        Dictionary mapping (i, j) to numerical expected distance
    """
    m, alpha = symbols('m alpha')

    numerical_distances = {}
    for (i, j), expr in expected_distances.items():
        val = expr.subs({m: m_val, alpha: alpha_val})
        numerical_distances[(i, j)] = float(val)

    return numerical_distances


def write_distance_analysis_to_md(M: int, case_name: str,
                                  expected_distances: Dict[Tuple[int, int], sympy.Expr],
                                  avg_distance: sympy.Expr,
                                  output_dir: str = None):
    """
    Write distance analysis results to a markdown file.

    Args:
        M: Number of agents
        case_name: Case identifier
        expected_distances: Dictionary of pairwise expected distances
        avg_distance: Average expected distance
        output_dir: Output directory
    """
    # Set default output directory
    if output_dir is None:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(script_dir, "results")

    filename = f"M{M}_{case_name}_distances.md"
    filepath = os.path.join(output_dir, filename)

    os.makedirs(output_dir, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        # Header
        f.write(f"# Agent Distance Analysis: M={M}, {case_name.upper()}\n\n")

        # Introduction
        f.write("## Expected Distances Between Agents\n\n")
        f.write("Distance is defined as:\n")
        f.write("- $d_{ij} = 0$ if agents $i$ and $j$ share the same data (same block)\n")
        f.write("- $d_{ij} = 1$ if agents $i$ and $j$ have different data (different blocks)\n\n")
        f.write("The expected distance is computed using the stationary distribution:\n\n")
        f.write("$$E[d_{ij}] = \\sum_s \\pi_s \\times d_{ij}(s)$$\n\n")

        # Pairwise expected distances
        f.write("## Pairwise Expected Distances\n\n")

        # Create a table
        f.write("| Agent Pair $(i,j)$ | Expected Distance $E[d_{ij}]$ |\n")
        f.write("|-------------------|-------------------------------|\n")

        for (i, j) in sorted(expected_distances.keys()):
            expr = expected_distances[(i, j)]
            f.write(f"| $({i},{j})$ | ${latex(expr)}$ |\n")

        f.write("\n")

        # Average distance
        f.write("## Average Expected Distance\n\n")
        f.write("Average over all agent pairs:\n\n")
        f.write(f"$$\\bar{{d}} = {latex(avg_distance)}$$\n\n")

        # Numerical examples
        f.write("## Numerical Evaluation\n\n")
        f.write("To evaluate numerically, substitute values for $m$ and $\\alpha$ into the expressions above.\n\n")
        f.write("For example, using Python with SymPy:\n\n")
        f.write("```python\n")
        f.write("from sympy import symbols\n")
        f.write("m, alpha = symbols('m alpha')\n")
        f.write("# Load results\n")
        f.write("results = load_results_by_case(3, 'case1')\n")
        f.write("# Substitute values\n")
        f.write("m_val, alpha_val = 0.3, 0.1\n")
        f.write("for (i, j), expr in expected_distances.items():\n")
        f.write("    val = float(expr.subs({m: m_val, alpha: alpha_val}).n())\n")
        f.write("    print(f'E[d_{{{i},{j}}}] = {val:.6f}')\n")
        f.write("```\n\n")

        # Interpretation
        f.write("## Interpretation\n\n")
        f.write("- **$E[d_{ij}] = 0$**: Agents $i$ and $j$ always share the same data\n")
        f.write("- **$E[d_{ij}] = 1$**: Agents $i$ and $j$ always have different data\n")
        f.write("- **$0 < E[d_{ij}] < 1$**: Agents sometimes share, sometimes differ\n\n")
        f.write("The average expected distance $\\bar{d}$ measures overall diversity:\n")
        f.write("- **$\\bar{d} \\approx 0$**: High synchronization (agents tend to share data)\n")
        f.write("- **$\\bar{d} \\approx 1$**: High diversity (agents tend to have different data)\n\n")

    print(f"Distance analysis written to: {filepath}")


def analyze_distances_from_results(M: int, case_name: str,
                                   output_dir: str = None) -> Dict:
    """
    Main function to analyze distances from saved results.

    Args:
        M: Number of agents
        case_name: Case identifier ("case1", "case2", "case3", or "case4")
        output_dir: Output directory for results

    Returns:
        Dictionary containing analysis results
    """
    print("="*80)
    print(f"Agent Distance Analysis: M={M}, {case_name.upper()}")
    print("="*80)

    # Load saved results
    print("\nLoading saved results...")
    results = load_results_by_case(M, case_name, output_dir=output_dir)
    states = results['states']
    pi = results['pi']

    print(f"Loaded {len(states)} states")

    # Compute expected distances
    expected_distances = compute_distance_expectations(states, pi, M)

    # Compute average distance
    avg_distance = compute_average_distance(expected_distances, M)
    print(f"\nAverage expected distance: {avg_distance}")

    # Write to markdown file
    write_distance_analysis_to_md(M, case_name, expected_distances, avg_distance, output_dir)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

    return {
        'expected_distances': expected_distances,
        'avg_distance': avg_distance,
        'states': states,
        'pi': pi,
        'M': M,
        'case_name': case_name
    }


if __name__ == "__main__":
    import sys

    # Default: analyze case1 with M=3
    if len(sys.argv) == 1:
        M = 3
        case_name = "case1"
    elif len(sys.argv) == 2:
        M = 3
        case_name = sys.argv[1]
    elif len(sys.argv) == 3:
        M = int(sys.argv[1])
        case_name = sys.argv[2]
    else:
        print("Usage:")
        print("  python analyze_distances.py              # M=3, case1")
        print("  python analyze_distances.py <case>       # M=3, specified case")
        print("  python analyze_distances.py <M> <case>   # Custom M and case")
        sys.exit(1)

    # Run analysis
    analyze_distances_from_results(M, case_name)
