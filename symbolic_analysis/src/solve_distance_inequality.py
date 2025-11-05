"""
Solve inequalities for expected distances between agents.

This module solves inequalities of the form E[d_ij] > E[d_kl] to find
parameter regions (m, α) where certain distance relationships hold.
"""

import sympy
from sympy import symbols, simplify, solve, latex, S
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import numpy as np
import os

from m_agent_stationary_symbolic import load_results_by_case
from analyze_distances import compute_distance_expectations


def solve_distance_inequality(expr1: sympy.Expr, expr2: sympy.Expr,
                              inequality: str = '>') -> sympy.Set:
    """
    Solve inequality expr1 <op> expr2 for symbolic parameters.

    Args:
        expr1: First expression (e.g., E[d_12])
        expr2: Second expression (e.g., E[d_13])
        inequality: Inequality operator ('>', '<', '>=', '<=')

    Returns:
        Solution set for the inequality
    """
    print(f"Solving inequality: expr1 {inequality} expr2")

    # Create inequality
    diff = simplify(expr1 - expr2)

    print("Difference (expr1 - expr2):")
    print(f"  {diff}")

    if inequality == '>':
        ineq = diff > 0
    elif inequality == '>=':
        ineq = diff >= 0
    elif inequality == '<':
        ineq = diff < 0
    elif inequality == '<=':
        ineq = diff <= 0
    else:
        raise ValueError(f"Unknown inequality operator: {inequality}")

    print("\nAttempting to solve symbolically...")

    try:
        # Try to solve the inequality
        m, alpha = symbols('m alpha', real=True, positive=True)
        solution = sympy.solve(ineq, (m, alpha))
        print("Solution found!")
        return solution
    except Exception as e:
        print(f"Symbolic solution failed: {e}")
        print("The inequality may be too complex for symbolic solution.")
        return None


def analyze_inequality_numerically(expr1: sympy.Expr, expr2: sympy.Expr,
                                   m_range: Tuple[float, float] = (0.01, 0.99),
                                   alpha_range: Tuple[float, float] = (0.0001, 100.0),
                                   grid_size: int = 100) -> Dict:
    """
    Numerically analyze where expr1 > expr2 in parameter space.

    Args:
        expr1: First expression
        expr2: Second expression
        m_range: Range for m parameter (min, max)
        alpha_range: Range for alpha parameter (min, max)
        grid_size: Number of grid points in each dimension

    Returns:
        Dictionary with grid data and boolean mask where expr1 > expr2
    """
    print("\nNumerical analysis of inequality...")
    print(f"  m range: {m_range}")
    print(f"  α range: {alpha_range}")
    print(f"  Grid size: {grid_size}x{grid_size}")

    m, alpha = symbols('m alpha')

    # Create grid
    m_vals = np.linspace(m_range[0], m_range[1], grid_size)
    alpha_vals = np.linspace(alpha_range[0], alpha_range[1], grid_size)
    M, A = np.meshgrid(m_vals, alpha_vals)

    # Evaluate difference expr1 - expr2 on grid
    diff = expr1 - expr2

    print("  Evaluating expressions on grid...")
    diff_vals = np.zeros_like(M)

    for i in range(grid_size):
        if i % 20 == 0:
            print(f"    Progress: {i}/{grid_size}")
        for j in range(grid_size):
            try:
                val = diff.subs({m: M[i, j], alpha: A[i, j]}).n()
                diff_vals[i, j] = float(val)
            except:
                diff_vals[i, j] = np.nan

    # Create boolean mask where expr1 > expr2
    mask = diff_vals > 0

    print(f"  Fraction where expr1 > expr2: {np.nanmean(mask):.3f}")

    return {
        'm_vals': m_vals,
        'alpha_vals': alpha_vals,
        'M': M,
        'A': A,
        'diff_vals': diff_vals,
        'mask': mask,
        'expr1_gt_expr2': mask
    }


def plot_inequality_region(analysis_result: Dict,
                           pair1: Tuple[int, int],
                           pair2: Tuple[int, int],
                           case_name: str,
                           output_dir: str = None):
    """
    Plot the parameter region where E[d_pair1] > E[d_pair2].

    Args:
        analysis_result: Result from analyze_inequality_numerically
        pair1: First agent pair (i, j)
        pair2: Second agent pair (k, l)
        case_name: Case identifier
        output_dir: Output directory for plot
    """
    # Set default output directory
    if output_dir is None:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(script_dir, "results")

    os.makedirs(output_dir, exist_ok=True)

    M = analysis_result['M']
    A = analysis_result['A']
    mask = analysis_result['mask']

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot regions
    ax.contourf(M, A, mask.astype(int), levels=[0, 0.5, 1],
                colors=['lightcoral', 'lightblue'], alpha=0.6)

    # Add contour line at boundary
    ax.contour(M, A, analysis_result['diff_vals'], levels=[0],
               colors='black', linewidths=2)

    # Labels and title
    ax.set_xlabel('m (coupling strength)', fontsize=12)
    ax.set_ylabel('α (innovation parameter)', fontsize=12)
    ax.set_title(f'Region where E[d_{{{pair1[0]},{pair1[1]}}}] > E[d_{{{pair2[0]},{pair2[1]}}}]\n'
                 f'Case: {case_name.upper()}', fontsize=14)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', alpha=0.6,
              label=f'E[d_{{{pair1[0]},{pair1[1]}}}] > E[d_{{{pair2[0]},{pair2[1]}}}]'),
        Patch(facecolor='lightcoral', alpha=0.6,
              label=f'E[d_{{{pair1[0]},{pair1[1]}}}] ≤ E[d_{{{pair2[0]},{pair2[1]}}}]')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Grid
    ax.grid(True, alpha=0.3)

    # Save figure
    filename = f"M3_{case_name}_inequality_d{pair1[0]}{pair1[1]}_vs_d{pair2[0]}{pair2[1]}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Plot saved to: {filepath}")


def write_inequality_analysis_to_md(pair1: Tuple[int, int],
                                    pair2: Tuple[int, int],
                                    expr1: sympy.Expr,
                                    expr2: sympy.Expr,
                                    analysis_result: Dict,
                                    case_name: str,
                                    M: int = 3,
                                    output_dir: str = None):
    """
    Write inequality analysis results to markdown file.

    Args:
        pair1: First agent pair
        pair2: Second agent pair
        expr1: Expression for E[d_pair1]
        expr2: Expression for E[d_pair2]
        analysis_result: Numerical analysis result
        case_name: Case identifier
        M: Number of agents
        output_dir: Output directory
    """
    # Set default output directory
    if output_dir is None:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(script_dir, "results")

    filename = f"M{M}_{case_name}_inequality_d{pair1[0]}{pair1[1]}_vs_d{pair2[0]}{pair2[1]}.md"
    filepath = os.path.join(output_dir, filename)

    os.makedirs(output_dir, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        # Header
        f.write(f"# Distance Inequality Analysis: M={M}, {case_name.upper()}\n\n")
        f.write(f"## Inequality: $E[d_{{{pair1[0]},{pair1[1]}}}] > E[d_{{{pair2[0]},{pair2[1]}}}]$\n\n")

        # Expressions
        f.write("## Symbolic Expressions\n\n")
        f.write(f"### $E[d_{{{pair1[0]},{pair1[1]}}}]$\n\n")
        f.write(f"$${latex(expr1)}$$\n\n")

        f.write(f"### $E[d_{{{pair2[0]},{pair2[1]}}}]$\n\n")
        f.write(f"$${latex(expr2)}$$\n\n")

        # Difference
        diff = simplify(expr1 - expr2)
        f.write("### Difference\n\n")
        f.write(f"$$E[d_{{{pair1[0]},{pair1[1]}}}] - E[d_{{{pair2[0]},{pair2[1]}}}] = {latex(diff)}$$\n\n")

        # Numerical analysis
        f.write("## Numerical Analysis\n\n")
        f.write(f"We seek parameter regions where $E[d_{{{pair1[0]},{pair1[1]}}}] > E[d_{{{pair2[0]},{pair2[1]}}}]$.\n\n")

        mask = analysis_result['mask']
        diff_vals = analysis_result['diff_vals']

        total = int(np.sum(~np.isnan(diff_vals)))
        positive = int(np.sum(mask))
        fraction = (positive / total) if total else float('nan')

        f.write(f"**Fraction of parameter space where inequality holds**: {positive}/{total} ({fraction:.3f})\n\n")

        # Plot reference
        plot_file = f"M{M}_{case_name}_inequality_d{pair1[0]}{pair1[1]}_vs_d{pair2[0]}{pair2[1]}.png"
        f.write(f"![Parameter Space]({plot_file})\n\n")

        # Interpretation
        f.write("## Interpretation\n\n")
        f.write(f"- **Blue region**: $E[d_{{{pair1[0]},{pair1[1]}}}] > E[d_{{{pair2[0]},{pair2[1]}}}]$ "
                f"(agents {pair1[0]} and {pair1[1]} are more distant than agents {pair2[0]} and {pair2[1]})\n")
        f.write(f"- **Red region**: $E[d_{{{pair1[0]},{pair1[1]}}}] \\leq E[d_{{{pair2[0]},{pair2[1]}}}]$ "
                f"(agents {pair1[0]} and {pair1[1]} are less distant or equally distant)\n")
        f.write("- **Black line**: Boundary where both distances are equal\n\n")

    print(f"Inequality analysis written to: {filepath}")


def analyze_distance_inequality(M: int, case_name: str,
                                pair1: Tuple[int, int],
                                pair2: Tuple[int, int],
                                output_dir: str = None) -> Dict:
    """
    Main function to analyze distance inequality for a given case.

    Args:
        M: Number of agents
        case_name: Case identifier
        pair1: First agent pair (i, j) with i < j
        pair2: Second agent pair (k, l) with k < l
        output_dir: Output directory

    Returns:
        Dictionary with analysis results
    """
    print("="*80)
    print(f"Distance Inequality Analysis: M={M}, {case_name.upper()}")
    print(f"Analyzing: E[d_{{{pair1[0]},{pair1[1]}}}] > E[d_{{{pair2[0]},{pair2[1]}}}]")
    print("="*80)

    # Load saved results
    print("\nLoading saved results...")
    results = load_results_by_case(M, case_name, output_dir=output_dir)
    states = results['states']
    pi = results['pi']

    # Compute expected distances
    print("\nComputing expected distances...")
    expected_distances = compute_distance_expectations(states, pi, M)

    # Get expressions for the two pairs
    expr1 = expected_distances[pair1]
    expr2 = expected_distances[pair2]

    print(f"\nE[d_{{{pair1[0]},{pair1[1]}}}] computed")
    print(f"E[d_{{{pair2[0]},{pair2[1]}}}] computed")

    # Try symbolic solution
    print("\n" + "-"*80)
    print("Attempting symbolic solution...")
    print("-"*80)
    symbolic_solution = solve_distance_inequality(expr1, expr2, '>')

    if symbolic_solution:
        print(f"Symbolic solution: {symbolic_solution}")
    else:
        print("Proceeding with numerical analysis...")

    # Numerical analysis
    print("\n" + "-"*80)
    print("Numerical Analysis")
    print("-"*80)
    analysis_result = analyze_inequality_numerically(expr1, expr2)

    # Create plot
    print("\nGenerating plot...")
    plot_inequality_region(analysis_result, pair1, pair2, case_name, output_dir)

    # Write to markdown
    print("\nWriting analysis to markdown...")
    write_inequality_analysis_to_md(pair1, pair2, expr1, expr2,
                                    analysis_result, case_name, M, output_dir)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

    return {
        'pair1': pair1,
        'pair2': pair2,
        'expr1': expr1,
        'expr2': expr2,
        'symbolic_solution': symbolic_solution,
        'numerical_analysis': analysis_result,
        'case_name': case_name,
        'M': M
    }


if __name__ == "__main__":
    import sys

    # Parse arguments
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python solve_distance_inequality.py <case> <i> <j> <k> <l>")
        print("\nAnalyze inequality E[d_ij] > E[d_kl]")
        print("\nExample:")
        print("  python solve_distance_inequality.py case1 1 2 1 3")
        print("  (Analyzes E[d_12] > E[d_13] for case1)")
        sys.exit(1)

    case_name = sys.argv[1]

    if len(sys.argv) == 6:
        i, j, k, l = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])

        # Ensure pairs are in canonical form (smaller index first)
        pair1 = (min(i, j), max(i, j))
        pair2 = (min(k, l), max(k, l))

        # Run analysis
        analyze_distance_inequality(M=3, case_name=case_name, pair1=pair1, pair2=pair2)

    else:
        print("Error: Expected 5 arguments")
        print("Usage: python solve_distance_inequality.py <case> <i> <j> <k> <l>")
        sys.exit(1)
