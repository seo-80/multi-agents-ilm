"""
F-Matrix Symbolic Computation for IBD Analysis

This module computes the stationary F-matrix (Identity By Descent probability)
using symbolic computation with SymPy.

F-matrix update equations (stationary state F(t+1) = F(t) = F):

For i ≠ j:
    F_ij = Σ_k Σ_l W_ik W_jl (1-μ_k)(1-μ_l) F_kl

For i = j:
    F_ii = 1/N + (1-1/N) Σ_k Σ_l W_ik W_il (1-μ_k)(1-μ_l) F_kl

In matrix form:
    D = diag(1-μ_0, 1-μ_1, ..., 1-μ_{M-1})
    G = W D F D W^T
    F_ij = G_ij (for i≠j)
    F_ii = 1/N + (1-1/N) G_ii

Symmetries of the F-matrix (always hold regardless of the model):
    1. Transpose symmetry: F_ij = F_ji
    2. Mirror symmetry around center c = M//2: F_{c-i,c-j} = F_{c+i,c+j}

These symmetries significantly reduce the number of independent variables
in the symbolic computation.
"""

import sympy
from sympy import symbols, Matrix, simplify, factor, cancel, latex, solve, Eq
import os
import pickle
from datetime import datetime
from typing import List, Dict, Any, Tuple


# ============================================================================
# 1. Model Building Functions
# ============================================================================

def build_W_matrix(M: int, center_prestige: bool, m: sympy.Symbol) -> Matrix:
    """
    Build the weight matrix W as a symbolic expression.

    W[i,j] represents the weight that agent i gives to agent j's data when copying.

    Args:
        M: Number of agents (odd)
        center_prestige: If True, use center-to-periphery asymmetric interaction
        m: Symbolic parameter for coupling strength

    Returns:
        M×M symbolic matrix W

    Model definitions:

    center_prestige = False (symmetric bidirectional):
        W[i,i] = 1 - m/2  if i=1 or i=M (endpoints)
        W[i,i] = 1 - m    if 1 < i < M (internal)
        W[i,j] = m/2      if |i-j| = 1 (adjacent)
        W[i,j] = 0        otherwise

    center_prestige = True (center-outward asymmetric):
        center c = (M+1)/2
        W[i,i] = 1        if i=c (center refers only to itself)
        W[i,i] = 1-m      if i≠c (non-center self-reference)
        W[i,j] = m        if j≤c and i=j-1 (left of center: j→i=j-1)
        W[i,j] = m        if j≥c and i=j+1 (right of center: j→i=j+1)
        W[i,j] = 0        otherwise
    """
    W = sympy.zeros(M, M)

    if not center_prestige:
        # Symmetric bidirectional model
        for i in range(1, M + 1):
            for j in range(1, M + 1):
                if i == j:
                    if i == 1 or i == M:  # Endpoints
                        W[i-1, j-1] = 1 - m/2
                    else:  # Internal
                        W[i-1, j-1] = 1 - m
                elif abs(i - j) == 1:  # Adjacent
                    W[i-1, j-1] = m/2
    else:
        # Center-prestige model (center-outward asymmetric)
        c = (M + 1) // 2  # Center agent

        for i in range(1, M + 1):
            for j in range(1, M + 1):
                if i == j:
                    if i == c:
                        W[i-1, j-1] = 1  # Center only refers to itself
                    else:
                        W[i-1, j-1] = 1 - m  # Non-center self-reference
                elif j <= c and i == j - 1:
                    # Left of center: j flows to left neighbor i=j-1
                    W[i-1, j-1] = m
                elif j >= c and i == j + 1:
                    # Right of center: j flows to right neighbor i=j+1
                    W[i-1, j-1] = m

    return W


def build_alpha_vector(M: int, centralized_neologism_creation: bool,
                       alpha: sympy.Symbol) -> List:
    """
    Build the innovation parameter vector for each agent.

    Args:
        M: Number of agents (odd)
        centralized_neologism_creation: If True, only center creates innovations
        alpha: Symbolic innovation parameter

    Returns:
        List of alpha values for each agent

    Definitions:
        centralized_neologism_creation = False:
            α_i = α for all i

        centralized_neologism_creation = True:
            α_c = α  (center agent c = (M+1)/2)
            α_i = 0  for all i ≠ c
    """
    if not centralized_neologism_creation:
        return [alpha] * M
    else:
        c = (M + 1) // 2  # Center agent
        alpha_vec = [sympy.Integer(0)] * M
        alpha_vec[c - 1] = alpha  # c-1 because of 0-indexing
        return alpha_vec


def compute_mutation_rates(alpha_vec: List, N: sympy.Symbol) -> List:
    """
    Compute mutation rate for each agent from innovation parameters.

    Args:
        alpha_vec: List of innovation parameters for each agent
        N: Population size (symbolic)

    Returns:
        List of mutation rates (symbolic expressions)

    Formula:
        μ_i = α_i / (N + α_i)
    """
    return [alpha_i / (N + alpha_i) for alpha_i in alpha_vec]


def get_case_name(center_prestige: bool, centralized_neologism_creation: bool) -> str:
    """
    Get case identifier string for the four model cases.

    Returns:
        String like "case1", "case2", "case3", "case4"
    """
    if not center_prestige and not centralized_neologism_creation:
        return "case1"
    elif center_prestige and not centralized_neologism_creation:
        return "case2"
    elif not center_prestige and centralized_neologism_creation:
        return "case3"
    else:  # center_prestige and centralized_neologism_creation
        return "case4"


# ============================================================================
# 2. F-Matrix Stationary State Computation
# ============================================================================

def identify_symmetries(M: int, center_prestige: bool) -> Dict[Tuple[int, int], str]:
    """
    Identify symmetries in the F-matrix to reduce the number of variables.

    The F-matrix always has the following symmetries regardless of the model:
    1. Transpose symmetry: F_ij = F_ji
    2. Mirror symmetry around center: F_{c-i,c-j} = F_{c+i,c+j}

    These symmetries allow us to use a canonical representative for each
    equivalence class of matrix positions.

    Args:
        M: Number of agents
        center_prestige: Not used (kept for backward compatibility)

    Returns:
        Dictionary mapping (i, j) to variable name
    """
    var_map = {}
    center = M // 2

    for i in range(M):
        for j in range(M):
            # Generate all symmetric positions under the two symmetries:
            # 1. F_ij = F_ji (transpose)
            # 2. F_{c-k,c-l} = F_{c+k,c+l} (mirror around center)
            positions = [
                (i, j),
                (j, i),  # transpose symmetry
                (2*center - i, 2*center - j),  # mirror symmetry
                (2*center - j, 2*center - i),  # both symmetries
            ]

            # Keep only valid positions (within bounds [0, M))
            valid_positions = [(x, y) for x, y in positions
                             if 0 <= x < M and 0 <= y < M]

            # Use canonical form: lexicographically smallest position
            canonical = min(valid_positions)

            # Generate variable name from canonical form
            ci, cj = canonical
            if ci == cj:
                # Diagonal element: characterized by distance from center
                dist_from_center = abs(ci - center)
                var_name = f'f_diag_{dist_from_center}'
            else:
                # Off-diagonal element: use canonical (i,j) pair
                var_name = f'f_{ci}_{cj}'

            var_map[(i, j)] = var_name

    return var_map


def compute_f_matrix_stationary(M: int = 3,
                                center_prestige: bool = False,
                                centralized_neologism_creation: bool = False,
                                output_dir: str = None,
                                verbose: bool = True) -> Tuple[Matrix, Dict]:
    """
    Compute the stationary F-matrix symbolically.

    Args:
        M: Number of agents (must be odd, default: 3)
        center_prestige: Use center-prestige model (default: False)
        centralized_neologism_creation: Only center creates innovations (default: False)
        output_dir: Directory to save output files (default: IBD_analysis/results)
        verbose: Print progress messages

    Returns:
        Tuple of (F_matrix, metadata):
            - F_matrix: Stationary F-matrix (symbolic)
            - metadata: Dict with W, alpha_vec, mu_vec, etc.
    """
    # Set default output directory
    if output_dir is None:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(script_dir, "results")

    # Check M is odd
    if M % 2 == 0:
        raise ValueError("M must be odd")

    case_name = get_case_name(center_prestige, centralized_neologism_creation)

    if verbose:
        print("="*80)
        print(f"F-Matrix Stationary State - Symbolic Computation")
        print("="*80)
        print(f"\nCase: {case_name.upper()}")
        print(f"Parameters: M={M}, center_prestige={center_prestige}, "
              f"centralized_neologism_creation={centralized_neologism_creation}\n")

    # Define symbolic variables
    N, m, alpha = symbols('N m alpha', real=True, positive=True)
    if verbose:
        print(f"Symbolic variables: N (population), m (coupling), α (innovation)\n")

    # Build W matrix
    if verbose:
        print("Building weight matrix W...")
    W = build_W_matrix(M, center_prestige, m)
    if verbose:
        print("Weight matrix W:")
        for i in range(M):
            row_str = "  ".join(str(W[i, j]) for j in range(M))
            print(f"  {row_str}")
        print()

    # Build alpha vector
    if verbose:
        print("Building innovation parameter vector...")
    alpha_vec = build_alpha_vector(M, centralized_neologism_creation, alpha)
    if verbose:
        print("Alpha vector:")
        for i, alpha_i in enumerate(alpha_vec, 1):
            print(f"  α_{i} = {alpha_i}")
        print()

    # Compute mutation rates
    if verbose:
        print("Computing mutation rates...")
    mu_vec = compute_mutation_rates(alpha_vec, N)
    if verbose:
        print("Mutation rates:")
        for i, mu_i in enumerate(mu_vec, 1):
            print(f"  μ_{i} = {mu_i}")
        print()

    # Identify symmetries
    if verbose:
        print("Identifying symmetries in F-matrix...")
        print("  Using: F_ij = F_ji (transpose) and F_{c-i,c-j} = F_{c+i,c+j} (mirror)")
    var_map = identify_symmetries(M, center_prestige)

    # Get unique variable names
    unique_vars = sorted(set(var_map.values()))
    if verbose:
        print(f"Number of unique F-matrix elements (due to symmetry): {len(unique_vars)}")
        print(f"Variables: {unique_vars}\n")

    # Create symbolic variables for F-matrix elements
    f_vars_dict = {var_name: symbols(var_name, real=True, positive=True)
                   for var_name in unique_vars}

    # Build symbolic F-matrix
    F_symbolic = sympy.zeros(M, M)
    for i in range(M):
        for j in range(M):
            var_name = var_map[(i, j)]
            F_symbolic[i, j] = f_vars_dict[var_name]

    if verbose:
        print("Building D matrix (diagonal mutation matrix)...")
    D = Matrix.diag(*[1 - mu for mu in mu_vec])

    if verbose:
        print("Computing G = W D F D W^T...")
    # Compute G = W @ D @ F @ D @ W.T
    G = W * D * F_symbolic * D * W.T

    # Simplify G (this might take time)
    if verbose:
        print("Simplifying G matrix elements...")
    G_simplified = sympy.zeros(M, M)
    for i in range(M):
        for j in range(M):
            if verbose and (i * M + j) % 5 == 0:
                print(f"  Simplifying G[{i},{j}]... ({i*M+j+1}/{M*M})")
            G_simplified[i, j] = simplify(G[i, j])

    # Set up equations
    if verbose:
        print("\nSetting up stationary equations F = G (modified for diagonal)...")
    equations = []

    for i in range(M):
        for j in range(M):
            if i == j:
                # Diagonal: F_ii = 1/N + (1 - 1/N) * G_ii
                lhs = F_symbolic[i, i]
                rhs = 1/N + (1 - 1/N) * G_simplified[i, i]
                equations.append(Eq(lhs, rhs))
            else:
                # Off-diagonal: F_ij = G_ij
                lhs = F_symbolic[i, j]
                rhs = G_simplified[i, j]
                equations.append(Eq(lhs, rhs))

    # Remove duplicate equations (due to symmetry)
    unique_equations = []
    seen_vars = set()
    for eq in equations:
        var_name = None
        for (i, j), name in var_map.items():
            if F_symbolic[i, j] == eq.lhs:
                var_name = name
                break
        if var_name and var_name not in seen_vars:
            unique_equations.append(eq)
            seen_vars.add(var_name)

    if verbose:
        print(f"Total equations: {len(equations)}")
        print(f"Unique equations (after removing symmetry duplicates): {len(unique_equations)}\n")

    # Solve the system
    if verbose:
        print("Solving system of equations...")
        print("(This may take several minutes depending on M and model complexity...)")

    solution = solve(unique_equations, list(f_vars_dict.values()), dict=True)

    if not solution:
        raise ValueError("No solution found for stationary F-matrix")

    if len(solution) > 1:
        if verbose:
            print(f"Warning: Multiple solutions found ({len(solution)}), using the first one")

    solution = solution[0]

    # Build final F-matrix with solutions
    if verbose:
        print("\nSimplifying solutions...")
    F_solution = sympy.zeros(M, M)
    for i in range(M):
        for j in range(M):
            var_name = var_map[(i, j)]
            F_solution[i, j] = simplify(solution[f_vars_dict[var_name]])

    if verbose:
        print("Solution found and simplified!\n")

    # Prepare metadata
    metadata = {
        'M': M,
        'center_prestige': center_prestige,
        'centralized_neologism_creation': centralized_neologism_creation,
        'case_name': case_name,
        'W': W,
        'alpha_vec': alpha_vec,
        'mu_vec': mu_vec,
        'symbolic_variables': ['N', 'm', 'alpha'],
        'timestamp': datetime.now().isoformat(),
    }

    # Save results
    save_results(M, case_name, F_solution, metadata, output_dir)
    write_results_to_md(M, case_name, F_solution, metadata, output_dir)

    if verbose:
        print("="*80)
        print("Computation complete!")
        print("="*80)

    return F_solution, metadata


# ============================================================================
# 3. Output Functions
# ============================================================================

def save_results(M: int, case_name: str, F_matrix: Matrix,
                metadata: Dict, output_dir: str):
    """
    Save computation results to a pickle file.

    Args:
        M: Number of agents
        case_name: Case identifier
        F_matrix: Stationary F-matrix
        metadata: Metadata dictionary
        output_dir: Output directory path
    """
    filename = f"M{M}_{case_name}.pkl"
    filepath = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'metadata': metadata,
        'F_matrix': F_matrix,
    }

    with open(filepath, 'wb') as f:
        pickle.dump(results, f)

    print(f"Results saved to: {filepath}")


# Global cache for loaded results to avoid redundant file I/O
_RESULTS_CACHE = {}


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load computation results from a pickle file.

    Args:
        filepath: Path to pickle file

    Returns:
        Dictionary containing metadata and F_matrix
    """
    with open(filepath, 'rb') as f:
        results = pickle.load(f)

    print(f"Results loaded from: {filepath}")
    print(f"  M = {results['metadata']['M']}")
    print(f"  Case: {results['metadata']['case_name']}")
    print(f"  Timestamp: {results['metadata']['timestamp']}")

    return results


def load_results_by_case(M: int, case_name: str, output_dir: str = None) -> Dict[str, Any]:
    """
    Load computation results by M and case name.

    Results are cached in memory to avoid redundant file I/O when the same
    file is loaded multiple times within a single process.

    Args:
        M: Number of agents
        case_name: Case identifier ("case1", "case2", "case3", or "case4")
        output_dir: Results directory path

    Returns:
        Dictionary containing computation results
    """
    # Determine output directory
    if output_dir is None:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(script_dir, "results")

    # Create cache key
    cache_key = (M, case_name, output_dir)

    # Check cache first
    if cache_key in _RESULTS_CACHE:
        print(f"Results loaded from cache: M={M}, case={case_name}")
        return _RESULTS_CACHE[cache_key]

    # Load from file if not cached
    filename = f"M{M}_{case_name}.pkl"
    filepath = os.path.join(output_dir, filename)
    results = load_results(filepath)

    # Store in cache
    _RESULTS_CACHE[cache_key] = results

    return results


def clear_results_cache():
    """
    Clear the results cache to free memory.

    This is useful when processing many different cases and memory usage
    becomes a concern.
    """
    global _RESULTS_CACHE
    num_cached = len(_RESULTS_CACHE)
    _RESULTS_CACHE.clear()
    print(f"Cleared {num_cached} cached result(s)")


def write_results_to_md(M: int, case_name: str, F_matrix: Matrix,
                        metadata: Dict, output_dir: str):
    """
    Write computation results to a markdown file.

    Args:
        M: Number of agents
        case_name: Case identifier
        F_matrix: Stationary F-matrix
        metadata: Metadata dictionary
        output_dir: Output directory path
    """
    filename = f"M{M}_{case_name}.md"
    filepath = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)

    W = metadata['W']
    alpha_vec = metadata['alpha_vec']
    mu_vec = metadata['mu_vec']
    center_prestige = metadata['center_prestige']
    centralized_neologism_creation = metadata['centralized_neologism_creation']

    with open(filepath, 'w', encoding='utf-8') as f:
        # Header
        f.write(f"# F-Matrix Stationary Solution: M={M}, {case_name.upper()}\n\n")

        # Parameters
        f.write("## Parameters\n\n")
        f.write(f"- **Number of agents (M)**: {M}\n")
        f.write(f"- **Center prestige**: {center_prestige}\n")
        f.write(f"- **Centralized neologism creation**: {centralized_neologism_creation}\n\n")

        # Case description
        f.write("## Case Description\n\n")
        if case_name == "case1":
            f.write("**Case 1**: Symmetric bidirectional interaction + Uniform innovation\n")
            f.write("- All agents interact symmetrically with neighbors\n")
            f.write("- All agents create innovations equally (α_i = α for all i)\n\n")
        elif case_name == "case2":
            f.write("**Case 2**: Center-outward asymmetric interaction + Uniform innovation\n")
            f.write("- Information flows from center to periphery\n")
            f.write("- All agents create innovations equally (α_i = α for all i)\n\n")
        elif case_name == "case3":
            f.write("**Case 3**: Symmetric bidirectional interaction + Center-only innovation\n")
            f.write("- All agents interact symmetrically with neighbors\n")
            f.write("- Only center agent creates innovations (α_c = α, α_i = 0 for i ≠ c)\n\n")
        else:  # case4
            f.write("**Case 4**: Center-outward asymmetric interaction + Center-only innovation\n")
            f.write("- Information flows from center to periphery\n")
            f.write("- Only center agent creates innovations (α_c = α, α_i = 0 for i ≠ c)\n\n")

        # Model equations
        f.write("## Model Equations\n\n")
        f.write("F-matrix update equations (stationary state):\n\n")
        f.write("For $i \\neq j$:\n")
        f.write("$$F_{ij} = \\sum_{k}\\sum_{l}W_{ik}W_{jl}(1-\\mu_k)(1-\\mu_l)F_{kl}$$\n\n")
        f.write("For $i = j$:\n")
        f.write("$$F_{ii} = \\frac{1}{N}+\\left(1-\\frac{1}{N}\\right)\\sum_{k}\\sum_{l}W_{ik}W_{il}(1-\\mu_k)(1-\\mu_l)F_{kl}$$\n\n")

        # Symbolic variables
        f.write("## Symbolic Variables\n\n")
        f.write("- **N**: Population size (N > 0)\n")
        f.write("- **m**: Coupling strength parameter (0 ≤ m ≤ 1)\n")
        f.write("- **α** (alpha): Innovation parameter (α > 0)\n\n")

        # Weight matrix
        f.write("## Weight Matrix W\n\n")
        f.write("$$\n")
        f.write("W = \\begin{bmatrix}\n")
        for i in range(M):
            row_elements = [latex(W[i, j]) for j in range(M)]
            row_str = " & ".join(row_elements)
            if i < M - 1:
                f.write(f"{row_str} \\\\\n")
            else:
                f.write(f"{row_str}\n")
        f.write("\\end{bmatrix}\n")
        f.write("$$\n\n")

        # Alpha and mu vectors
        f.write("## Innovation Parameters\n\n")
        f.write("### Alpha vector (α_i)\n\n")
        for i, alpha_i in enumerate(alpha_vec, 1):
            f.write(f"- $\\alpha_{{{i}}} = {latex(alpha_i)}$\n")
        f.write("\n")

        f.write("### Mutation rates\n\n")
        f.write("$$\\mu_i = \\frac{\\alpha_i}{N + \\alpha_i}$$\n\n")
        for i, mu_i in enumerate(mu_vec, 1):
            f.write(f"- $\\mu_{{{i}}} = {latex(mu_i)}$\n")
        f.write("\n")

        # F-matrix solution
        f.write("## F-Matrix (Stationary Solution)\n\n")
        f.write("$$\n")
        f.write("F = \\begin{bmatrix}\n")
        for i in range(M):
            row_elements = [latex(F_matrix[i, j]) for j in range(M)]
            row_str = " & ".join(row_elements)
            if i < M - 1:
                f.write(f"{row_str} \\\\\n")
            else:
                f.write(f"{row_str}\n")
        f.write("\\end{bmatrix}\n")
        f.write("$$\n\n")

        # Individual elements
        f.write("## F-Matrix Elements\n\n")
        for i in range(M):
            for j in range(M):
                f.write(f"$$F_{{{i+1},{j+1}}} = {latex(F_matrix[i, j])}$$\n\n")

    print(f"Results written to: {filepath}")


# ============================================================================
# 4. Main Function
# ============================================================================

def main():
    """
    Main function to compute F-matrix for all cases and multiple M values.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Compute F-matrix stationary solutions')
    parser.add_argument('--M', type=int, nargs='+', default=[3],
                       help='Agent counts to compute (must be odd), e.g., --M 3 5 7')
    parser.add_argument('--cases', type=str, nargs='+',
                       default=['case1', 'case2', 'case3', 'case4'],
                       help='Cases to compute, e.g., --cases case1 case2')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: IBD_analysis/results)')

    args = parser.parse_args()

    # Map case names to boolean flags
    case_configs = {
        'case1': (False, False),  # (center_prestige, centralized_neologism_creation)
        'case2': (True, False),
        'case3': (False, True),
        'case4': (True, True),
    }

    for M in args.M:
        if M % 2 == 0:
            print(f"Warning: M={M} is even, skipping (M must be odd)")
            continue

        for case_name in args.cases:
            if case_name not in case_configs:
                print(f"Warning: Unknown case '{case_name}', skipping")
                continue

            center_prestige, centralized_neologism_creation = case_configs[case_name]

            print(f"\n{'='*80}")
            print(f"Computing M={M}, {case_name}")
            print(f"{'='*80}\n")

            try:
                compute_f_matrix_stationary(
                    M=M,
                    center_prestige=center_prestige,
                    centralized_neologism_creation=centralized_neologism_creation,
                    output_dir=args.output_dir,
                    verbose=True
                )
            except Exception as e:
                print(f"\nError computing M={M}, {case_name}:")
                print(f"  {type(e).__name__}: {e}")
                print("Continuing with next case...\n")
                continue


if __name__ == "__main__":
    main()
