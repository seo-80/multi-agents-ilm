"""
M-Agent Infinite Alleles Model - Symbolic Stationary State Computation

This program computes the stationary state of M agents (odd number) in the
Infinite Alleles Model using symbolic computation. It models state transitions
due to data copying and innovation among agents arranged on a one-dimensional
lattice.

The model implements parent-side mutation where mutations occur at the source
during copying, allowing agents with α_i=0 to receive new mutations from others.
"""

import sympy
from sympy import symbols, Matrix, simplify, factor, cancel, expand, latex
from itertools import chain, combinations, product
from typing import List, Set, FrozenSet, Tuple, Iterator, Dict, Any
import os
import pickle
from datetime import datetime


# ============================================================================
# 1. Utility Functions
# ============================================================================

def powerset(iterable) -> Iterator:
    """
    Generate the power set of an iterable.

    Args:
        iterable: Any iterable object

    Returns:
        Iterator yielding all subsets from empty set to full set

    Example:
        powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def partitions(collection: List) -> List[List[Set]]:
    """
    Generate all partitions of a collection.

    A partition is a way of grouping elements into non-empty, non-overlapping
    blocks that cover all elements.

    Args:
        collection: List of elements to partition

    Returns:
        List of partitions, where each partition is a list of sets (blocks)

    Example:
        partitions([1,2,3]) --> [
            [{1,2,3}],
            [{1,2},{3}],
            [{1,3},{2}],
            [{1},{2,3}],
            [{1},{2},{3}]
        ]
    """
    if len(collection) == 1:
        return [[{collection[0]}]]

    first = collection[0]
    rest = collection[1:]
    result = []

    for smaller_partition in partitions(rest):
        # Add first element to each existing block
        for i, block in enumerate(smaller_partition):
            new_partition = [b.copy() for b in smaller_partition]
            new_partition[i].add(first)
            result.append(new_partition)

        # Create new block with just first element
        new_partition = smaller_partition + [{first}]
        result.append(new_partition)

    return result


def get_all_partitions(M: int) -> List[FrozenSet[FrozenSet[int]]]:
    """
    Get all set partitions of M agents (numbered 1 to M).

    Args:
        M: Number of agents (must be odd)

    Returns:
        List of partitions as frozenset of frozensets (for immutability)
    """
    agents = list(range(1, M + 1))
    all_parts = partitions(agents)

    # Convert to frozenset of frozensets to make hashable and order-independent
    result = []
    seen = set()

    for part in all_parts:
        frozen_part = frozenset(frozenset(block) for block in part)
        if frozen_part not in seen:
            seen.add(frozen_part)
            result.append(frozen_part)

    return result


# ============================================================================
# 2. Model Building Functions
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


def compute_mutation_rates(alpha_vec: List) -> List:
    """
    Compute mutation rate for each agent from innovation parameters.

    Args:
        alpha_vec: List of innovation parameters for each agent

    Returns:
        List of mutation rates (symbolic expressions)

    Formula:
        μ_i = α_i / (1 + α_i)

    Note: N_i = 1 is fixed, hence this formula.
    """
    return [alpha_i / (1 + alpha_i) for alpha_i in alpha_vec]


# ============================================================================
# 3. Probability Calculation Functions
# ============================================================================

def prob_copy_block(agent_id: int, block: FrozenSet[int],
                    W: Matrix, mu_vec: List) -> sympy.Expr:
    """
    Calculate probability that agent copies from a block (with parent-side mutation).

    In the parent-side mutation model, mutations occur at the source during copying.
    Agent i successfully copies from block C_l with probability:

        P(i -> C_l) = Σ_{j∈C_l} W_{ij} × (1 - μ_j)

    Args:
        agent_id: Agent ID (1-indexed)
        block: Set of agent IDs forming a block
        W: Weight matrix
        mu_vec: Mutation rate vector

    Returns:
        Symbolic expression for copy probability
    """
    prob = sympy.Integer(0)
    i = agent_id - 1  # Convert to 0-indexed

    for j in block:
        j_idx = j - 1  # Convert to 0-indexed
        # W[i,j] * (1 - mu_j): copy from j without mutation at j
        prob += W[i, j_idx] * (1 - mu_vec[j_idx])

    return prob


def prob_receive_mutation_from(agent_i: int, agent_j: int,
                                W: Matrix, mu_vec: List) -> sympy.Expr:
    """
    Calculate probability that agent i receives a new mutation from agent j.

    In parent-side mutation model:
        P(i receives new mutation from j) = W_{ij} × μ_j

    When agent i copies from j (with weight W_{ij}) and j mutates (with rate μ_j),
    agent i receives a completely new mutation, forming a singleton block.

    Args:
        agent_i: Receiving agent ID (1-indexed)
        agent_j: Mutation source agent ID (1-indexed)
        W: Weight matrix
        mu_vec: Mutation rate vector

    Returns:
        Symbolic expression for new mutation reception probability
    """
    i = agent_i - 1  # Convert to 0-indexed
    j = agent_j - 1
    return W[i, j] * mu_vec[j]


# ============================================================================
# 4. Transition Probability Functions
# ============================================================================

def enumerate_valid_mappings(S: FrozenSet[FrozenSet[int]],
                            S_prime: FrozenSet[FrozenSet[int]]) -> Iterator[Tuple]:
    """
    Enumerate all valid mappings from S' blocks to (S blocks ∪ {mutation}).

    A valid mapping satisfies:
    1. Each non-singleton block in S' maps to exactly one block in S (injective)
    2. Each singleton in S' maps to either:
       - An unused block in S (not mapped by non-singletons), or
       - {mutation} (new mutation)
    3. Mapping to S blocks must be injective (no S block receives >1 arrow)
    4. Only singletons can map to {mutation}

    Args:
        S: Current state (set partition)
        S_prime: Next state (set partition)

    Yields:
        Tuple of (non_singleton_mapping, singleton_mapping):
        - non_singleton_mapping: dict {S' non-singleton block index -> S block}
        - singleton_mapping: dict {S' singleton index -> S block or 'mutation'}

    Example:
        S = {{1,2,3}}, S' = {{1},{2,3}}
        Yields: (
            {0: frozenset({1,2,3})},  # {2,3}' -> {1,2,3}
            {0: 'mutation'}            # {1}' -> mutation
        )
    """
    from itertools import permutations, combinations

    # Separate singletons and non-singletons in S'
    singletons_prime = [b for b in S_prime if len(b) == 1]
    non_singletons_prime = [b for b in S_prime if len(b) > 1]

    # Get S's blocks
    blocks_S = list(S)

    # Check feasibility
    if len(non_singletons_prime) > len(blocks_S):
        return  # Impossible - no valid mappings

    # Generate all injective mappings for non-singletons
    if len(non_singletons_prime) == 0:
        non_singleton_mappings = [{}]
    else:
        non_singleton_mappings = []
        for perm in permutations(blocks_S, len(non_singletons_prime)):
            mapping = {i: block for i, block in enumerate(perm)}
            non_singleton_mappings.append(mapping)

    # For each non-singleton mapping
    for non_sing_map in non_singleton_mappings:
        # Get unused blocks in S
        used_blocks = set(non_sing_map.values())
        unused_blocks = [b for b in blocks_S if b not in used_blocks]

        num_singletons = len(singletons_prime)

        if num_singletons == 0:
            # No singletons - yield just the non-singleton mapping
            yield (non_sing_map, {})
        else:
            # Enumerate all singleton mappings to (unused_blocks ∪ {mutation})
            # Strategy: choose which singletons use unused blocks, rest use mutation

            for num_use_unused in range(min(num_singletons, len(unused_blocks)) + 1):
                # Choose num_use_unused blocks from unused_blocks
                for chosen_blocks in combinations(unused_blocks, num_use_unused):
                    # Choose which num_use_unused singletons will use these blocks
                    for chosen_singleton_indices in combinations(range(num_singletons), num_use_unused):
                        # Map chosen singletons to chosen blocks (all permutations)
                        for block_perm in permutations(chosen_blocks):
                            # Build singleton mapping
                            sing_map = {}
                            for singleton_idx in range(num_singletons):
                                if singleton_idx in chosen_singleton_indices:
                                    # Maps to a block
                                    pos = chosen_singleton_indices.index(singleton_idx)
                                    sing_map[singleton_idx] = block_perm[pos]
                                else:
                                    # Maps to mutation
                                    sing_map[singleton_idx] = 'mutation'

                            yield (non_sing_map, sing_map)


def transition_probability(S: FrozenSet[FrozenSet[int]],
                          S_prime: FrozenSet[FrozenSet[int]],
                          W: Matrix, mu_vec: List, M: int) -> sympy.Expr:
    """
    Calculate transition probability P(S'|S) from state S to state S'.

    This implements the parent-side mutation model where mutations occur at
    the source during copying.

    Args:
        S: Current state (set partition)
        S_prime: Next state (set partition)
        W: Weight matrix
        mu_vec: Mutation rate vector
        M: Number of agents

    Returns:
        Symbolic expression for transition probability P(S'|S)

    Algorithm:
        1. Enumerate all valid mappings from S' blocks to (S blocks ∪ {mutation})
        2. For each mapping, calculate the probability that all agents follow it
        3. Sum probabilities across all valid mappings
    """
    # Separate singletons and non-singletons in S'
    singletons_prime = [b for b in S_prime if len(b) == 1]
    non_singletons_prime = [b for b in S_prime if len(b) > 1]

    total_prob = sympy.Integer(0)

    # Enumerate all valid mappings and sum their probabilities
    for non_sing_map, sing_map in enumerate_valid_mappings(S, S_prime):
        prob_this_mapping = sympy.Integer(1)

        # Probability for non-singletons: each agent copies from mapped S block
        for idx, block_prime in enumerate(non_singletons_prime):
            block_S = non_sing_map[idx]
            for agent in block_prime:
                prob_agent = prob_copy_block(agent, block_S, W, mu_vec)
                prob_this_mapping *= prob_agent

        # Probability for singletons: copy from mapped block or receive mutation
        for singleton_idx, singleton in enumerate(singletons_prime):
            agent = list(singleton)[0]

            target = sing_map[singleton_idx]
            if target == 'mutation':
                # Agent receives new mutation from any source
                prob_agent = sum(
                    prob_receive_mutation_from(agent, j, W, mu_vec)
                    for j in range(1, M + 1)
                )
            else:
                # Agent copies from target block in S
                prob_agent = prob_copy_block(agent, target, W, mu_vec)

            prob_this_mapping *= prob_agent

        total_prob += prob_this_mapping

    return total_prob


def build_transition_matrix(states: List[FrozenSet[FrozenSet[int]]],
                           W: Matrix, mu_vec: List, M: int) -> Matrix:
    """
    Build the transition probability matrix P as symbolic expressions.

    Args:
        states: List of all states
        W: Weight matrix
        mu_vec: Mutation rate vector
        M: Number of agents

    Returns:
        Transition probability matrix P where P[i,j] = P(state_j | state_i)

    Properties:
        - Each row sums to 1 (can be verified symbolically)
        - P[i,j] represents probability of transitioning from state i to state j
    """
    n = len(states)
    P = sympy.zeros(n, n)

    print(f"Building transition matrix ({n}×{n})...")
    for i, S in enumerate(states):
        if i % 5 == 0:
            print(f"  Processing state {i+1}/{n}...")
        for j, S_prime in enumerate(states):
            P[i, j] = transition_probability(S, S_prime, W, mu_vec, M)

    print("Transition matrix built.")
    return P


# ============================================================================
# 5. Stationary Distribution Calculation
# ============================================================================

def find_stationary_distribution(P: Matrix) -> Matrix:
    """
    Find the stationary distribution of transition matrix P symbolically.

    The stationary distribution π satisfies:
        π^T P = π^T
    or equivalently:
        π^T (P - I) = 0
    with the normalization constraint:
        Σ π_i = 1

    We solve this as a linear system instead of an eigenvalue problem,
    which is much faster for symbolic computation.

    Args:
        P: Transition probability matrix (symbolic)

    Returns:
        Stationary distribution π as a column vector (symbolic)

    Algorithm:
        1. Set up linear equations: (P - I)^T π = 0
        2. Replace one equation with normalization: Σ π_i = 1
        3. Solve the linear system using sympy.solve
    """
    print("Computing stationary distribution...")
    n = P.shape[0]

    # Create symbolic variables for π
    pi_vars = [symbols(f'pi_{i}', real=True, positive=True) for i in range(n)]

    print(f"  Setting up linear system for {n} variables...")

    # Build equations from π^T (P - I) = 0
    # Equivalently: (P^T - I) π = 0
    P_T = P.T
    I = sympy.eye(n)
    A = P_T - I

    equations = []

    # Use n-1 equations from (P^T - I) π = 0
    for i in range(n - 1):
        eq = sum(A[i, j] * pi_vars[j] for j in range(n))
        equations.append(eq)

    # Add normalization constraint: Σ π_i = 1
    normalization_eq = sum(pi_vars) - 1
    equations.append(normalization_eq)

    print(f"  Solving system of {len(equations)} equations...")
    # Solve the linear system
    solution = sympy.solve(equations, pi_vars, dict=True)

    if not solution:
        raise ValueError("No solution found for stationary distribution")

    if len(solution) > 1:
        print(f"  Warning: Multiple solutions found ({len(solution)}), using the first one")

    solution = solution[0]

    # Convert solution to column vector
    pi = Matrix([simplify(solution[pi_vars[i]]) for i in range(n)])

    print("Stationary distribution computed.")
    return pi


# ============================================================================
# 6. Output Functions
# ============================================================================

def partition_to_string(partition: FrozenSet[FrozenSet[int]]) -> str:
    """
    Convert a partition to a readable string representation.

    Args:
        partition: Set partition as frozenset of frozensets

    Returns:
        String representation like "{{1,2}, {3}}"

    Blocks are sorted by size (descending) and then by smallest element.
    """
    blocks = [sorted(list(block)) for block in partition]
    # Sort by size (descending) then by first element
    blocks.sort(key=lambda b: (-len(b), b[0]))

    block_strs = ["{" + ",".join(map(str, block)) + "}" for block in blocks]
    return "{" + ", ".join(block_strs) + "}"


def simplify_expression(expr) -> sympy.Expr:
    """
    Simplify a SymPy expression using various techniques.

    Args:
        expr: SymPy expression

    Returns:
        Simplified expression
    """
    expr = simplify(expr)
    expr = cancel(expr)
    expr = factor(expr)
    return expr


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


def save_results(M: int, center_prestige: bool, centralized_neologism_creation: bool,
                states: List, W: Matrix, alpha_vec: List, mu_vec: List,
                P: Matrix, pi: Matrix, output_dir: str = None) -> str:
    """
    Save computation results to a pickle file for later analysis.

    Args:
        M: Number of agents
        center_prestige: Center prestige condition
        centralized_neologism_creation: Centralized neologism creation condition
        states: List of all states
        W: Weight matrix
        alpha_vec: Innovation parameter vector
        mu_vec: Mutation rate vector
        P: Transition probability matrix
        pi: Stationary distribution
        output_dir: Output directory path

    Returns:
        Path to saved pickle file
    """
    # Set default output directory to symbolic_analysis/results
    if output_dir is None:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(script_dir, "results")

    case_name = get_case_name(center_prestige, centralized_neologism_creation)
    filename = f"M{M}_{case_name}.pkl"
    filepath = os.path.join(output_dir, filename)

    os.makedirs(output_dir, exist_ok=True)

    # Package results
    results = {
        'metadata': {
            'M': M,
            'center_prestige': center_prestige,
            'centralized_neologism_creation': centralized_neologism_creation,
            'case_name': case_name,
            'timestamp': datetime.now().isoformat(),
        },
        'states': states,
        'W': W,
        'alpha_vec': alpha_vec,
        'mu_vec': mu_vec,
        'P': P,
        'pi': pi,
    }

    # Save to pickle
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)

    print(f"Results saved to: {filepath}")
    return filepath


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load computation results from a pickle file.

    Args:
        filepath: Path to pickle file

    Returns:
        Dictionary containing:
            - 'metadata': dict with M, center_prestige, centralized_neologism_creation, etc.
            - 'states': List of all states
            - 'W': Weight matrix
            - 'alpha_vec': Innovation parameter vector
            - 'mu_vec': Mutation rate vector
            - 'P': Transition probability matrix
            - 'pi': Stationary distribution
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

    Args:
        M: Number of agents
        case_name: Case identifier ("case1", "case2", "case3", or "case4")
        output_dir: Results directory path

    Returns:
        Dictionary containing computation results
    """
    # Set default output directory to symbolic_analysis/results
    if output_dir is None:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(script_dir, "results")

    filename = f"M{M}_{case_name}.pkl"
    filepath = os.path.join(output_dir, filename)
    return load_results(filepath)


def write_results_to_md(M: int, center_prestige: bool,
                        centralized_neologism_creation: bool,
                        states: List, W: Matrix, alpha_vec: List,
                        mu_vec: List, P: Matrix, pi: Matrix,
                        output_dir: str = None):
    """
    Write computation results to a markdown file.

    Args:
        M: Number of agents
        center_prestige: Center prestige condition
        centralized_neologism_creation: Centralized neologism creation condition
        states: List of all states
        W: Weight matrix
        alpha_vec: Innovation parameter vector
        mu_vec: Mutation rate vector
        P: Transition probability matrix
        pi: Stationary distribution
        output_dir: Output directory path
    """
    # Set default output directory to symbolic_analysis/results
    if output_dir is None:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(script_dir, "results")

    case_name = get_case_name(center_prestige, centralized_neologism_creation)
    filename = f"M{M}_{case_name}.md"
    filepath = os.path.join(output_dir, filename)

    os.makedirs(output_dir, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        # Header
        f.write(f"# M-Agent Stationary State Analysis: M={M}, {case_name.upper()}\n\n")

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

        # Symbolic variables
        f.write("## Symbolic Variables\n\n")
        f.write("- **m**: Coupling strength parameter (0 ≤ m ≤ 1)\n")
        f.write("- **α** (alpha): Innovation parameter (α > 0)\n\n")

        # Weight matrix
        f.write("## Weight Matrix W\n\n")
        f.write("$$\n")
        f.write("W = \\begin{bmatrix}\n")
        for i in range(M):
            row_elements = [latex(simplify_expression(W[i, j])) for j in range(M)]
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

        f.write("### Mutation rates (μ_i = α_i / (1 + α_i))\n\n")
        for i, mu_i in enumerate(mu_vec, 1):
            f.write(f"- $\\mu_{{{i}}} = {latex(simplify_expression(mu_i))}$\n")
        f.write("\n")

        # States
        f.write("## State Space\n\n")
        f.write(f"**Total number of states**: {len(states)}\n\n")
        for i, state in enumerate(states):
            f.write(f"{i+1}. {partition_to_string(state)}\n")
        f.write("\n")

        # Transition matrix (can be very large, so we show it compactly)
        f.write("## Transition Probability Matrix P\n\n")
        f.write(f"Matrix size: {len(states)}×{len(states)}\n\n")
        f.write("$P[i,j] = P(\\text{state}_j | \\text{state}_i)$\n\n")

        # For display, show matrix in LaTeX format
        f.write("$$\n")
        f.write("P = \\begin{bmatrix}\n")
        for i in range(len(states)):
            row_elements = [latex(simplify_expression(P[i, j])) for j in range(len(states))]
            row_str = " & ".join(row_elements)
            if i < len(states) - 1:
                f.write(f"{row_str} \\\\\n")
            else:
                f.write(f"{row_str}\n")
        f.write("\\end{bmatrix}\n")
        f.write("$$\n\n")

        # Stationary distribution
        f.write("## Stationary Distribution π\n\n")
        f.write("The long-term probability of being in each state:\n\n")
        for i, state in enumerate(states):
            prob = simplify_expression(pi[i])
            f.write(f"**State {i+1}**: {partition_to_string(state)}\n\n")
            f.write(f"$$\\pi_{{{i+1}}} = {latex(prob)}$$\n\n")

        # Verification
        f.write("## Verification\n\n")
        f.write("Sum of stationary probabilities:\n\n")
        f.write(f"$$\\sum_i \\pi_i = {latex(simplify_expression(sum(pi)))}$$\n\n")
        f.write("(Should equal 1)\n\n")

    print(f"Results written to: {filepath}")


# ============================================================================
# 7. Main Computation Function
# ============================================================================

def compute_stationary_state_symbolic(M: int = 3,
                                     center_prestige: bool = False,
                                     centralized_neologism_creation: bool = False,
                                     output_dir: str = None):
    """
    Compute the stationary state of M-agent infinite alleles model symbolically.

    This is the main entry point for the symbolic computation.

    Args:
        M: Number of agents (must be odd, default: 3)
        center_prestige: Use center-prestige model (default: False)
        centralized_neologism_creation: Only center creates innovations (default: False)
        output_dir: Directory to save output files (default: symbolic_analysis/results)

    Returns:
        Tuple of (states, pi, P, W):
            - states: List of all states
            - pi: Stationary distribution (symbolic)
            - P: Transition probability matrix (symbolic)
            - W: Weight matrix (symbolic)
    """
    # Set default output directory to symbolic_analysis/results
    if output_dir is None:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(script_dir, "results")

    # Check M is odd
    if M % 2 == 0:
        raise ValueError("M must be odd")

    case_name = get_case_name(center_prestige, centralized_neologism_creation)

    print("="*80)
    print(f"M-Agent Infinite Alleles Model - Symbolic Stationary State Computation")
    print("="*80)
    print(f"\nCase: {case_name.upper()}")
    print(f"Parameters: M={M}, center_prestige={center_prestige}, "
          f"centralized_neologism_creation={centralized_neologism_creation}\n")

    # Define symbolic variables
    m, alpha = symbols('m alpha', real=True, positive=True)
    print(f"Symbolic variables: m (coupling), α (innovation)\n")

    # Build W matrix
    print("Building weight matrix W...")
    W = build_W_matrix(M, center_prestige, m)
    print("Weight matrix W:")
    for i in range(M):
        row_str = "  ".join(str(simplify_expression(W[i, j])) for j in range(M))
        print(f"  {row_str}")
    print()

    # Build alpha vector
    print("Building innovation parameter vector...")
    alpha_vec = build_alpha_vector(M, centralized_neologism_creation, alpha)
    print("Alpha vector:")
    for i, alpha_i in enumerate(alpha_vec, 1):
        print(f"  α_{i} = {alpha_i}")
    print()

    # Compute mutation rates
    print("Computing mutation rates...")
    mu_vec = compute_mutation_rates(alpha_vec)
    print("Mutation rates:")
    for i, mu_i in enumerate(mu_vec, 1):
        print(f"  μ_{i} = {simplify_expression(mu_i)}")
    print()

    # Get all states
    print("Enumerating all states...")
    states = get_all_partitions(M)
    print(f"Total number of states: {len(states)}")
    print("States:")
    for i, state in enumerate(states):
        print(f"  {i+1}. {partition_to_string(state)}")
    print()

    # Build transition matrix
    P = build_transition_matrix(states, W, mu_vec, M)
    print()

    # Find stationary distribution
    pi = find_stationary_distribution(P)
    print()

    # Display results
    print("="*80)
    print("RESULTS")
    print("="*80)
    print("\nStationary Distribution:\n")
    for i, state in enumerate(states):
        prob = simplify_expression(pi[i])
        print(f"State {i+1}: {partition_to_string(state)}")
        print(f"  π_{i+1} = {prob}\n")

    print("Sum of probabilities:", simplify_expression(sum(pi)))
    print()

    # Write to files
    write_results_to_md(M, center_prestige, centralized_neologism_creation,
                       states, W, alpha_vec, mu_vec, P, pi, output_dir)

    save_results(M, center_prestige, centralized_neologism_creation,
                states, W, alpha_vec, mu_vec, P, pi, output_dir)

    print("="*80)
    print("Computation complete!")
    print("="*80)

    return states, pi, P, W
