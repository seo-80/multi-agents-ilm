"""
Trace simulation update logic to find symmetry breaking

This script manually traces through a few simulation steps to understand
why d_01 ≠ d_21 when they should be equal.
"""

import numpy as np
import sys
import os
sys.path.insert(0, 'src')

from ilm import networks

# Set random seed for reproducibility
np.random.seed(42)

# Parameters (CASE 4)
agents_count = 3
N_i = 1
m = 0.01
alpha_per_data = 0.001
alpha = alpha_per_data * N_i

# Network matrix
network_args = {"outward_flow_rate": m}
network_matrix = networks.network(agents_count, args=network_args)

print("="*80)
print("Simulation Update Trace (CASE 4)")
print("="*80)

print("\nParameters:")
print(f"  agents_count = {agents_count}")
print(f"  N_i = {N_i}")
print(f"  m = {m}")
print(f"  alpha_per_data = {alpha_per_data}")
print(f"  alpha = {alpha}")

print("\nNetwork matrix W:")
for i in range(agents_count):
    print(f"  Agent {i}: ", end="")
    for j in range(agents_count):
        print(f"{network_matrix[i][j]:.4f}  ", end="")
    print()

# Mutation rates
alphas = np.zeros(agents_count, dtype=float)
alphas[agents_count // 2] = alpha  # Center only
mu = alphas / (N_i + alphas)

print("\nMutation rates:")
for i in range(agents_count):
    print(f"  Agent {i}: mu = {mu[i]:.6f}")

# Initialize state (all agents start with same data)
state = np.zeros((agents_count, N_i, 3), dtype=int)
for i in range(agents_count):
    for k in range(N_i):
        state[i, k] = [0, 0, 0]  # Initial data

print("\n" + "="*80)
print("Initial state:")
print("="*80)
for i in range(agents_count):
    print(f"  Agent {i}: {tuple(state[i, 0])}")

# Run simulation for a few steps
print("\n" + "="*80)
print("Running simulation...")
print("="*80)

t = 0
for step in range(20):
    t += 1

    # 1. Sample data flow counts
    data_flow_count = networks.generate_data_flow_count(
        data_flow_rate=network_matrix,
        total_data_count=N_i
    )

    print(f"\nStep {t}:")
    print(f"  Data flow count:")
    for i in range(agents_count):
        print(f"    Agent {i} receives from: ", end="")
        for j in range(agents_count):
            if data_flow_count[i][j] > 0:
                print(f"agent{j}({int(data_flow_count[i][j])}) ", end="")
        print()

    # 2. Build copy list
    i_idx_list = []
    j_idx_list = []
    local_k_list = []

    for i in range(agents_count):
        cnt = 0
        for j, count in enumerate(data_flow_count[i]):
            for _ in range(int(count)):
                i_idx_list.append(i)
                j_idx_list.append(j)
                local_k_list.append(cnt)
                cnt += 1

    i_idx = np.array(i_idx_list)
    j_idx = np.array(j_idx_list)
    local_k = np.array(local_k_list)

    # 3. Mutation decisions
    mu_j = mu[j_idx]
    mutation_flags = np.random.rand(len(i_idx)) < mu_j

    print(f"  Mutations:")
    for idx in range(len(i_idx)):
        if mutation_flags[idx]:
            print(f"    Agent {i_idx[idx]} receives NEW mutation from agent {j_idx[idx]}")

    # 4. Random indices for copying
    random_indices = np.random.randint(N_i, size=len(i_idx))

    # 5. Update state
    next_state = np.zeros_like(state)

    # Mutations
    if mutation_flags.any():
        for idx in np.where(mutation_flags)[0]:
            i = i_idx[idx]
            k = local_k[idx]
            next_state[i, k] = [t, i, k]

    # Copies
    if (~mutation_flags).any():
        for idx in np.where(~mutation_flags)[0]:
            i = i_idx[idx]
            j = j_idx[idx]
            k = local_k[idx]
            rand_k = random_indices[idx]
            next_state[i, k] = state[j, rand_k]

    state = next_state

    print(f"  New state:")
    for i in range(agents_count):
        print(f"    Agent {i}: {tuple(state[i, 0])}")

    # Check if agents share data
    print(f"  Shared data:")
    same_01 = (tuple(state[0, 0]) == tuple(state[1, 0]))
    same_02 = (tuple(state[0, 0]) == tuple(state[2, 0]))
    same_12 = (tuple(state[1, 0]) == tuple(state[2, 0]))
    print(f"    Agent 0 vs 1: {'SAME' if same_01 else 'DIFF'}")
    print(f"    Agent 0 vs 2: {'SAME' if same_02 else 'DIFF'}")
    print(f"    Agent 1 vs 2: {'SAME' if same_12 else 'DIFF'}")

print("\n" + "="*80)
print("Analysis")
print("="*80)

print("""
Expected behavior for CASE 4 (outward + center-only):
  - Only Agent 1 (center) can mutate
  - Agent 0 copies from Agent 1 with prob 0.01
  - Agent 2 copies from Agent 1 with prob 0.01
  - Agent 1 always copies from itself (no mutation: prob 0.999)

By symmetry:
  - Agent 0 and Agent 2 should behave identically
  - d_01 should equal d_21

If d_01 ≠ d_21, there is a bug in:
  1. Network matrix generation
  2. Data flow sampling
  3. Update logic
  4. Distance calculation
""")
