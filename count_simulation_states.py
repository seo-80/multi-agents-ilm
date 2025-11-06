"""
Count which states are observed in simulation

Map each observed configuration to one of the 5 theoretical states.
"""

import numpy as np
import glob
import os
import collections

sim_dir = "data/naive_simulation/raw/outward_flow-nonzero_alpha_center_fr_0.01_agents_3_N_i_1_alpha_0.001"

# Load state files from long simulation (save_idx 6 onwards)
state_files = []
for idx in range(6, 72):
    file_path = os.path.join(sim_dir, f"state_{idx}.npy")
    if os.path.exists(file_path):
        state_files.append(file_path)

print(f"Found {len(state_files)} state files")

if len(state_files) == 0:
    print("No files found!")
    exit(1)

def classify_state(state):
    """
    Classify a state into one of the 5 theoretical states.

    Returns:
        int: State number (1-5), or 0 if unclassifiable
    """
    agents_count, N_i, _ = state.shape

    if agents_count != 3 or N_i != 1:
        return 0

    # Extract data for each agent
    data = [tuple(state[i, 0]) for i in range(3)]

    # Count unique data
    unique_data = set(data)
    n_unique = len(unique_data)

    if n_unique == 1:
        # State 1: All agents have same data
        return 1

    elif n_unique == 2:
        # Two agents share, one is different
        # Find which agents share
        counter = collections.Counter(data)
        shared_data = [d for d, count in counter.items() if count == 2][0]
        shared_agents = [i for i in range(3) if data[i] == shared_data]
        isolated_agent = [i for i in range(3) if data[i] != shared_data][0]

        # Map to 1-indexed states
        if isolated_agent == 0:  # Agent 0 isolated
            # State 2: {{1},{2,3}}
            return 2
        elif isolated_agent == 1:  # Agent 1 isolated
            # State 3: {{2},{1,3}}
            return 3
        elif isolated_agent == 2:  # Agent 2 isolated
            # State 4: {{3},{1,2}}
            return 4

    elif n_unique == 3:
        # State 5: All agents different
        return 5

    return 0

# Count states
state_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 0: 0}

for state_file in state_files:
    state = np.load(state_file)
    state_num = classify_state(state)
    state_counts[state_num] += 1

print("\n" + "="*80)
print("State frequencies in simulation:")
print("="*80)

total = len(state_files)

for state_num in [1, 2, 3, 4, 5]:
    count = state_counts[state_num]
    freq = count / total
    print(f"\nState {state_num}: {count}/{total} = {freq:.4f}")

    if state_num == 1:
        print("  {{1,2,3}} - All agents same")
    elif state_num == 2:
        print("  {{1},{2,3}} - Agent 0 isolated, Agents 1,2 share")
    elif state_num == 3:
        print("  {{2},{1,3}} - Agent 1 isolated, Agents 0,2 share")
    elif state_num == 4:
        print("  {{3},{1,2}} - Agent 2 isolated, Agents 0,1 share")
    elif state_num == 5:
        print("  {{1},{2},{3}} - All agents different")

if state_counts[0] > 0:
    print(f"\nUnclassifiable states: {state_counts[0]}")

print("\n" + "="*80)
print("Comparison with symbolic analysis:")
print("="*80)

# Expected from symbolic analysis
symbolic_pi = {
    1: 0.8643,
    2: 0.0439,
    3: 0.0434,
    4: 0.0439,
    5: 0.0045
}

print(f"\n{'State':<8} {'Simulation':<12} {'Symbolic':<12} {'Difference':<12}")
print("-" * 48)
for state_num in [1, 2, 3, 4, 5]:
    sim_freq = state_counts[state_num] / total
    sym_freq = symbolic_pi[state_num]
    diff = sim_freq - sym_freq
    print(f"{state_num:<8} {sim_freq:.6f}     {sym_freq:.6f}     {diff:+.6f}")

print("\n" + "="*80)
print("Symmetry check:")
print("="*80)

freq_2 = state_counts[2] / total
freq_4 = state_counts[4] / total

print(f"\nState 2 frequency (Agent 0 isolated): {freq_2:.6f}")
print(f"State 4 frequency (Agent 2 isolated): {freq_4:.6f}")
print(f"Difference: {abs(freq_2 - freq_4):.6f}")

if abs(freq_2 - freq_4) < 0.01:
    print("✓ Symmetric (within 1%)")
else:
    rel_diff = abs(freq_2 - freq_4) / freq_2 * 100
    print(f"✗ NOT symmetric! Relative difference: {rel_diff:.1f}%")

# Statistical significance test
from scipy import stats
# Binomial test: if symmetric, each should occur with p=0.5 among States 2 and 4
n_2_or_4 = state_counts[2] + state_counts[4]
if n_2_or_4 > 0:
    # Under null hypothesis (symmetric), State 2 occurs with p=0.5
    result = stats.binomtest(state_counts[2], n_2_or_4, 0.5)
    print(f"\nBinomial test p-value: {result.pvalue:.4f}")
    if result.pvalue < 0.05:
        print("  ✗ Statistically significant asymmetry (p < 0.05)")
    else:
        print("  ✓ No significant asymmetry (p >= 0.05)")
        print(f"  (With only {n_2_or_4} samples, statistical power is low)")
