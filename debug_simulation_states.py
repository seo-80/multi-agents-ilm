"""
Debug simulation states to understand why d_12 ≠ d_23

This script directly examines simulation states to check for symmetry breaking.
"""

import numpy as np
import glob
import os
import collections

# Load simulation results
import sys

# Use command-line argument if provided
if len(sys.argv) > 1:
    sim_dir = sys.argv[1]
else:
    sim_dir = "data/naive_simulation/raw/outward_flow-nonzero_alpha_center_fr_0.01_agents_3_N_i_1_alpha_0.001"

# Find state files
state_files = sorted(
    glob.glob(os.path.join(sim_dir, "state_*.npy")),
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1])
)

if not state_files:
    print(f"No state files found in {sim_dir}")
    import sys
    sys.exit(1)

print(f"Found {len(state_files)} state files")
print("="*80)

# Examine last few states in detail
for state_file in state_files[-5:]:
    base_name = os.path.basename(state_file)
    save_idx = base_name.split('_')[1].split('.')[0]

    state = np.load(state_file)
    agents_count, N_i, _ = state.shape

    print(f"\nState file: {base_name}")
    print(f"  Shape: {state.shape} (agents={agents_count}, N_i={N_i})")

    # For N_i=1, just check if data matches
    print(f"\n  Data (as tuples):")
    for i in range(agents_count):
        data_tuple = tuple(state[i, 0])
        print(f"    Agent {i}: {data_tuple}")

    # Check which agents share data
    print(f"\n  Shared data:")
    for i in range(agents_count):
        for j in range(i+1, agents_count):
            data_i = tuple(state[i, 0])
            data_j = tuple(state[j, 0])
            if data_i == data_j:
                print(f"    Agent {i} and Agent {j}: SAME")
            else:
                print(f"    Agent {i} and Agent {j}: DIFFERENT")

    # Compute distances manually
    print(f"\n  Manual distance calculation (for N_i=1):")
    counters = [collections.Counter([tuple(d) for d in state[i]]) for i in range(agents_count)]
    for i in range(agents_count):
        for j in range(i+1, agents_count):
            all_keys = set(counters[i]) | set(counters[j])
            dist_ij = sum(abs(counters[i][k] - counters[j][k]) for k in all_keys)
            print(f"    d_{i}{j} = {dist_ij} (Manhattan distance)")
            # For N_i=1, distance should be 0 or 2
            if dist_ij not in [0, 2]:
                print(f"      WARNING: Unexpected distance value!")

print("\n" + "="*80)
print("Analysis")
print("="*80)

# Load all distance files
distance_files = sorted(
    glob.glob(os.path.join(sim_dir, "distance_*.npy")),
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1])
)

if distance_files:
    print(f"\nFound {len(distance_files)} distance files")

    # Compute average distances
    all_distances = []
    for f in distance_files:
        dist = np.load(f)
        all_distances.append(dist)

    all_distances = np.array(all_distances)

    # Skip first 20% as burn-in
    n_samples = all_distances.shape[0]
    burn_in = max(1, n_samples // 5)

    avg_distances = np.mean(all_distances[burn_in:], axis=0)

    print(f"\nAverage distance matrix (skipping first {burn_in} samples):")
    for i in range(3):
        print(f"  Agent {i}: ", end="")
        for j in range(3):
            print(f"{avg_distances[i,j]:.6f}  ", end="")
        print()

    # Check symmetry
    print(f"\nSymmetry check:")
    print(f"  d_01 vs d_10: {avg_distances[0,1]:.6f} vs {avg_distances[1,0]:.6f}")
    print(f"  d_02 vs d_20: {avg_distances[0,2]:.6f} vs {avg_distances[2,0]:.6f}")
    print(f"  d_12 vs d_21: {avg_distances[1,2]:.6f} vs {avg_distances[2,1]:.6f}")

    if not np.allclose(avg_distances, avg_distances.T):
        print("  WARNING: Distance matrix is not symmetric!")
    else:
        print("  ✓ Distance matrix is symmetric")

    # Check expected symmetry d_01 = d_21 (both endpoints to center)
    print(f"\nExpected symmetry (endpoints are equivalent):")
    print(f"  d_01 (left to center):  {avg_distances[0,1]:.6f}")
    print(f"  d_21 (right to center): {avg_distances[2,1]:.6f}")

    if np.isclose(avg_distances[0,1], avg_distances[2,1], rtol=0.01):
        print("  ✓ Endpoints are roughly symmetric")
    else:
        diff = abs(avg_distances[0,1] - avg_distances[2,1])
        rel_diff = diff / avg_distances[0,1] * 100
        print(f"  ✗ Endpoints are NOT symmetric! Difference: {diff:.6f} ({rel_diff:.1f}%)")

    # Check time series for convergence
    print(f"\nConvergence check:")
    d_01_series = all_distances[:, 0, 1]
    d_02_series = all_distances[:, 0, 2]
    d_12_series = all_distances[:, 1, 2]

    print(f"  d_01 (over time): min={d_01_series.min():.1f}, max={d_01_series.max():.1f}, final={d_01_series[-1]:.1f}")
    print(f"  d_02 (over time): min={d_02_series.min():.1f}, max={d_02_series.max():.1f}, final={d_02_series[-1]:.1f}")
    print(f"  d_12 (over time): min={d_12_series.min():.1f}, max={d_12_series.max():.1f}, final={d_12_series[-1]:.1f}")

    # For N_i=1, distances should be 0 or 2
    print(f"\n  All distances are in {{0, 2}}: {np.all(np.isin(all_distances, [0, 2]))}")
    if not np.all(np.isin(all_distances, [0, 2])):
        unique_values = np.unique(all_distances)
        print(f"    Unique values found: {unique_values}")
else:
    print("No distance files found")
