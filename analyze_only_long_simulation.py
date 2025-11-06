"""
Analyze only the long simulation results (save_idx >= 6)
"""

import numpy as np
import glob
import os

sim_dir = "data/naive_simulation/raw/outward_flow-nonzero_alpha_center_fr_0.01_agents_3_N_i_1_alpha_0.001"

# Load only distance files from save_idx 6 onwards
distance_files = []
for idx in range(6, 72):  # save_idx 6-71
    file_path = os.path.join(sim_dir, f"distance_{idx}.npy")
    if os.path.exists(file_path):
        distance_files.append(file_path)

print(f"Found {len(distance_files)} distance files from long simulation (save_idx 6-71)")

if len(distance_files) == 0:
    print("No files found!")
    exit(1)

# Load all distances
all_distances = []
for f in distance_files:
    dist = np.load(f)
    all_distances.append(dist)

all_distances = np.array(all_distances)

print(f"Shape: {all_distances.shape}")

# Skip first 20% as burn-in
n_samples = all_distances.shape[0]
burn_in = max(1, n_samples // 5)

print(f"\nUsing {n_samples - burn_in} samples (skipping first {burn_in} as burn-in)")

avg_distances = np.mean(all_distances[burn_in:], axis=0)

print("\n" + "="*80)
print("Average distance matrix (long simulation only):")
print("="*80)

for i in range(3):
    print(f"  Agent {i}: ", end="")
    for j in range(3):
        print(f"{avg_distances[i,j]:.6f}  ", end="")
    print()

# Check symmetry
d_01 = avg_distances[0, 1]
d_02 = avg_distances[0, 2]
d_12 = avg_distances[1, 2]
d_10 = avg_distances[1, 0]
d_20 = avg_distances[2, 0]
d_21 = avg_distances[2, 1]

print("\n" + "="*80)
print("Symmetry analysis:")
print("="*80)

print(f"\nDistance matrix symmetry:")
print(f"  d_01 = d_10: {d_01:.6f} = {d_10:.6f}  {'✓' if np.isclose(d_01, d_10) else '✗'}")
print(f"  d_02 = d_20: {d_02:.6f} = {d_20:.6f}  {'✓' if np.isclose(d_02, d_20) else '✗'}")
print(f"  d_12 = d_21: {d_12:.6f} = {d_21:.6f}  {'✓' if np.isclose(d_12, d_21) else '✗'}")

print(f"\nEndpoint symmetry (d_01 should equal d_21):")
print(f"  d_01 (left→center):  {d_01:.6f}")
print(f"  d_21 (right→center): {d_21:.6f}")
print(f"  Difference: {abs(d_01 - d_21):.6f}")

if np.isclose(d_01, d_21, rtol=0.01):
    print("  ✓ Endpoints are symmetric (within 1%)")
else:
    rel_diff = abs(d_01 - d_21) / d_01 * 100
    print(f"  ✗ Endpoints are NOT symmetric! Relative difference: {rel_diff:.1f}%")

# Compare with symbolic analysis
print("\n" + "="*80)
print("Comparison with symbolic analysis:")
print("="*80)

symbolic_d12 = 0.0918172736
symbolic_d13 = 0.0922686318

# For N_i=1, simulation uses Manhattan distance (0 or 2)
# Symbolic uses 0/1 distance
# To compare, we need to normalize simulation by dividing by 2

norm_d01 = d_01 / 2
norm_d02 = d_02 / 2
norm_d12 = d_12 / 2

print(f"\nNormalized simulation distances (divide by 2 for N_i=1):")
print(f"  E[d_12] = {norm_d01:.6f}  (symbolic: {symbolic_d12:.6f})")
print(f"  E[d_13] = {norm_d02:.6f}  (symbolic: {symbolic_d13:.6f})")
print(f"  E[d_23] = {norm_d12:.6f}  (symbolic: {0.0918172736:.6f})")

print(f"\nSign comparison:")
sim_sign = "E[d_12] < E[d_13]" if norm_d01 < norm_d02 else "E[d_12] > E[d_13]" if norm_d01 > norm_d02 else "E[d_12] = E[d_13]"
sym_sign = "E[d_12] < E[d_13]" if symbolic_d12 < symbolic_d13 else "E[d_12] > E[d_13]"

print(f"  Simulation: {sim_sign}")
print(f"  Symbolic:   {sym_sign}")

if sim_sign == sym_sign:
    print("  ✓ Signs MATCH!")
else:
    print("  ✗ Signs DO NOT MATCH!")

# Time series analysis
print("\n" + "="*80)
print("Time series analysis:")
print("="*80)

d_01_series = all_distances[:, 0, 1]
d_02_series = all_distances[:, 0, 2]
d_12_series = all_distances[:, 1, 2]

print(f"\nOver {n_samples} samples:")
print(f"  d_01: mean={np.mean(d_01_series):.4f}, std={np.std(d_01_series):.4f}")
print(f"  d_02: mean={np.mean(d_02_series):.4f}, std={np.std(d_02_series):.4f}")
print(f"  d_12: mean={np.mean(d_12_series):.4f}, std={np.std(d_12_series):.4f}")

# Check if all values are in {0, 2}
unique_vals = np.unique(all_distances)
print(f"\nUnique distance values: {unique_vals}")
if np.all(np.isin(unique_vals, [0, 2])):
    print("  ✓ All distances are in {0, 2} as expected for N_i=1")
else:
    print(f"  ✗ Unexpected distance values found!")
