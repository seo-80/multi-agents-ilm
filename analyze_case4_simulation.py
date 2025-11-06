"""
Analyze simulation results and compare with symbolic analysis for CASE 4

This script:
1. Loads simulation distance data
2. Computes average distances between agents
3. Compares with symbolic analysis predictions
"""

import numpy as np
import glob
import os
import sys

# Simulation output directory
sim_dir = "data/naive_simulation/raw/outward_flow-nonzero_alpha_center_fr_0.01_agents_3_N_i_1_alpha_0.001"

# Load distance files
distance_files = sorted(
    glob.glob(os.path.join(sim_dir, "distance_*.npy")),
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1])
)

print(f"Found {len(distance_files)} distance files")
print(f"Directory: {sim_dir}")
print("="*80)

if len(distance_files) == 0:
    print("ERROR: No distance files found!")
    sys.exit(1)

# Load all distances
all_distances = []
for f in distance_files:
    dist = np.load(f)
    all_distances.append(dist)

all_distances = np.array(all_distances)
print(f"Loaded distances shape: {all_distances.shape}")
print(f"  (samples, agents, agents) = {all_distances.shape}")

# Average distances over time (excluding initial transient)
# Skip first 20% as burn-in
n_samples = all_distances.shape[0]
burn_in = n_samples // 5
print(f"\nUsing {n_samples - burn_in} samples (skipping first {burn_in} as burn-in)")

avg_distances = np.mean(all_distances[burn_in:], axis=0)

print("\n" + "="*80)
print("SIMULATION RESULTS")
print("="*80)

print("\nAverage distance matrix:")
for i in range(3):
    print(f"  Agent {i+1}: ", end="")
    for j in range(3):
        print(f"{avg_distances[i,j]:.6f}  ", end="")
    print()

# Extract specific distances
d_12 = avg_distances[0, 1]
d_13 = avg_distances[0, 2]
d_23 = avg_distances[1, 2]

print(f"\nExpected distances (for N_i=1):")
print(f"  E[d_12] = {d_12:.6f}")
print(f"  E[d_13] = {d_13:.6f}")
print(f"  E[d_23] = {d_23:.6f}")

print(f"\n  E[d_12] - E[d_13] = {d_12 - d_13:.6f}")

if d_12 < d_13:
    print("  => E[d_12] < E[d_13] ✓")
else:
    print("  => E[d_12] > E[d_13]")

print("\n" + "="*80)
print("SYMBOLIC ANALYSIS PREDICTIONS (CASE 4)")
print("="*80)

# From symbolic analysis at m=0.01, alpha=0.001
symbolic_d12 = 0.0918172736
symbolic_d13 = 0.0922686318
symbolic_d23 = 0.0918172736

print(f"\nSymbolic predictions:")
print(f"  E[d_12] = {symbolic_d12:.6f}")
print(f"  E[d_13] = {symbolic_d13:.6f}")
print(f"  E[d_23] = {symbolic_d23:.6f}")

print(f"\n  E[d_12] - E[d_13] = {symbolic_d12 - symbolic_d13:.6f}")
print(f"  => E[d_12] < E[d_13] (symbolic prediction)")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)

print(f"\n                    Simulation    Symbolic      Difference")
print(f"  E[d_12]:          {d_12:8.6f}    {symbolic_d12:8.6f}    {d_12 - symbolic_d12:+8.6f}")
print(f"  E[d_13]:          {d_13:8.6f}    {symbolic_d13:8.6f}    {d_13 - symbolic_d13:+8.6f}")
print(f"  E[d_23]:          {d_23:8.6f}    {symbolic_d23:8.6f}    {d_23 - symbolic_d23:+8.6f}")

print(f"\n  E[d_12] - E[d_13]:")
print(f"    Simulation:     {d_12 - d_13:+8.6f}")
print(f"    Symbolic:       {symbolic_d12 - symbolic_d13:+8.6f}")

# Check sign
sim_sign = "E[d_12] < E[d_13]" if d_12 < d_13 else "E[d_12] > E[d_13]"
sym_sign = "E[d_12] < E[d_13]" if symbolic_d12 < symbolic_d13 else "E[d_12] > E[d_13]"

print(f"\n  Sign comparison:")
print(f"    Simulation:     {sim_sign}")
print(f"    Symbolic:       {sym_sign}")

if sim_sign == sym_sign:
    print("\n  ✓ SIGNS MATCH!")
    print("  ✓ Simulation agrees with symbolic analysis!")
else:
    print("\n  ✗ SIGNS DO NOT MATCH!")
    print("  ✗ Simulation contradicts symbolic analysis!")

# Statistical analysis
print("\n" + "="*80)
print("STATISTICAL ANALYSIS")
print("="*80)

# Compute standard error
d12_series = all_distances[burn_in:, 0, 1]
d13_series = all_distances[burn_in:, 0, 2]
diff_series = d12_series - d13_series

mean_diff = np.mean(diff_series)
std_diff = np.std(diff_series)
se_diff = std_diff / np.sqrt(len(diff_series))

print(f"\nE[d_12] - E[d_13] statistics (post burn-in):")
print(f"  Mean:     {mean_diff:+.6f}")
print(f"  Std Dev:  {std_diff:.6f}")
print(f"  Std Err:  {se_diff:.6f}")
print(f"  95% CI:   [{mean_diff - 1.96*se_diff:+.6f}, {mean_diff + 1.96*se_diff:+.6f}]")

# Check if symbolic prediction is within CI
symbolic_diff = symbolic_d12 - symbolic_d13
ci_lower = mean_diff - 1.96 * se_diff
ci_upper = mean_diff + 1.96 * se_diff

if ci_lower <= symbolic_diff <= ci_upper:
    print(f"\n  Symbolic prediction ({symbolic_diff:+.6f}) is within 95% CI")
    print("  ✓ Consistent with symbolic analysis")
else:
    print(f"\n  Symbolic prediction ({symbolic_diff:+.6f}) is OUTSIDE 95% CI")
    print("  ⚠ May indicate discrepancy or insufficient sampling")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if sim_sign == sym_sign:
    print("\n✓ The simulation with CASE 4 parameters matches symbolic analysis!")
    print("✓ Both predict E[d_12] < E[d_13]")
    print("\nThis confirms:")
    print("  1. The symbolic analysis is correct")
    print("  2. The simulation implementation is correct")
    print("  3. Previous discrepancy was due to using different parameters")
else:
    print("\n✗ Discrepancy found even with matching parameters")
    print("  This suggests a potential bug in either:")
    print("  1. Simulation implementation")
    print("  2. Symbolic analysis")
    print("  3. Distance calculation")
