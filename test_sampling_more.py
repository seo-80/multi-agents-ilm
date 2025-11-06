"""
Test with much larger sample size to see if Agent 2 actually samples from center
"""

import numpy as np
import sys
import os
sys.path.insert(0, 'src')

from ilm import networks

agents_count = 3
m = 0.01
N_i = 1

# Generate network matrix
network_args = {"outward_flow_rate": m}
W = networks.network(agents_count, args=network_args)

print("="*80)
print("Large-scale sampling test")
print("="*80)

print(f"\nNetwork matrix W:")
for i in range(agents_count):
    print(f"  Agent {i}: {W[i]}")

# Sample many times
n_samples = 100000

print(f"\nSampling {n_samples:,} times...")

samples = []
for _ in range(n_samples):
    result = networks.generate_data_flow_count(
        data_flow_rate=W,
        total_data_count=N_i
    )
    samples.append(result)

samples = np.array(samples)

print(f"\nEmpirical frequency for each agent:")
for i in range(agents_count):
    freq = np.mean(samples[:, i, :], axis=0)
    print(f"\n  Agent {i}:")
    print(f"    Expected: {W[i]}")
    print(f"    Observed: {freq}")

    # Check error
    error = np.abs(freq - W[i])
    max_error = np.max(error)
    print(f"    Max error: {max_error:.6f}")

    if max_error < 0.001:
        print(f"    ✓ MATCH (within 0.1%)")
    else:
        print(f"    ⚠ Some deviation (but may be OK)")

# Focus on Agent 2
print(f"\n" + "="*80)
print("Detailed analysis for Agent 2:")
print("="*80)

agent2_samples = samples[:, 2, :]
counts = np.sum(agent2_samples, axis=0)
freq = counts / n_samples

print(f"\nAgent 2 copying sources ({n_samples:,} samples):")
print(f"  From Agent 0: {counts[0]:,} times ({freq[0]:.6f}, expected: {W[2][0]:.6f})")
print(f"  From Agent 1: {counts[1]:,} times ({freq[1]:.6f}, expected: {W[2][1]:.6f})")
print(f"  From Agent 2: {counts[2]:,} times ({freq[2]:.6f}, expected: {W[2][2]:.6f})")

# Statistical test
expected_from_center = W[2][1] * n_samples
observed_from_center = counts[1]
std_error = np.sqrt(n_samples * W[2][1] * (1 - W[2][1]))

print(f"\nStatistical test (binomial):")
print(f"  Expected count from center: {expected_from_center:.1f}")
print(f"  Observed count from center: {observed_from_center}")
print(f"  Standard error: {std_error:.1f}")
print(f"  Z-score: {(observed_from_center - expected_from_center) / std_error:.3f}")

if abs(observed_from_center - expected_from_center) < 3 * std_error:
    print(f"  ✓ Within 3 standard deviations (normal variation)")
else:
    print(f"  ✗ More than 3 standard deviations (likely a bug!)")

# Check for Agent 0 too
print(f"\n" + "="*80)
print("Comparison: Agent 0 vs Agent 2 (should be symmetric)")
print("="*80)

agent0_samples = samples[:, 0, :]
agent0_from_center = np.sum(agent0_samples[:, 1])
agent2_from_center = np.sum(agent2_samples[:, 1])

print(f"\nCopying from center:")
print(f"  Agent 0: {agent0_from_center:,} times ({agent0_from_center/n_samples:.6f})")
print(f"  Agent 2: {agent2_from_center:,} times ({agent2_from_center/n_samples:.6f})")
print(f"  Expected: {W[0][1]:.6f} (both should be same)")

diff = abs(agent0_from_center - agent2_from_center)
print(f"\n  Difference: {diff:,} ({diff/n_samples*100:.3f}%)")

# Binomial test for symmetry
pooled_freq = (agent0_from_center + agent2_from_center) / (2 * n_samples)
std_diff = np.sqrt(2 * n_samples * pooled_freq * (1 - pooled_freq))

print(f"  Z-score for difference: {diff / std_diff:.3f}")

if diff / std_diff < 3:
    print(f"  ✓ Symmetric (within 3σ)")
else:
    print(f"  ✗ NOT symmetric (>3σ difference)")
