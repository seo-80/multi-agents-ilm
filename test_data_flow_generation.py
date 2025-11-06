"""
Test the generate_data_flow_count function to see if it's working correctly
"""

import numpy as np
import sys
import os
sys.path.insert(0, 'src')

from ilm import networks

# Set random seed
np.random.seed(42)

# Parameters
agents_count = 3
m = 0.01
N_i = 1

# Network matrix
network_args = {"outward_flow_rate": m}
W = networks.network(agents_count, args=network_args)

print("="*80)
print("Testing generate_data_flow_count")
print("="*80)

print("\nNetwork matrix W:")
for i in range(agents_count):
    print(f"  Agent {i}: {[f'{W[i][j]:.4f}' for j in range(agents_count)]}")

print("\nRow sums (should all be 1.0):")
for i in range(agents_count):
    row_sum = sum(W[i])
    print(f"  Agent {i}: {row_sum:.6f}")

if not all(abs(sum(W[i]) - 1.0) < 1e-10 for i in range(agents_count)):
    print("\n  ✗ WARNING: Row sums are not 1.0!")
    print("  This means the network matrix is not a valid probability distribution!")
else:
    print("\n  ✓ Row sums are 1.0")

print("\n" + "="*80)
print("Sampling data flow counts")
print("="*80)

# Sample many times to see distribution
num_samples = 100
samples = []

for _ in range(num_samples):
    data_flow_count = networks.generate_data_flow_count(
        data_flow_rate=W,
        total_data_count=N_i
    )
    samples.append(data_flow_count)

samples = np.array(samples)

print(f"\nSampled {num_samples} times with N_i={N_i}")

print("\nEmpirical distribution (should match W for large samples):")
empirical_W = np.mean(samples, axis=0) / N_i

for i in range(agents_count):
    print(f"\n  Agent {i}:")
    print(f"    Expected (W): {[f'{W[i][j]:.4f}' for j in range(agents_count)]}")
    print(f"    Empirical:    {[f'{empirical_W[i][j]:.4f}' for j in range(agents_count)]}")

print("\n" + "="*80)
print("Check specific cases")
print("="*80)

# For N_i=1, each agent must choose exactly one source
print(f"\nFor N_i=1, each sample should have exactly one '1' per row:")

for i in range(min(10, num_samples)):
    print(f"\n  Sample {i}:")
    for agent in range(agents_count):
        row = samples[i, agent]
        print(f"    Agent {agent}: {row}  (sum={sum(row)})")

print("\n" + "="*80)
print("Frequency of Agent 0 copying from each source")
print("="*80)

agent_0_sources = samples[:, 0, :]
freq_0 = np.sum(agent_0_sources, axis=0) / num_samples

print(f"\nAgent 0 (left endpoint):")
print(f"  Copies from Agent 0 (self):   {freq_0[0]:.4f} (expected: {W[0][0]:.4f})")
print(f"  Copies from Agent 1 (center): {freq_0[1]:.4f} (expected: {W[0][1]:.4f})")
print(f"  Copies from Agent 2 (right):  {freq_0[2]:.4f} (expected: {W[0][2]:.4f})")

print(f"\nAgent 2 (right endpoint):")
agent_2_sources = samples[:, 2, :]
freq_2 = np.sum(agent_2_sources, axis=0) / num_samples
print(f"  Copies from Agent 0 (left):   {freq_2[0]:.4f} (expected: {W[2][0]:.4f})")
print(f"  Copies from Agent 1 (center): {freq_2[1]:.4f} (expected: {W[2][1]:.4f})")
print(f"  Copies from Agent 2 (self):   {freq_2[2]:.4f} (expected: {W[2][2]:.4f})")

print("\n" + "="*80)
print("Analysis")
print("="*80)

# Check if Agent 0 and Agent 2 behave symmetrically
freq_0_from_center = freq_0[1]
freq_2_from_center = freq_2[1]

print(f"\nSymmetry check:")
print(f"  Agent 0 copies from center: {freq_0_from_center:.4f}")
print(f"  Agent 2 copies from center: {freq_2_from_center:.4f}")
print(f"  Difference: {abs(freq_0_from_center - freq_2_from_center):.4f}")

if abs(freq_0_from_center - freq_2_from_center) < 0.05:
    print("  ✓ Symmetric (within 5%)")
else:
    print("  ✗ NOT symmetric!")

# Check if frequencies match expected probabilities
print(f"\nFrequency vs Expected:")
for i in range(agents_count):
    print(f"\n  Agent {i}:")
    for j in range(agents_count):
        expected = W[i][j]
        observed = empirical_W[i][j]
        error = abs(observed - expected)
        status = "✓" if error < 0.05 else "✗"
        print(f"    From Agent {j}: observed={observed:.4f}, expected={expected:.4f}, error={error:.4f} {status}")
