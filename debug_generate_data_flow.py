"""
Debug generate_data_flow_count function to find why Agent 2 doesn't sample correctly
"""

import numpy
import sys
import os
sys.path.insert(0, 'src')

from ilm import networks

#numpy.random.seed(42)

agents_count = 3
m = 0.01

# Generate network matrix
network_args = {"outward_flow_rate": m}
W = networks.network(agents_count, args=network_args)

print("="*80)
print("Debugging generate_data_flow_count")
print("="*80)

print(f"\nNetwork matrix W:")
print(f"  Type: {type(W)}")
print(f"  Shape: {W.shape}")
print(f"  dtype: {W.dtype}")

for i in range(agents_count):
    print(f"  Agent {i}: {W[i]}")

# Manually replicate what generate_data_flow_count does
print(f"\n" + "="*80)
print("Manual replication of generate_data_flow_count:")
print("="*80)

N_i = 1
total_data_counts = [N_i for _ in W]

print(f"\ntotal_data_counts: {total_data_counts}")

# Create empty array like in the function
data_flow_count = numpy.empty_like(W, dtype=int)

print(f"\nEmpty array:")
print(f"  Type: {type(data_flow_count)}")
print(f"  Shape: {data_flow_count.shape}")
print(f"  dtype: {data_flow_count.dtype}")
print(f"  Contents (uninitialized):")
for i in range(agents_count):
    print(f"    {data_flow_count[i]}")

print(f"\n" + "="*80)
print("Iterating and sampling:")
print("="*80)

for i, rate in enumerate(W):
    print(f"\nIteration {i}:")
    print(f"  rate = {rate}")
    print(f"  type(rate) = {type(rate)}")
    print(f"  rate.dtype = {rate.dtype}")
    print(f"  n = {total_data_counts[i]}")

    # Check if rate sums to 1.0
    rate_sum = numpy.sum(rate)
    print(f"  sum(rate) = {rate_sum:.10f}")

    if not numpy.isclose(rate_sum, 1.0):
        print(f"  ✗ WARNING: rate does not sum to 1.0!")

    # Sample
    sample = numpy.random.multinomial(n=total_data_counts[i], pvals=rate)

    print(f"  sample = {sample}")
    print(f"  type(sample) = {type(sample)}")
    print(f"  sample.dtype = {sample.dtype}")

    # Assign
    data_flow_count[i] = sample

    print(f"  data_flow_count[{i}] after assignment: {data_flow_count[i]}")

print(f"\n" + "="*80)
print("Final data_flow_count:")
print("="*80)

for i in range(agents_count):
    print(f"  Agent {i}: {data_flow_count[i]}")

# Compare with actual function
print(f"\n" + "="*80)
print("Comparing with actual function:")
print("="*80)

numpy.random.seed(42)
result1 = networks.generate_data_flow_count(
    data_flow_rate=W,
    total_data_count=N_i
)

print(f"\nResult from function (seed=42):")
for i in range(agents_count):
    print(f"  Agent {i}: {result1[i]}")

# Sample multiple times to see distribution
print(f"\n" + "="*80)
print("Sampling 100 times to see distribution:")
print("="*80)

samples = []
for _ in range(100):
    result = networks.generate_data_flow_count(
        data_flow_rate=W,
        total_data_count=N_i
    )
    samples.append(result)

samples = numpy.array(samples)

print(f"\nEmpirical frequency for each agent:")
for i in range(agents_count):
    freq = numpy.mean(samples[:, i, :], axis=0)
    print(f"\n  Agent {i}:")
    print(f"    Expected: {W[i]}")
    print(f"    Observed: {freq}")

    # Check specific issue with Agent 2
    if i == 2:
        freq_from_center = freq[1]
        expected_from_center = W[i][1]
        print(f"    Copies from center:")
        print(f"      Expected: {expected_from_center:.4f}")
        print(f"      Observed: {freq_from_center:.4f}")

        if numpy.isclose(freq_from_center, 0.0) and not numpy.isclose(expected_from_center, 0.0):
            print(f"      ✗ BUG: Agent 2 never copies from center!")
