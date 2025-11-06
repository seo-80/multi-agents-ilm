"""
Trace network matrix generation step by step to find the bug
"""

import numpy

agents_count = 3
m = 0.01
center_index = agents_count // 2  # = 1

print("="*80)
print("Tracing Network Generation for CASE 4")
print("="*80)

print(f"\nParameters:")
print(f"  agents_count = {agents_count}")
print(f"  outward_flow_rate = {m}")
print(f"  center_index = {center_index}")

# Start with identity
W = numpy.identity(agents_count)

print(f"\nInitial matrix (identity):")
for i in range(agents_count):
    print(f"  Agent {i}: {W[i]}")

# Apply outward flow
print(f"\nApplying outward flow:")
print(f"  Code: if ai < center_index: W[ai][ai+1] += m, W[ai][ai] -= m")
print(f"        if ai > center_index: W[ai][ai-1] += m, W[ai][ai] -= m")

for ai in range(agents_count):
    print(f"\n  Agent {ai} (ai={ai}, center_index={center_index}):")

    if ai < center_index:
        print(f"    ai < center_index: TRUE")
        print(f"    W[{ai}][{ai+1}] += {m}  (was: {W[ai][ai+1]:.4f})")
        W[ai][ai+1] += m
        print(f"    W[{ai}][{ai+1}] = {W[ai][ai+1]:.4f}")

        print(f"    W[{ai}][{ai}] -= {m}  (was: {W[ai][ai]:.4f})")
        W[ai][ai] -= m
        print(f"    W[{ai}][{ai}] = {W[ai][ai]:.4f}")

    elif ai > center_index:
        print(f"    ai > center_index: TRUE")
        print(f"    W[{ai}][{ai-1}] += {m}  (was: {W[ai][ai-1]:.4f})")
        W[ai][ai-1] += m
        print(f"    W[{ai}][{ai-1}] = {W[ai][ai-1]:.4f}")

        print(f"    W[{ai}][{ai}] -= {m}  (was: {W[ai][ai]:.4f})")
        W[ai][ai] -= m
        print(f"    W[{ai}][{ai}] = {W[ai][ai]:.4f}")

    else:
        print(f"    ai == center_index: TRUE (no modification)")

    print(f"    Row after: {W[ai]}")

print(f"\n" + "="*80)
print("Final matrix:")
print("="*80)

for i in range(agents_count):
    print(f"  Agent {i}: {W[i]}")
    print(f"    Row sum: {W[i].sum():.6f}")

print(f"\n" + "="*80)
print("Expected matrix for CASE 4:")
print("="*80)

expected_W = numpy.array([
    [0.99, 0.01, 0.00],
    [0.00, 1.00, 0.00],
    [0.00, 0.01, 0.99]
])

for i in range(agents_count):
    print(f"  Agent {i}: {expected_W[i]}")

print(f"\n" + "="*80)
print("Comparison:")
print("="*80)

for i in range(agents_count):
    print(f"\n  Agent {i}:")
    print(f"    Actual:   {W[i]}")
    print(f"    Expected: {expected_W[i]}")
    diff = W[i] - expected_W[i]
    print(f"    Diff:     {diff}")

    if numpy.allclose(W[i], expected_W[i]):
        print(f"    ✓ MATCH")
    else:
        print(f"    ✗ MISMATCH!")

print(f"\n" + "="*80)
print("Test multinomial sampling with Agent 2's row:")
print("="*80)

numpy.random.seed(42)

W_row2 = W[2]
print(f"\nAgent 2's row: {W_row2}")
print(f"  Sum: {W_row2.sum():.6f}")

# Check if row sums to 1.0
if not numpy.isclose(W_row2.sum(), 1.0):
    print(f"  ✗ WARNING: Row does not sum to 1.0!")
else:
    print(f"  ✓ Row sums to 1.0")

# Check if all probabilities are non-negative
if not all(W_row2 >= 0):
    print(f"  ✗ WARNING: Row contains negative probabilities!")
    print(f"  Negative indices: {numpy.where(W_row2 < 0)[0]}")
else:
    print(f"  ✓ All probabilities are non-negative")

# Sample 10 times
print(f"\n  Sampling 10 times with multinomial(n=1, pvals={W_row2}):")

for i in range(10):
    sample = numpy.random.multinomial(n=1, pvals=W_row2)
    selected = numpy.argmax(sample)
    print(f"    Sample {i}: {sample} → selected agent {selected}")

print(f"\n  Expected distribution: Agent 0: 0%, Agent 1: 1%, Agent 2: 99%")

# Sample 1000 times for better statistics
samples = []
for _ in range(1000):
    sample = numpy.random.multinomial(n=1, pvals=W_row2)
    selected = numpy.argmax(sample)
    samples.append(selected)

samples = numpy.array(samples)
counts = numpy.bincount(samples, minlength=3)
freqs = counts / 1000

print(f"\n  Empirical distribution (1000 samples):")
print(f"    Agent 0: {freqs[0]:.3f} (expected: 0.000)")
print(f"    Agent 1: {freqs[1]:.3f} (expected: 0.010)")
print(f"    Agent 2: {freqs[2]:.3f} (expected: 0.990)")
