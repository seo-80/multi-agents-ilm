"""
Test network matrix generation to understand the direction of information flow
"""

import numpy as np
import sys
import os
sys.path.insert(0, 'src')

from ilm import networks

# Test CASE 4: outward flow with M=3
M = 3
m = 0.01

print("="*80)
print("Testing Network Matrix for M=3, outward_flow_rate=0.01")
print("="*80)

network_args = {"outward_flow_rate": m}
W = networks.network(M, args=network_args)

print("\nNetwork Matrix W:")
print("  (W[i][j] = probability that agent i copies from agent j)")
print()
for i in range(M):
    print(f"  Agent {i}: ", end="")
    for j in range(M):
        print(f"{W[i][j]:.4f}  ", end="")
    print()

print("\nInterpretation:")
print("  Agent 0 (left endpoint):")
print(f"    - Copies from self: {W[0][0]:.4f}")
print(f"    - Copies from center (agent 1): {W[0][1]:.4f}")
print(f"    - Copies from right (agent 2): {W[0][2]:.4f}")

print("\n  Agent 1 (center):")
print(f"    - Copies from left (agent 0): {W[1][0]:.4f}")
print(f"    - Copies from self: {W[1][1]:.4f}")
print(f"    - Copies from right (agent 2): {W[1][2]:.4f}")

print("\n  Agent 2 (right endpoint):")
print(f"    - Copies from left (agent 0): {W[2][0]:.4f}")
print(f"    - Copies from center (agent 1): {W[2][1]:.4f}")
print(f"    - Copies from self: {W[2][2]:.4f}")

print("\n" + "="*80)
print("EXPECTED for OUTWARD FLOW (center → periphery)")
print("="*80)
print("""
Expected: Information flows FROM center TO periphery
  - Endpoints copy FROM center
  - Center does not copy from endpoints (or copies from self)

Expected matrix:
  Agent 0: [0.99, 0.01, 0.00]  <- copies from center with prob 0.01
  Agent 1: [0.00, 1.00, 0.00]  <- center only copies from self
  Agent 2: [0.00, 0.01, 0.99]  <- copies from center with prob 0.01
""")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)

expected_W = np.array([
    [0.99, 0.01, 0.00],
    [0.00, 1.00, 0.00],
    [0.00, 0.01, 0.99]
])

print("\nExpected W:")
for i in range(M):
    print(f"  Agent {i}: ", end="")
    for j in range(M):
        print(f"{expected_W[i][j]:.4f}  ", end="")
    print()

print("\nActual W:")
for i in range(M):
    print(f"  Agent {i}: ", end="")
    for j in range(M):
        print(f"{W[i][j]:.4f}  ", end="")
    print()

print("\nDifference (Actual - Expected):")
for i in range(M):
    print(f"  Agent {i}: ", end="")
    for j in range(M):
        diff = W[i][j] - expected_W[i][j]
        print(f"{diff:+.4f}  ", end="")
    print()

if np.allclose(W, expected_W):
    print("\n✓ Network matrix matches expected outward flow!")
else:
    print("\n✗ Network matrix DOES NOT match expected outward flow!")
    print("\nActual interpretation:")
    if W[0][1] > 0:
        print("  - Agent 0 copies from agent 1 (center)")
    if W[2][1] > 0:
        print("  - Agent 2 copies from agent 1 (center)")
    if W[1][0] > 0:
        print("  - Agent 1 copies from agent 0 (left)")
    if W[1][2] > 0:
        print("  - Agent 1 copies from agent 2 (right)")

print("\n" + "="*80)
print("CODE ANALYSIS")
print("="*80)

print("""
From networks.py, the outward_flow implementation:

```python
if "outward_flow_rate" in args:
    outward_frow_rate=args["outward_flow_rate"]
    for ai in range(agents_count):
        if ai < center_index:
            return_network[ai][ai+1]+=outward_frow_rate
            return_network[ai][ai]-=outward_frow_rate
        elif ai > center_index:
            return_network[ai][ai-1]+=outward_frow_rate
            return_network[ai][ai]-=outward_frow_rate
```

This means:
  - ai < center_index: W[ai][ai+1] += m (left agents copy from RIGHT neighbor)
  - ai > center_index: W[ai][ai-1] += m (right agents copy from LEFT neighbor)

For M=3, center_index = 1:
  - Agent 0 (ai=0 < 1): W[0][1] += 0.01  ✓ copies from center
  - Agent 2 (ai=2 > 1): W[2][1] += 0.01  ✓ copies from center

So the implementation IS correct for outward flow!
""")
