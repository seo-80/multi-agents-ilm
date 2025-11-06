"""
Analyze the local_k logic in simulation update

The question: Is local_k correctly indexing the data slots?
"""

import numpy as np

print("="*80)
print("Analyzing local_k logic")
print("="*80)

# Simulate the index building process
agents_count = 3
N_i = 1

# Example data_flow_count
# Agent 0 copies from Agent 1 (1 time)
# Agent 1 copies from Agent 1 (1 time)
# Agent 2 copies from Agent 2 (1 time)
data_flow_count = np.array([
    [0, 1, 0],  # Agent 0: 1 copy from agent 1
    [0, 1, 0],  # Agent 1: 1 copy from agent 1
    [0, 0, 1],  # Agent 2: 1 copy from agent 2
])

print("\nExample data_flow_count:")
for i in range(agents_count):
    print(f"  Agent {i}: {data_flow_count[i]}")

# Build index arrays as in the simulation
i_idx_list = []
j_idx_list = []
local_k_list = []

print("\nBuilding index arrays:")
print("  (i_idx, j_idx, local_k) represents: Agent i's data slot local_k copies from Agent j")

for i in range(agents_count):
    cnt = 0
    print(f"\n  Agent {i} (cnt starts at 0):")
    for j, count in enumerate(data_flow_count[i]):
        print(f"    From Agent {j}: {int(count)} copies")
        for _ in range(int(count)):
            i_idx_list.append(i)
            j_idx_list.append(j)
            local_k_list.append(cnt)
            print(f"      → (i={i}, j={j}, local_k={cnt})")
            cnt += 1

i_idx = np.array(i_idx_list)
j_idx = np.array(j_idx_list)
local_k = np.array(local_k_list)

print("\n" + "="*80)
print("Result arrays:")
print("="*80)

for idx in range(len(i_idx)):
    print(f"  [{idx}]: i={i_idx[idx]}, j={j_idx[idx]}, local_k={local_k[idx]}")
    print(f"        → state[{i_idx[idx]}, {local_k[idx]}] will be updated from state[{j_idx[idx]}, ?]")

print("\n" + "="*80)
print("Analysis:")
print("="*80)

print("""
The logic increments cnt for each copy operation for agent i, regardless of
which agent j it's copying from.

For N_i=1:
  - Each agent has exactly 1 data slot (index 0)
  - data_flow_count[i] sums to N_i=1
  - Therefore, local_k should always be 0

Let's check:
""")

for i in range(agents_count):
    copies_for_agent_i = (i_idx == i)
    local_k_values = local_k[copies_for_agent_i]
    print(f"  Agent {i}: local_k values = {local_k_values}")

    if np.all(local_k_values == 0):
        print(f"    ✓ All are 0 (correct for N_i=1)")
    else:
        print(f"    ✗ Contains non-zero values (BUG!)")

print("\n" + "="*80)
print("Test with N_i=2:")
print("="*80)

N_i = 2
data_flow_count_2 = np.array([
    [1, 1, 0],  # Agent 0: 1 from self, 1 from agent 1
    [0, 2, 0],  # Agent 1: 2 from self
    [0, 1, 1],  # Agent 2: 1 from agent 1, 1 from self
])

print("\ndata_flow_count (N_i=2):")
for i in range(agents_count):
    print(f"  Agent {i}: {data_flow_count_2[i]} (sum={data_flow_count_2[i].sum()})")

i_idx_list = []
j_idx_list = []
local_k_list = []

for i in range(agents_count):
    cnt = 0
    for j, count in enumerate(data_flow_count_2[i]):
        for _ in range(int(count)):
            i_idx_list.append(i)
            j_idx_list.append(j)
            local_k_list.append(cnt)
            cnt += 1

i_idx = np.array(i_idx_list)
j_idx = np.array(j_idx_list)
local_k = np.array(local_k_list)

print("\nIndex arrays:")
for idx in range(len(i_idx)):
    print(f"  [{idx}]: i={i_idx[idx]}, j={j_idx[idx]}, local_k={local_k[idx]}")

print("\nlocal_k values for each agent:")
for i in range(agents_count):
    copies_for_agent_i = (i_idx == i)
    local_k_values = local_k[copies_for_agent_i]
    print(f"  Agent {i}: local_k = {local_k_values}")

    expected = np.arange(N_i)
    if np.array_equal(local_k_values, expected):
        print(f"    ✓ Correct: covers all slots [0, {N_i-1}]")
    else:
        print(f"    ⚠ Values: {local_k_values}, Expected: {expected}")

print("\n" + "="*80)
print("Conclusion:")
print("="*80)

print("""
The local_k logic IS correct:
  - For N_i=1: local_k is always 0
  - For N_i=2: local_k covers [0, 1]
  - In general: local_k enumerates data slots 0 to N_i-1 for each agent

This is NOT the source of the asymmetry bug.
""")
