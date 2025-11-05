"""
Analysis of self-transition probabilities for State 2 vs State 3

This is KEY to understanding why π₃ might be greater than π₂
despite P(5→2) > P(5→3).

Parameters: M=3, outward flow, m=0.01, alpha=0.001 (center only)

W matrix (1-indexed):
       j=1      j=2      j=3
i=1 [ 0.99     0.01      0   ]
i=2 [  0        1        0   ]
i=3 [  0       0.01     0.99 ]

μ₁ = 0, μ₂ ≈ 0.001, μ₃ = 0

================================================================================
STATE 2: {{1,2},{3}} - LEFT and CENTER share data, RIGHT different
================================================================================

For State 2 → State 2, we need agents 1 and 2 to STAY in the same block,
and agent 3 to stay alone.

Agents 1 and 2 currently share data D_A.
Agent 3 has different data D_B.

Possible mappings for transition {{1,2},{3}} → {{1,2},{3}}:

Mapping 1: Block {1,2} in S' copies from block {1,2} in S
- Agent 1 copies from {1,2}:
  - From agent 1: W[1,1] × (1-μ₁) = 0.99 × 1 = 0.99
  - From agent 2: W[1,2] × (1-μ₂) = 0.01 × 0.999 ≈ 0.00999
  - Total: 0.99 + 0.00999 ≈ 0.99999

- Agent 2 copies from {1,2}:
  - From agent 1: W[2,1] × (1-μ₁) = 0 × 1 = 0
  - From agent 2: W[2,2] × (1-μ₂) = 1 × 0.999 ≈ 0.999
  - Total: 0.999

- Agent 3 copies from {3} or mutates:
  - From agent 3: W[3,3] × (1-μ₃) = 0.99 × 1 = 0.99
  - Mutation: W[3,2] × μ₂ = 0.01 × 0.001 ≈ 0.00001
  - Total: 0.99 + 0.00001 ≈ 0.99001

Combined: 0.99999 × 0.999 × 0.99001 ≈ 0.989

Wait, but there's a PROBLEM here!

Agent 1 and 2 need to STAY TOGETHER. But:
- Agent 1 can copy from either agent 1 OR agent 2
- Agent 2 can ONLY copy from agent 2 (W[2,1] = 0)

If they both copy from agent 2's CURRENT data, they stay together.
But agent 2 might MUTATE!

Let me recalculate more carefully...

For agents 1 and 2 to stay together:
- Agent 2 keeps its data (no mutation): prob ≈ 0.999
- Agent 1 copies from agent 2: prob ≈ 0.01 × 0.999 ≈ 0.00999
  OR agent 1 keeps its own data (which is same as agent 2's): prob ≈ 0.99

Actually, since they already share data D_A:
- If agent 2 keeps D_A: prob 0.999
- If agent 1 keeps D_A or copies from agent 2 (who has D_A): prob ≈ 0.99 + 0.00999 ≈ 0.99999

But CRITICAL: if agent 2 MUTATES (prob 0.001), they split!

So prob(stay together) ≈ 0.999 × 0.99999 ≈ 0.998

And agent 3 stays different: prob ≈ 0.99

Total: P(State 2 → State 2) ≈ 0.998 × 0.99 ≈ 0.988

================================================================================
STATE 3: {{1,3},{2}} - LEFT and RIGHT share data, CENTER different
================================================================================

For State 3 → State 3, we need agents 1 and 3 to STAY in the same block,
and agent 2 to stay alone.

Agents 1 and 3 currently share data D_A.
Agent 2 has different data D_B.

For agents 1 and 3 to stay together, they need to keep D_A or copy from
each other (but they can't - W[1,3] = W[3,1] = 0!).

So they can only KEEP their own data:
- Agent 1 keeps D_A: W[1,1] × (1-μ₁) = 0.99
- Agent 3 keeps D_A: W[3,3] × (1-μ₃) = 0.99

If either copies from agent 2, they get agent 2's data and split from the other.

Actually, let me think about this differently...

For State 3 to persist:
- Agents 1 and 3 must have the SAME data
- Agent 2 must have DIFFERENT data

Currently, agents 1 and 3 have D_A, agent 2 has D_B.

Agent 1 can:
- Keep D_A: prob 0.99
- Copy from agent 2 (get D_B): prob 0.01 × (1-μ₂) ≈ 0.00999
- Mutate from agent 2 (get new data): prob 0.01 × μ₂ ≈ 0.00001

Agent 3 can:
- Keep D_A: prob 0.99
- Copy from agent 2 (get D_B): prob 0.01 × (1-μ₂) ≈ 0.00999
- Mutate from agent 2 (get new data): prob 0.01 × μ₂ ≈ 0.00001

For agents 1 and 3 to STAY TOGETHER:
- Both keep D_A: 0.99 × 0.99 = 0.9801
- Both copy from agent 2 (get same D_B): 0.00999 × 0.00999 ≈ 0.0001
- Both mutate from agent 2 (get DIFFERENT new data): 0  (each gets unique mutation)

So prob(1 and 3 stay together) ≈ 0.9801 + 0.0001 ≈ 0.9802

Agent 2 can do anything (it just needs to be different from D_A or the new shared data):
- This is almost always satisfied

Actually, agent 2 being different is automatic if agents 1,3 keep D_A and agent 2
keeps D_B or mutates.

So: P(State 3 → State 3) ≈ 0.9802

================================================================================
COMPARISON
================================================================================

P(State 2 → State 2) ≈ 0.988
P(State 3 → State 3) ≈ 0.9802

State 2 has slightly HIGHER self-transition probability!

But let's think about the RATE OF ESCAPE:

Escape rate from State 2: 1 - 0.988 = 0.012
Escape rate from State 3: 1 - 0.9802 = 0.0198

State 3 has a HIGHER escape rate (less stable).

Hmm, this doesn't immediately explain why π₃ > π₂...

Wait, I need to think about this more systematically. Let me reconsider...

Actually, the key might be in the OTHER transitions!

State 1 → State 3 vs State 1 → State 2
State 4 → State 3 vs State 4 → State 2

Let me calculate these...

================================================================================
STATE 1 → STATE 2 or STATE 3
================================================================================

State 1: {{1,2,3}} - all three share data D

For State 1 → State 2 {{1,2},{3}}:
- Agents 1 and 2 stay together (keep D or copy from each other)
- Agent 3 gets different data

Since all start with D:
- Agent 1 keeps D or copies from anyone with D: high prob
- Agent 2 keeps D (W[2,2]=1, μ₂ small): prob ≈ 0.999
- Agent 3 must get different data:
  - Mutate from agent 2: prob ≈ 0.01 × 0.001 ≈ 0.00001

Very rare!

For State 1 → State 3 {{1,3},{2}}:
- Agents 1 and 3 stay together (keep D)
- Agent 2 gets different data (mutates)

- Agent 1 keeps D: prob ≈ 0.99
- Agent 3 keeps D: prob ≈ 0.99
- Agent 2 mutates: prob ≈ 0.001

Combined: 0.99 × 0.99 × 0.001 ≈ 0.00098

P(State 1 → State 3) ≈ 100× P(State 1 → State 2)!

THIS IS THE KEY!

From State 1 (all same), it's much easier to transition to State 3 than State 2,
because agent 2 (center) is the only one that mutates!

================================================================================
FINAL INSIGHT
================================================================================

The key is that with center-only innovation and outward flow:

1. State 1 (all same) preferentially flows to State 3 (center different)
   because the center is the only one that mutates.

2. State 3 is relatively stable (agents 1,3 keep old data, center keeps changing)

3. State 2 requires the center and one peripheral agent to share data,
   but the center keeps changing, making this rare.

This explains why π₃ > π₂ in the stationary distribution!
"""

print(__doc__)
