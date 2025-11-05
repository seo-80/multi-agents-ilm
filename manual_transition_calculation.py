"""
Manual calculation of transition probabilities for M=3, outward flow, center-only
to identify the bug.

Parameters:
- M = 3
- m = 0.01
- alpha = 0.001 (only at center agent 2)
- N_i = 1

Agents (1-indexed):
- Agent 1: left endpoint
- Agent 2: center
- Agent 3: right endpoint

W matrix (1-indexed):
       j=1      j=2      j=3
i=1 [ 1-m       m        0   ]
i=2 [  0        1        0   ]
i=3 [  0        m       1-m  ]

with m = 0.01:
       j=1      j=2      j=3
i=1 [ 0.99     0.01      0   ]
i=2 [  0        1        0   ]
i=3 [  0       0.01     0.99 ]

Mutation rates (1-indexed):
- μ₁ = 0
- μ₂ = 0.001 / 1.001 ≈ 0.000999
- μ₃ = 0

================================================================================
STATE TRANSITIONS
================================================================================

State 5: {{1},{2},{3}} (all agents have different data)

We want to calculate:
1. P(State 5 → State 2) where State 2 = {{1,2},{3}}
2. P(State 5 → State 3) where State 3 = {{1,3},{2}}

--------------------------------------------------------------------------------
TRANSITION TO STATE 2: {{1,2},{3}}
--------------------------------------------------------------------------------

For agents 1 and 2 to share data, and agent 3 to have different data:
- Agents 1 and 2 must copy from the same source (without mutation)
- Agent 3 must NOT copy from that source (or mutate)

Possible scenarios:

Scenario A: Agents 1 and 2 both copy from agent 1's current data
- Agent 1 copies from agent 1: W[1,1] × (1-μ₁) = 0.99 × 1 = 0.99
- Agent 2 copies from agent 1: W[2,1] × (1-μ₁) = 0 × 1 = 0
- IMPOSSIBLE (W[2,1] = 0)

Scenario B: Agents 1 and 2 both copy from agent 2's current data
- Agent 1 copies from agent 2: W[1,2] × (1-μ₂) = 0.01 × 0.999001 ≈ 0.00999
- Agent 2 copies from agent 2: W[2,2] × (1-μ₂) = 1 × 0.999001 ≈ 0.999001
- Agent 3 must NOT copy from agent 2:
  - Agent 3 copies from agent 3: W[3,3] × (1-μ₃) = 0.99 × 1 = 0.99
  - OR Agent 3 mutates from agent 2: W[3,2] × μ₂ = 0.01 × 0.000999 ≈ 0.00001

  Prob(agent 3 NOT from agent 2's old data) = 0.99 + 0.00001 ≈ 0.99001

Combined probability: 0.00999 × 0.999001 × 0.99001 ≈ 0.00988

Scenario C: Agents 1 and 2 both copy from agent 3's current data
- Agent 1 copies from agent 3: W[1,3] × (1-μ₃) = 0 × 1 = 0
- IMPOSSIBLE (W[1,3] = 0)

Scenario D: Agents 1 and 2 both receive NEW mutation from agent 2
- Agent 1 mutates from agent 2: W[1,2] × μ₂ = 0.01 × 0.000999 ≈ 0.00001
- Agent 2 mutates from agent 2: W[2,2] × μ₂ = 1 × 0.000999 ≈ 0.000999
- But these are DIFFERENT mutations! Each mutation is unique.
- IMPOSSIBLE

Total P(State 5 → State 2) ≈ 0.00988

--------------------------------------------------------------------------------
TRANSITION TO STATE 3: {{1,3},{2}}
--------------------------------------------------------------------------------

For agents 1 and 3 to share data, and agent 2 to have different data:

Scenario A: Agents 1 and 3 both copy from agent 1's current data
- Agent 1 copies from agent 1: W[1,1] × (1-μ₁) = 0.99 × 1 = 0.99
- Agent 3 copies from agent 1: W[3,1] × (1-μ₁) = 0 × 1 = 0
- IMPOSSIBLE (W[3,1] = 0)

Scenario B: Agents 1 and 3 both copy from agent 2's current data
- Agent 1 copies from agent 2: W[1,2] × (1-μ₂) = 0.01 × 0.999001 ≈ 0.00999
- Agent 3 copies from agent 2: W[3,2] × (1-μ₂) = 0.01 × 0.999001 ≈ 0.00999
- Agent 2 must NOT keep its old data:
  - Agent 2 mutates: W[2,2] × μ₂ = 1 × 0.000999 ≈ 0.000999

Combined probability: 0.00999 × 0.00999 × 0.000999 ≈ 0.0000001

Scenario C: Agents 1 and 3 both copy from agent 3's current data
- Agent 1 copies from agent 3: W[1,3] × (1-μ₃) = 0 × 1 = 0
- IMPOSSIBLE (W[1,3] = 0)

Total P(State 5 → State 3) ≈ 0.0000001

================================================================================
CONCLUSION FROM TRANSITION PROBABILITIES
================================================================================

P(State 5 → State 2) ≈ 0.00988
P(State 5 → State 3) ≈ 0.0000001

State 2 is MUCH more likely than State 3!

This suggests π₂ > π₃, which means E[d_12] < E[d_13].

But the simulation shows E[d_12] > E[d_13]!

================================================================================
WAIT... LET ME RECONSIDER
================================================================================

Actually, I need to think about the ENTIRE transition matrix and steady state.

The question is: what is the steady-state probability of each state?

Let me reconsider the transitions more carefully...

Actually, in the outward flow model:
- Center agent NEVER copies from others (W[2,1] = W[2,3] = 0)
- Center only keeps its data or mutates
- Peripheral agents can only copy from center or themselves

This means:
- Center is constantly creating new mutations
- Periphery copies from center

In steady state:
- Center almost always has unique data (constantly mutating)
- When periphery copies from center, they get the SAME data
- But center immediately creates NEW data

So the typical sequence:
1. All different: {{1},{2},{3}}
2. Agents 1 and 3 copy from agent 2's current data D
3. Now: {{1,3},{2}} where 1,3 have D and 2 has new data D'
4. Agent 2 creates another new data D''
5. Back to all different or agents 1,3 still share old D

This analysis suggests State 3 {{1,3},{2}} should be MORE common!

Let me recalculate more carefully...

================================================================================
KEY INSIGHT
================================================================================

The problem might be in how I'm calculating transition probabilities.

When I calculated P(State 5 → State 3), I required agent 2 to MUTATE.
But that's VERY rare (μ₂ ≈ 0.001).

Actually, for State 3 {{1,3},{2}}, agent 2 just needs to have DIFFERENT data
from agents 1 and 3. If agent 2 keeps its OLD data (prob ≈ 0.999), and agents
1 and 3 copy from each other or from somewhere else, that's also State 3.

Wait, but agents 1 and 3 CAN'T copy from each other (W[1,3] = W[3,1] = 0).

Let me think about this more systematically...

================================================================================
SYSTEMATIC APPROACH: ALL TRANSITIONS FROM STATE 5
================================================================================

State 5: {{1},{2},{3}} - all have different data D₁, D₂, D₃

Next state depends on what each agent does:

Agent 1 can:
- Copy from 1: prob 0.99 → keeps D₁
- Copy from 2 without mutation: prob 0.01 × 0.999 ≈ 0.00999 → gets D₂
- Copy from 2 with mutation: prob 0.01 × 0.001 ≈ 0.00001 → gets NEW D₄

Agent 2 can:
- Keep data 2 without mutation: prob 1 × 0.999 ≈ 0.999 → keeps D₂
- Mutate: prob 1 × 0.001 ≈ 0.001 → gets NEW D₄

Agent 3 can:
- Copy from 3: prob 0.99 → keeps D₃
- Copy from 2 without mutation: prob 0.01 × 0.999 ≈ 0.00999 → gets D₂
- Copy from 2 with mutation: prob 0.01 × 0.001 ≈ 0.00001 → gets NEW D₅

Resulting state {{1,3},{2}}:
- Agent 1 and 3 both get D₂
- Agent 2 keeps D₂ (old) OR gets new D₄

Actually, if agent 2 keeps D₂, and agents 1,3 both get D₂, then all three
have D₂, which is State 1, not State 3!

For State 3, we need:
- Agents 1 and 3 share some data
- Agent 2 has different data

Hmm, this is getting complex. Let me reconsider the entire approach...
"""

print(__doc__)
