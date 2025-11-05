"""
Test to understand the difference between naive_simulation and symbolic_analysis
for N_i=1 case.
"""

# Simulating the key difference:
# In naive_simulation with N_i=1, each agent's single data point
# is replaced every timestep using multinomial sampling.

# Hypothesis: The issue might be in how the multinomial sampling works.

# Example network matrix for agent i (bidirectional, m=0.1, agent is at position 1):
# network_matrix[1] = [0.05, 0.9, 0.05]
# This means:
# - 5% chance to copy from agent 0
# - 90% chance to copy from itself (agent 1)
# - 5% chance to copy from agent 2

# With multinomial(n=1, pvals=[0.05, 0.9, 0.05]):
# Result is one of: [1,0,0], [0,1,0], [0,0,1]
# This means exactly ONE source is chosen

# In symbolic_analysis:
# prob_copy_block(agent=1, block={1}, W, mu)
#   = W[1,1] * (1 - mu[1])
#   = 0.9 * (1 - mu_1)

# This seems to match...

# But wait! Let's think about what happens when the agent is in a block
# with OTHER agents.

print("="*80)
print("Testing Multinomial Sampling vs Symbolic Block Probability")
print("="*80)

# Case 1: Agent copies from a block containing multiple agents
print("\nCase: Agent 0 copies from block {1,2} in state S")
print("Assume W[0,1] = 0.05, W[0,2] = 0, W[0,0] = 0.95")
print("Assume mu_1 = mu_2 = 0.1")

print("\nSymbolic analysis:")
print("  prob_copy_block(0, {1,2}, W, mu)")
print("  = W[0,1]*(1-mu_1) + W[0,2]*(1-mu_2)")
print("  = 0.05*0.9 + 0*0.9")
print("  = 0.045")

print("\nNaive simulation:")
print("  multinomial(n=1, pvals=[0.95, 0.05, 0])")
print("  Chooses: agent 0 with prob 0.95, agent 1 with prob 0.05, agent 2 with prob 0")
print("  If agent 1 is chosen:")
print("    - Mutation check: mutate with prob mu_1 = 0.1")
print("    - If no mutation: copy from agent 1")
print("  Total prob of copying from block {1,2} without mutation:")
print("    = 0.05 * (1 - 0.1) + 0 * (1 - 0.1)")
print("    = 0.045")

print("\n✓ These match!")

print("\n" + "="*80)
print("The models SHOULD be equivalent...")
print("But why does E[d_04] > E[d_06] differ?")
print("="*80)

print("\nPossible issues to investigate:")
print("1. Are the distance calculations really different?")
print("2. Is there a subtle difference in how states are defined?")
print("3. Is the time averaging in simulation done correctly?")
print("4. Are the boundary conditions (endpoints) handled identically?")
print("5. Is there an issue with how 'new mutations' are treated?")

print("\nLet me check the mutation case more carefully...")

print("\n" + "="*80)
print("Mutation Case Analysis")
print("="*80)

print("\nWhen agent 0 receives a mutation:")

print("\nSymbolic analysis:")
print("  prob_receive_mutation_from(0, j, W, mu)")
print("  = W[0,j] * mu[j]")
print("  Total mutation prob = Σ_j W[0,j] * mu[j]")

print("\nNaive simulation:")
print("  Step 1: multinomial chooses source j with prob W[0,j]")
print("  Step 2: mutation occurs with prob mu[j]")
print("  Total mutation prob = Σ_j W[0,j] * mu[j]")

print("\n✓ These also match!")

print("\n" + "="*80)
print("CRITICAL QUESTION:")
print("="*80)
print("\nIn naive_simulation, when N_i=1:")
print("- Each agent's data is REPLACED every timestep")
print("- This means the agent loses its old data")
print()
print("In symbolic_analysis:")
print("- Does the transition happen the same way?")
print("- Or is there a different interpretation?")
print()
print("Need to verify: Are ALL agents updated SIMULTANEOUSLY in both models?")
