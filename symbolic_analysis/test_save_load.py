"""
Quick test to verify save/load functionality works correctly.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from m_agent_stationary_symbolic import (
    compute_stationary_state_symbolic,
    load_results_by_case,
    partition_to_string
)
from sympy import simplify

print("="*80)
print("Testing Save/Load Functionality")
print("="*80)

# Run a simple computation (M=3, case1)
print("\n[1/3] Computing results for M=3, case1...")
states, pi, P, W = compute_stationary_state_symbolic(
    M=3,
    center_prestige=False,
    centralized_neologism_creation=False
)

print("\n[2/3] Loading saved results...")
loaded = load_results_by_case(3, "case1")

print("\n[3/3] Verifying loaded data matches computed data...")

# Check metadata
assert loaded['metadata']['M'] == 3
assert loaded['metadata']['case_name'] == "case1"
print("✓ Metadata correct")

# Check states
assert len(loaded['states']) == len(states)
assert set(loaded['states']) == set(states)
print(f"✓ States correct ({len(states)} states)")

# Check W matrix dimensions
assert loaded['W'].shape == W.shape
print(f"✓ W matrix dimensions correct ({W.shape})")

# Check P matrix dimensions
assert loaded['P'].shape == P.shape
print(f"✓ P matrix dimensions correct ({P.shape})")

# Check pi dimensions
assert len(loaded['pi']) == len(pi)
print(f"✓ Stationary distribution dimensions correct ({len(pi)} states)")

# Check symbolic expressions match (compare simplified forms)
print("\nVerifying symbolic expressions match...")
for i in range(len(pi)):
    diff = simplify(loaded['pi'][i] - pi[i])
    assert diff == 0, f"Mismatch in pi[{i}]"
print("✓ Stationary distribution expressions match")

print("\n" + "="*80)
print("Save/Load functionality test PASSED!")
print("="*80)
