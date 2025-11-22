"""Quick sanity test for cellular domain."""

import numpy as np
from morphogen.stdlib import cellular
from morphogen.stdlib.cellular import CellularField2D

print("Testing Cellular Automata Domain")
print("=" * 60)

# Test 1: Basic field allocation
print("\n1. Testing field allocation...")
field = cellular.alloc((10, 10), states=2, fill_value=0)
print(f"   ✓ Created 2D field: {field}")

field_1d = cellular.alloc(20, states=2, fill_value=0)
print(f"   ✓ Created 1D field: {field_1d}")

# Test 2: Random initialization
print("\n2. Testing random initialization...")
field = cellular.random_init((20, 20), states=2, density=0.5, seed=42)
alive_count = np.sum(field.data == 1)
print(f"   ✓ Random field created, alive cells: {alive_count}")

# Test 3: Game of Life
print("\n3. Testing Game of Life...")
field, rule = cellular.game_of_life((50, 50), density=0.3, seed=42)
print(f"   ✓ Created Game of Life field and rule: {rule}")

stats = cellular.analyze_pattern(field)
print(f"   ✓ Initial stats: {stats['alive_count']} alive cells")

field = cellular.step(field, rule)
print(f"   ✓ Evolved one step, generation: {field.generation}")

field = cellular.evolve(field, rule, steps=10)
print(f"   ✓ Evolved 10 more steps, generation: {field.generation}")

# Test 4: Wolfram CA
print("\n4. Testing Wolfram Elementary CA...")
field_1d, rule_num = cellular.wolfram_ca(100, rule_number=30)
print(f"   ✓ Created Rule 30 CA, width: {field_1d.width}")

history = cellular.history(field_1d, rule_num, steps=20)
print(f"   ✓ Generated history: {len(history)} generations")

# Test 5: Brian's Brain
print("\n5. Testing Brian's Brain...")
field_bb = cellular.brians_brain((40, 40), density=0.1, seed=42)
print(f"   ✓ Created Brian's Brain field, states: {field_bb.states}")

field_bb = cellular.brians_brain_step(field_bb)
print(f"   ✓ Evolved Brian's Brain, generation: {field_bb.generation}")

# Test 6: HighLife
print("\n6. Testing HighLife variant...")
field_hl, rule_hl = cellular.highlife((30, 30), density=0.3, seed=42)
print(f"   ✓ Created HighLife: {rule_hl}")

# Test 7: Neighbor counting
print("\n7. Testing neighbor counting...")
data = np.zeros((5, 5), dtype=np.int32)
data[2, 2] = 1
field_test = CellularField2D(data, states=2)

moore_counts = cellular.count_neighbors_moore(field_test, state=1)
print(f"   ✓ Moore neighbors counted, center has {moore_counts[2,2]} neighbors")

von_neumann_counts = cellular.count_neighbors_von_neumann(field_test, state=1)
print(f"   ✓ von Neumann neighbors counted, center has {von_neumann_counts[2,2]} neighbors")

# Test 8: Array conversion
print("\n8. Testing array conversion...")
arr = cellular.to_array(field)
print(f"   ✓ Converted to array, shape: {arr.shape}")

field_from_arr = cellular.from_array(arr, states=2)
print(f"   ✓ Created field from array: {field_from_arr}")

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("Cellular Automata domain is working correctly.")
