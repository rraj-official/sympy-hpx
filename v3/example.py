#!/usr/bin/env python3
"""
Example script for sympy-hpx v3 - Multiple Equations
Demonstrates the genFunc functionality with multiple equations processed together.
"""

from sympy import *
from sympy_codegen import genFunc
import numpy as np

print("=== sympy-hpx v3 Example: Multiple Equations ===")

# Create SymPy symbols
i = Idx("i")
a = IndexedBase("a")
b = IndexedBase("b")
c = IndexedBase("c")
r = IndexedBase("r")
r2 = IndexedBase("r2")
d = Symbol("d")  # scalar

# Create multiple equations as specified in requirements
equations = [
    Eq(r[i], d*a[i] + b[i+1]*c[i-2]),
    Eq(r2[i], r[i] + a[i]**2)
]

print(f"Multiple equations:")
for i, eq in enumerate(equations):
    print(f"  {i+1}: {eq}")

# Generate the function
print("\nGenerating C++ multi-equation function...")
a_bc = genFunc(equations)
print("Function generated successfully!")

# Create test vectors - need to be large enough for stencil access
size = 8
va = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
vb = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
vc = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
vr = np.zeros(size)
vr2 = np.zeros(size)
d_val = 2.0

print(f"\nInput data:")
print(f"d = {d_val}")
print(f"va = {va}")
print(f"vb = {vb}")
print(f"vc = {vc}")

# Call the generated multi-equation function
# Arguments: result vectors first (vr, vr2), then input vectors (va, vb, vc), then scalars (d)
print("\nCalling generated multi-equation function...")
a_bc(vr, vr2, va, vb, vc, d_val)

print(f"\nResults:")
print(f"vr  = {vr}")
print(f"vr2 = {vr2}")

# Verify the results manually
print(f"\nManual verification:")
print("Note: Multi-equation stencil operates on valid indices only (i-2 >= 0 and i+1 < n)")
print("Valid range: i from 2 to 6 (inclusive)")

success = True
for i in range(2, 7):  # Valid stencil range
    # First equation: r[i] = d*a[i] + b[i+1]*c[i-2]
    expected_r = d_val * va[i] + vb[i+1] * vc[i-2]
    actual_r = vr[i]
    
    # Second equation: r2[i] = r[i] + a[i]**2
    expected_r2 = expected_r + va[i]**2
    actual_r2 = vr2[i]
    
    print(f"i={i}:")
    print(f"  r[{i}]:  expected = {d_val}*{va[i]} + {vb[i+1]}*{vc[i-2]} = {expected_r:.2f}, actual = {actual_r:.2f}")
    print(f"  r2[{i}]: expected = {expected_r:.2f} + {va[i]}^2 = {expected_r2:.2f}, actual = {actual_r2:.2f}")
    
    if abs(actual_r - expected_r) < 1e-10 and abs(actual_r2 - expected_r2) < 1e-10:
        print(f"  ✓ CORRECT")
    else:
        print(f"  ✗ ERROR: r diff = {abs(actual_r - expected_r)}, r2 diff = {abs(actual_r2 - expected_r2)}")
        success = False

# Check that boundary indices are zero (not computed)
print(f"\nBoundary checks:")
boundary_indices = [0, 1, 7]
for idx in boundary_indices:
    if vr[idx] != 0.0 or vr2[idx] != 0.0:
        print(f"  ✗ ERROR: boundary i={idx} should be 0, got vr={vr[idx]}, vr2={vr2[idx]}")
        success = False
    else:
        print(f"  ✓ i={idx}: vr={vr[idx]}, vr2={vr2[idx]} (correct - boundary)")

if success:
    print(f"\n✓ Multi-equation operation completed successfully!")
else:
    print(f"\n✗ Multi-equation operation had errors!")

print(f"\nKey advantages of multi-equation processing:")
print(f"- Single loop for multiple related calculations")
print(f"- Efficient memory access patterns")
print(f"- Automatic dependency handling (r2 uses r)")
print(f"- Unified stencil bounds for all equations") 