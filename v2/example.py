#!/usr/bin/env python3
"""
Example script for sympy-hpx v2 - Stencil Operations
Demonstrates the genFunc functionality with stencil patterns (offset indices).
"""

from sympy import *
from sympy_codegen import genFunc
import numpy as np

print("=== sympy-hpx v2 Example: Stencil Operations ===")

# Create SymPy symbols
i = Idx("i")
a = IndexedBase("a")
b = IndexedBase("b")
c = IndexedBase("c")
r = IndexedBase("r")
d = Symbol("d")  # scalar

# Create the stencil equation: r[i] = d*a[i] + b[i+1]*c[i-2]
equation = Eq(r[i], d*a[i] + b[i+1]*c[i-2])
print(f"Stencil equation: {equation}")

# Generate the function
print("Generating C++ stencil function...")
a_bc = genFunc(equation)
print("Function generated successfully!")

# Create test vectors - need to be large enough for stencil access
size = 10
va = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
vb = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
vc = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
vr = np.zeros(size)
d_val = 2.0

print(f"\nInput data:")
print(f"d = {d_val}")
print(f"va = {va}")
print(f"vb = {vb}")
print(f"vc = {vc}")

# Call the generated stencil function
print("\nCalling generated stencil function...")
a_bc(vr, va, vb, vc, d_val)

print(f"\nResults:")
print(f"vr = {vr}")

# Verify the result manually for a few indices
print(f"\nManual verification:")
print("Note: Stencil operates on valid indices only (i-2 >= 0 and i+1 < n)")
print("Valid range: i from 2 to 8 (inclusive)")

for i in range(2, 9):  # Valid stencil range
    expected = d_val * va[i] + vb[i+1] * vc[i-2]
    actual = vr[i]
    print(f"i={i}: expected = {d_val}*{va[i]} + {vb[i+1]}*{vc[i-2]} = {expected:.2f}, actual = {actual:.2f}")
    
    if abs(actual - expected) < 1e-10:
        print(f"  ✓ CORRECT")
    else:
        print(f"  ✗ ERROR: difference = {abs(actual - expected)}")

# Check that boundary indices are zero (not computed)
print(f"\nBoundary checks:")
print(f"vr[0] = {vr[0]} (should be 0 - boundary)")
print(f"vr[1] = {vr[1]} (should be 0 - boundary)")
print(f"vr[9] = {vr[9]} (should be 0 - boundary)")

print(f"\n✓ Stencil operation completed successfully!") 