#!/usr/bin/env python3
"""
Example script that exactly matches the requirements format for sympy-hpx v1.
"""

from sympy import *
from sympy_codegen import genFunc
import numpy as np

# Exactly as specified in requirements
i = Idx("i")
a = IndexedBase("a")
b = IndexedBase("b")
c = IndexedBase("c")
r = IndexedBase("r")
d = Symbol("d")  # a scalar or constant

# Eq is the equality expression from sympy
a_bc = genFunc(Eq(r[i], d*a[i] + b[i]*c[i]))

# Create numpy arrays as specified
va = np.zeros(100)
vb = np.zeros(100)
vc = np.zeros(100)
vr = np.zeros(100)

# Fill with some test data
va.fill(1.0)
vb.fill(2.0)
vc.fill(3.0)
d_val = 4.0

# Call the generated function
a_bc(vr, va, vb, vc, d_val)

# Print results
print("Example from requirements:")
print(f"d = {d_val}")
print(f"First 10 elements of va: {va[:10]}")
print(f"First 10 elements of vb: {vb[:10]}")
print(f"First 10 elements of vc: {vc[:10]}")
print(f"First 10 elements of vr: {vr[:10]}")
print(f"Expected (d*a + b*c): {d_val * va[0] + vb[0] * vc[0]}")
print(f"All elements should be: {d_val * 1.0 + 2.0 * 3.0} = {d_val + 6.0}")

# Verify
expected_value = d_val * 1.0 + 2.0 * 3.0
if np.allclose(vr, expected_value):
    print("✓ SUCCESS: Generated function works correctly!")
else:
    print("✗ ERROR: Generated function produced incorrect results!") 