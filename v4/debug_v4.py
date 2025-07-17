#!/usr/bin/env python3

import os
import sys
sys.path.append('.')
from sympy_codegen import genFunc
import sympy as sp
import numpy as np

print('Debug v4 argument counting...')

# Test 1: Simple 1D case
i = sp.Symbol('i', integer=True)
a = sp.IndexedBase('a')
b = sp.IndexedBase('b')
r = sp.IndexedBase('r')
k = sp.Symbol('k')

equation = sp.Eq(r[i], k*a[i] + b[i])
print('1D equation:', equation)

# Let me check the analysis results
from sympy_codegen import MultiDimCodeGenerator
generator = MultiDimCodeGenerator()
vector_vars, scalar_vars, result_vars, stencil_info, array_dims = generator._analyze_equations([equation])

print(f'vector_vars: {vector_vars}')
print(f'scalar_vars: {scalar_vars}')
print(f'result_vars: {result_vars}')
print(f'array_dims: {array_dims}')

# Determine if multi-dimensional
is_multidim = any(dim > 1 for dim in array_dims.values())
max_dim = max(array_dims.values()) if array_dims else 1

print(f'is_multidim: {is_multidim}')
print(f'max_dim: {max_dim}')

# Calculate expected arguments
expected_args = len(result_vars) + len(vector_vars) + len(scalar_vars)
if is_multidim:
    expected_args += max_dim  # shape parameters
else:
    expected_args += 1  # size parameter for 1D

print(f'Expected arguments: {expected_args}')
print(f'Breakdown: {len(result_vars)} result + {len(vector_vars)} input + {len(scalar_vars)} scalar + {1 if not is_multidim else max_dim} size/shape')

# Test the function generation
func = genFunc(equation)
print('Function generated')

# Try with the expected number of arguments
va = np.array([1.0, 2.0, 3.0])
vb = np.array([4.0, 5.0, 6.0])
vr = np.zeros(3)
n = 3
k_val = 2.0

print(f'Calling with: result({1}), inputs({len(vector_vars)}), size({1}), scalars({len(scalar_vars)})')
try:
    func(vr, va, vb, n, k_val)
    print('Success!')
    print('Result:', vr)
except Exception as e:
    print('Error:', e)
    print('Let me try different argument orders...')
    
    # Try without size parameter
    try:
        func(vr, va, vb, k_val)
        print('Success without size parameter!')
        print('Result:', vr)
    except Exception as e2:
        print('Also failed without size:', e2) 