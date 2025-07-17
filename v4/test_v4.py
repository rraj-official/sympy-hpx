#!/usr/bin/env python3

import os
import sys
sys.path.append('.')
from sympy_codegen import genFunc
import sympy as sp
import numpy as np

print('Testing v4 (multi-dimensional support)...')

# Test 1: Simple 1D case
try:
    print('\n=== Test 1: Simple 1D case ===')
    i = sp.Symbol('i', integer=True)
    a = sp.IndexedBase('a')
    b = sp.IndexedBase('b')
    r = sp.IndexedBase('r')
    k = sp.Symbol('k')
    
    equation = sp.Eq(r[i], k*a[i] + b[i])
    print('1D equation:', equation)
    
    # Generate function
    func = genFunc(equation)
    print('Function generated successfully')
    
    # Test 1D functionality - signature: result_arrays, input_arrays, size, scalars
    va = np.array([1.0, 2.0, 3.0])
    vb = np.array([4.0, 5.0, 6.0])
    vr = np.zeros(3)
    n = 3
    k_val = 2.0
    
    func(vr, va, vb, n, k_val)  # result, inputs, size, scalars
    print('1D result:', vr)
    print('Expected: [6.0, 9.0, 12.0]')
    print('1D test passed:', np.allclose(vr, [6.0, 9.0, 12.0]))
    
except Exception as e:
    print('Error in 1D test:', e)
    import traceback
    traceback.print_exc()

print('\n=== File cleanup test ===')
files_before_exit = [f for f in os.listdir('.') if f.startswith('generated_')]
print(f'Files currently in directory: {len(files_before_exit)}')
print('Files will be cleaned up when Python exits...')

print('\nAll tests completed!') 