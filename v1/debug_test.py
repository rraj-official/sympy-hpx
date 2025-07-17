#!/usr/bin/env python3

import os
import sys
sys.path.append('.')
from sympy_codegen import genFunc
import sympy as sp
import numpy as np

print('Testing v1 with debug info...')
print('Current directory:', os.getcwd())
print('Files before:', [f for f in os.listdir('.') if f.startswith('generated_')])

# Test basic functionality
i = sp.Symbol('i', integer=True)
a = sp.IndexedBase('a')
b = sp.IndexedBase('b')
r = sp.IndexedBase('r')
d = sp.Symbol('d')

equation = sp.Eq(r[i], d*a[i] + b[i])
print('Equation:', equation)

try:
    # Generate function
    func = genFunc(equation)
    print('Function generated successfully')
    
    # Check if files were created locally
    files_after = [f for f in os.listdir('.') if f.startswith('generated_')]
    print('Files after generation:', files_after)
    
    # Test the function
    va = np.array([1.0, 2.0, 3.0])
    vb = np.array([4.0, 5.0, 6.0])
    vr = np.zeros(3)
    d_val = 2.0
    
    func(vr, va, vb, d_val)
    print('Function result:', vr)
    print('Expected: [6.0, 9.0, 12.0] (d*a + b = 2*[1,2,3] + [4,5,6])')
    print('Test passed:', np.allclose(vr, [6.0, 9.0, 12.0]))
    
except Exception as e:
    print('Error occurred:', str(e))
    import traceback
    traceback.print_exc()
    
    # Check what files exist
    print('Files after error:', [f for f in os.listdir('.') if f.startswith('generated_')]) 