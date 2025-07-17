#!/usr/bin/env python3
"""
Test script for sympy-hpx v1 - SymPy code generation
Demonstrates the genFunc functionality with the example from requirements.
"""

from sympy import *
import numpy as np
from sympy_codegen import genFunc

def test_basic_functionality():
    """Test the basic functionality as specified in requirements."""
    print("=== sympy-hpx v1 Test: Basic SymPy Code Generation ===")
    
    # Create SymPy symbols as in the requirements
    i = Idx("i")
    a = IndexedBase("a")
    b = IndexedBase("b") 
    c = IndexedBase("c")
    r = IndexedBase("r")
    d = Symbol("d")  # a scalar or constant
    
    # Create the equation: r[i] = d*a[i] + b[i]*c[i]
    equation = Eq(r[i], d*a[i] + b[i]*c[i])
    print(f"SymPy equation: {equation}")
    
    # Generate the function
    print("Generating C++ function...")
    a_bc = genFunc(equation)
    print("Function generated successfully!")
    
    # Create test vectors
    print("\nCreating test vectors...")
    va = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    vb = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    vc = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    vr = np.zeros(5)
    d_val = 2.0
    
    print(f"va = {va}")
    print(f"vb = {vb}")
    print(f"vc = {vc}")
    print(f"d = {d_val}")
    
    # Call the generated function
    print("\nCalling generated function...")
    a_bc(r=vr, a=va, b=vb, c=vc, d=d_val)
    
    print(f"Result vr = {vr}")
    
    # Verify the result manually
    expected = d_val * va + vb * vc
    print(f"Expected   = {expected}")
    
    # Check if results match
    if np.allclose(vr, expected):
        print("✓ Test PASSED: Results match expected values!")
    else:
        print("✗ Test FAILED: Results don't match expected values!")
        print(f"Difference: {vr - expected}")

def test_larger_vectors():
    """Test with larger vectors to verify performance."""
    print("\n=== Testing with larger vectors (100 elements) ===")
    
    # Create SymPy symbols
    i = Idx("i")
    a = IndexedBase("a")
    b = IndexedBase("b")
    c = IndexedBase("c") 
    r = IndexedBase("r")
    d = Symbol("d")
    
    # Create the equation
    equation = Eq(r[i], d*a[i] + b[i]*c[i])
    
    # Generate the function
    a_bc = genFunc(equation)
    
    # Create larger test vectors
    size = 100
    va = np.random.rand(size)
    vb = np.random.rand(size)
    vc = np.random.rand(size)
    vr = np.zeros(size)
    d_val = 3.14
    
    print(f"Testing with vectors of size {size}")
    
    # Call the generated function
    a_bc(r=vr, a=va, b=vb, c=vc, d=d_val)
    
    # Verify the result
    expected = d_val * va + vb * vc
    
    if np.allclose(vr, expected):
        print("✓ Large vector test PASSED!")
    else:
        print("✗ Large vector test FAILED!")
        max_diff = np.max(np.abs(vr - expected))
        print(f"Maximum difference: {max_diff}")

def test_different_expression():
    """Test with a different mathematical expression."""
    print("\n=== Testing with different expression ===")
    
    # Create SymPy symbols for: result[i] = a[i]^2 + b[i] - c
    i = Idx("i")
    a = IndexedBase("a")
    b = IndexedBase("b")
    result = IndexedBase("result")
    c = Symbol("c")  # scalar
    
    # Create equation: result[i] = a[i]^2 + b[i] - c
    equation = Eq(result[i], a[i]**2 + b[i] - c)
    print(f"SymPy equation: {equation}")
    
    # Generate the function
    func = genFunc(equation)
    
    # Create test data
    va = np.array([1.0, 2.0, 3.0, 4.0])
    vb = np.array([0.5, 1.0, 1.5, 2.0])
    vresult = np.zeros(4)
    c_val = 0.25
    
    print(f"va = {va}")
    print(f"vb = {vb}")
    print(f"c = {c_val}")
    
    # Call the function
    func(result=vresult, a=va, b=vb, c=c_val)
    
    print(f"Result = {vresult}")
    
    # Verify manually
    expected = va**2 + vb - c_val
    print(f"Expected = {expected}")
    
    if np.allclose(vresult, expected):
        print("✓ Different expression test PASSED!")
    else:
        print("✗ Different expression test FAILED!")

if __name__ == "__main__":
    try:
        test_basic_functionality()
        test_larger_vectors()
        test_different_expression()
        print("\n=== All tests completed ===")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc() 