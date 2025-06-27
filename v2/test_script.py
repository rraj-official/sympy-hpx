#!/usr/bin/env python3
"""
Comprehensive test script for sympy-hpx v2 - Stencil Operations
Tests various stencil patterns and edge cases.
"""

from sympy import *
import numpy as np
from sympy_codegen import genFunc

def test_basic_stencil():
    """Test the basic stencil pattern from requirements."""
    print("=== Test 1: Basic Stencil Pattern ===")
    
    # Create SymPy symbols
    i = Idx("i")
    a = IndexedBase("a")
    b = IndexedBase("b")
    c = IndexedBase("c")
    r = IndexedBase("r")
    d = Symbol("d")
    
    # Create the stencil equation: r[i] = d*a[i] + b[i+1]*c[i-2]
    equation = Eq(r[i], d*a[i] + b[i+1]*c[i-2])
    print(f"Equation: {equation}")
    
    # Generate the function
    a_bc = genFunc(equation)
    
    # Create test data
    size = 8
    va = np.arange(1, size+1, dtype=float)  # [1, 2, 3, 4, 5, 6, 7, 8]
    vb = np.arange(0.1, size*0.1+0.1, 0.1)  # [0.1, 0.2, ..., 0.8]
    vc = np.arange(10, size*10+10, 10)      # [10, 20, ..., 80]
    vr = np.zeros(size)
    d_val = 3.0
    
    print(f"Input: d={d_val}, va={va}, vb={vb}, vc={vc}")
    
    # Call the function
    a_bc(vr, va, vb, vc, d_val)
    
    print(f"Result: vr={vr}")
    
    # Verify results for valid stencil range (i=2 to i=6)
    success = True
    for i in range(2, 7):  # Valid range: 2 <= i <= 6
        expected = d_val * va[i] + vb[i+1] * vc[i-2]
        actual = vr[i]
        if abs(actual - expected) > 1e-10:
            print(f"  ✗ ERROR at i={i}: expected {expected}, got {actual}")
            success = False
        else:
            print(f"  ✓ i={i}: {actual} (correct)")
    
    # Check boundaries are zero
    for i in [0, 1, 7]:
        if vr[i] != 0.0:
            print(f"  ✗ ERROR: boundary i={i} should be 0, got {vr[i]}")
            success = False
    
    if success:
        print("✓ Basic stencil test PASSED")
    else:
        print("✗ Basic stencil test FAILED")
    
    return success

def test_simple_offset():
    """Test simple forward offset pattern."""
    print("\n=== Test 2: Simple Forward Offset ===")
    
    i = Idx("i")
    a = IndexedBase("a")
    r = IndexedBase("r")
    
    # r[i] = a[i+1]
    equation = Eq(r[i], a[i+1])
    print(f"Equation: {equation}")
    
    func = genFunc(equation)
    
    size = 5
    va = np.array([10, 20, 30, 40, 50], dtype=float)
    vr = np.zeros(size)
    
    func(vr, va)
    
    print(f"Input: va={va}")
    print(f"Result: vr={vr}")
    
    # Expected: vr[i] = va[i+1], valid for i=0 to i=3
    expected = np.array([20, 30, 40, 50, 0], dtype=float)
    
    if np.allclose(vr, expected):
        print("✓ Simple offset test PASSED")
        return True
    else:
        print(f"✗ Simple offset test FAILED: expected {expected}")
        return False

def test_backward_offset():
    """Test simple backward offset pattern."""
    print("\n=== Test 3: Simple Backward Offset ===")
    
    i = Idx("i")
    a = IndexedBase("a")
    r = IndexedBase("r")
    
    # r[i] = a[i-1]
    equation = Eq(r[i], a[i-1])
    print(f"Equation: {equation}")
    
    func = genFunc(equation)
    
    size = 5
    va = np.array([10, 20, 30, 40, 50], dtype=float)
    vr = np.zeros(size)
    
    func(vr, va)
    
    print(f"Input: va={va}")
    print(f"Result: vr={vr}")
    
    # Expected: vr[i] = va[i-1], valid for i=1 to i=4
    expected = np.array([0, 10, 20, 30, 40], dtype=float)
    
    if np.allclose(vr, expected):
        print("✓ Backward offset test PASSED")
        return True
    else:
        print(f"✗ Backward offset test FAILED: expected {expected}")
        return False

def test_symmetric_stencil():
    """Test symmetric stencil pattern."""
    print("\n=== Test 4: Symmetric Stencil ===")
    
    i = Idx("i")
    a = IndexedBase("a")
    r = IndexedBase("r")
    
    # r[i] = a[i-1] + a[i] + a[i+1] (3-point average)
    equation = Eq(r[i], a[i-1] + a[i] + a[i+1])
    print(f"Equation: {equation}")
    
    func = genFunc(equation)
    
    size = 6
    va = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    vr = np.zeros(size)
    
    func(vr, va)
    
    print(f"Input: va={va}")
    print(f"Result: vr={vr}")
    
    # Expected: vr[i] = va[i-1] + va[i] + va[i+1], valid for i=1 to i=4
    expected = np.zeros(size)
    for i in range(1, 5):
        expected[i] = va[i-1] + va[i] + va[i+1]
    
    print(f"Expected: {expected}")
    
    if np.allclose(vr, expected):
        print("✓ Symmetric stencil test PASSED")
        return True
    else:
        print("✗ Symmetric stencil test FAILED")
        return False

def test_no_stencil():
    """Test pattern with no stencil (regular indexing)."""
    print("\n=== Test 5: No Stencil (Regular) ===")
    
    i = Idx("i")
    a = IndexedBase("a")
    b = IndexedBase("b")
    r = IndexedBase("r")
    c = Symbol("c")
    
    # r[i] = c*a[i] + b[i] (same as v1)
    equation = Eq(r[i], c*a[i] + b[i])
    print(f"Equation: {equation}")
    
    func = genFunc(equation)
    
    size = 4
    va = np.array([1, 2, 3, 4], dtype=float)
    vb = np.array([0.5, 1.0, 1.5, 2.0], dtype=float)
    vr = np.zeros(size)
    c_val = 2.0
    
    func(vr, va, vb, c_val)
    
    print(f"Input: c={c_val}, va={va}, vb={vb}")
    print(f"Result: vr={vr}")
    
    # Expected: vr[i] = c*va[i] + vb[i] for all i
    expected = c_val * va + vb
    
    if np.allclose(vr, expected):
        print("✓ No stencil test PASSED")
        return True
    else:
        print(f"✗ No stencil test FAILED: expected {expected}")
        return False

if __name__ == "__main__":
    print("sympy-hpx v2 - Stencil Operations Test Suite")
    print("=" * 50)
    
    tests = [
        test_basic_stencil,
        test_simple_offset,
        test_backward_offset,
        test_symmetric_stencil,
        test_no_stencil
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n" + "=" * 50)
    print(f"Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests PASSED!")
    else:
        print(f"❌ {total - passed} tests FAILED") 