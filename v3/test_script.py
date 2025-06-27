#!/usr/bin/env python3
"""
Comprehensive Test Script for sympy-hpx v3 - Multiple Equations
Tests various multi-equation scenarios and edge cases.
"""

from sympy import *
from sympy_codegen import genFunc
import numpy as np

def test_basic_multi_equation():
    """Test basic multi-equation functionality as specified in requirements."""
    print("=== Test 1: Basic Multi-Equation ===")
    
    # Create symbols
    i = Idx("i")
    a = IndexedBase("a")
    b = IndexedBase("b")
    c = IndexedBase("c")
    r = IndexedBase("r")
    r2 = IndexedBase("r2")
    d = Symbol("d")
    
    # Create equations
    equations = [
        Eq(r[i], d*a[i] + b[i+1]*c[i-2]),
        Eq(r2[i], r[i] + a[i]**2)
    ]
    
    print(f"Equations: {equations}")
    
    # Generate function
    func = genFunc(equations)
    
    # Test data
    size = 8
    va = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    vb = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    vc = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    vr = np.zeros(size)
    vr2 = np.zeros(size)
    d_val = 2.0
    
    # Call function
    func(vr, vr2, va, vb, vc, d_val)
    
    # Verify results
    success = True
    for i in range(2, 7):  # Valid stencil range
        expected_r = d_val * va[i] + vb[i+1] * vc[i-2]
        expected_r2 = expected_r + va[i]**2
        
        if abs(vr[i] - expected_r) > 1e-10 or abs(vr2[i] - expected_r2) > 1e-10:
            success = False
            break
    
    print(f"Result: {'✓ PASS' if success else '✗ FAIL'}")
    return success

def test_simple_multi_equation():
    """Test simple multi-equation without stencils."""
    print("\n=== Test 2: Simple Multi-Equation (No Stencils) ===")
    
    # Create symbols
    i = Idx("i")
    a = IndexedBase("a")
    b = IndexedBase("b")
    r1 = IndexedBase("r1")
    r2 = IndexedBase("r2")
    x = Symbol("x")
    y = Symbol("y")
    
    # Simple equations without stencils
    equations = [
        Eq(r1[i], x*a[i] + y*b[i]),
        Eq(r2[i], r1[i] * a[i])
    ]
    
    print(f"Equations: {equations}")
    
    # Generate function
    func = genFunc(equations)
    
    # Test data
    size = 5
    va = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    vb = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    vr1 = np.zeros(size)
    vr2 = np.zeros(size)
    x_val = 3.0
    y_val = 2.0
    
    # Call function
    func(vr1, vr2, va, vb, x_val, y_val)
    
    # Verify results
    success = True
    for i in range(size):
        expected_r1 = x_val * va[i] + y_val * vb[i]
        expected_r2 = expected_r1 * va[i]
        
        if abs(vr1[i] - expected_r1) > 1e-10 or abs(vr2[i] - expected_r2) > 1e-10:
            success = False
            print(f"Mismatch at i={i}: r1 expected={expected_r1}, got={vr1[i]}, r2 expected={expected_r2}, got={vr2[i]}")
            break
    
    print(f"Result: {'✓ PASS' if success else '✗ FAIL'}")
    return success

def test_three_equations():
    """Test with three equations."""
    print("\n=== Test 3: Three Equations ===")
    
    # Create symbols
    i = Idx("i")
    a = IndexedBase("a")
    b = IndexedBase("b")
    r1 = IndexedBase("r1")
    r2 = IndexedBase("r2")
    r3 = IndexedBase("r3")
    k = Symbol("k")
    
    # Three equations
    equations = [
        Eq(r1[i], k * a[i]),
        Eq(r2[i], r1[i] + b[i]),
        Eq(r3[i], r1[i] * r2[i])
    ]
    
    print(f"Equations: {equations}")
    
    # Generate function
    func = genFunc(equations)
    
    # Test data
    size = 4
    va = np.array([1.0, 2.0, 3.0, 4.0])
    vb = np.array([0.1, 0.2, 0.3, 0.4])
    vr1 = np.zeros(size)
    vr2 = np.zeros(size)
    vr3 = np.zeros(size)
    k_val = 5.0
    
    # Call function
    func(vr1, vr2, vr3, va, vb, k_val)
    
    # Verify results
    success = True
    for i in range(size):
        expected_r1 = k_val * va[i]
        expected_r2 = expected_r1 + vb[i]
        expected_r3 = expected_r1 * expected_r2
        
        if (abs(vr1[i] - expected_r1) > 1e-10 or 
            abs(vr2[i] - expected_r2) > 1e-10 or 
            abs(vr3[i] - expected_r3) > 1e-10):
            success = False
            break
    
    print(f"Result: {'✓ PASS' if success else '✗ FAIL'}")
    return success

def test_mixed_stencil_patterns():
    """Test equations with different stencil patterns."""
    print("\n=== Test 4: Mixed Stencil Patterns ===")
    
    # Create symbols
    i = Idx("i")
    a = IndexedBase("a")
    b = IndexedBase("b")
    c = IndexedBase("c")
    r1 = IndexedBase("r1")
    r2 = IndexedBase("r2")
    
    # Mixed stencil equations
    equations = [
        Eq(r1[i], a[i-1] + a[i] + a[i+1]),  # 3-point stencil
        Eq(r2[i], r1[i] + b[i+2])           # Uses r1 result + forward stencil
    ]
    
    print(f"Equations: {equations}")
    
    # Generate function
    func = genFunc(equations)
    
    # Test data
    size = 8
    va = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    vb = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    vr1 = np.zeros(size)
    vr2 = np.zeros(size)
    
    # Call function
    func(vr1, vr2, va, vb)
    
    # Verify results for valid range (considering all stencil patterns)
    # Need: i-1 >= 0, i+1 < n, i+2 < n
    # So valid range is i from 1 to n-3 (inclusive)
    success = True
    for i in range(1, 6):  # Valid range: 1 to 5
        expected_r1 = va[i-1] + va[i] + va[i+1]
        expected_r2 = expected_r1 + vb[i+2]
        
        if abs(vr1[i] - expected_r1) > 1e-10 or abs(vr2[i] - expected_r2) > 1e-10:
            success = False
            print(f"Mismatch at i={i}: r1 expected={expected_r1}, got={vr1[i]}, r2 expected={expected_r2}, got={vr2[i]}")
            break
    
    print(f"Result: {'✓ PASS' if success else '✗ FAIL'}")
    return success

def test_single_equation_compatibility():
    """Test that single equations still work (backward compatibility)."""
    print("\n=== Test 5: Single Equation Compatibility ===")
    
    # Create symbols
    i = Idx("i")
    a = IndexedBase("a")
    b = IndexedBase("b")
    r = IndexedBase("r")
    k = Symbol("k")
    
    # Single equation
    equation = Eq(r[i], k * a[i] + b[i])
    
    print(f"Single equation: {equation}")
    
    # Generate function (should accept single equation)
    func = genFunc(equation)
    
    # Test data
    size = 5
    va = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    vb = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    vr = np.zeros(size)
    k_val = 2.0
    
    # Call function
    func(vr, va, vb, k_val)
    
    # Verify results
    success = True
    for i in range(size):
        expected = k_val * va[i] + vb[i]
        if abs(vr[i] - expected) > 1e-10:
            success = False
            break
    
    print(f"Result: {'✓ PASS' if success else '✗ FAIL'}")
    return success

def run_all_tests():
    """Run all tests and report results."""
    print("Running sympy-hpx v3 Multi-Equation Tests...\n")
    
    tests = [
        test_basic_multi_equation,
        test_simple_multi_equation,
        test_three_equations,
        test_mixed_stencil_patterns,
        test_single_equation_compatibility
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ FAIL - Exception: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n{'='*50}")
    print(f"sympy-hpx v3 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Multi-equation functionality is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    run_all_tests() 