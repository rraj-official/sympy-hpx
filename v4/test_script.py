#!/usr/bin/env python3
"""
Comprehensive Test Script for sympy-hpx v4 - Multi-Dimensional Support
Tests 1D, 2D, and 3D array operations with various stencil patterns.
"""

from sympy import *
from sympy_codegen import genFunc
import numpy as np

def test_1d_backward_compatibility():
    """Test that 1D equations still work (backward compatibility with v3)."""
    print("=== Test 1: 1D Backward Compatibility ===")
    
    # 1D symbols
    i = Idx("i")
    a = IndexedBase("a")
    b = IndexedBase("b")
    r = IndexedBase("r")
    k = Symbol("k")
    
    # Simple 1D equation
    equation = Eq(r[i], k * a[i] + b[i])
    
    print(f"1D equation: {equation}")
    
    # Generate function
    func = genFunc(equation)
    
    # Test data
    size = 5
    va = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    vb = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    vr = np.zeros(size)
    k_val = 3.0
    
    # Call function
    func(vr, va, vb, k_val)
    
    # Verify results
    expected = k_val * va + vb
    success = np.allclose(vr, expected)
    
    print(f"Input a: {va}")
    print(f"Input b: {vb}")
    print(f"k = {k_val}")
    print(f"Result: {vr}")
    print(f"Expected: {expected}")
    print(f"Result: {'✓ PASS' if success else '✗ FAIL'}")
    return success

def test_2d_simple():
    """Test simple 2D operations without stencils."""
    print("\n=== Test 2: 2D Simple Operations ===")
    
    # 2D symbols
    i, j = symbols('i j', integer=True)
    a = IndexedBase("a")
    b = IndexedBase("b")
    r = IndexedBase("r")
    k = Symbol("k")
    
    # Simple 2D equation: r[i,j] = k * a[i,j] + b[i,j]
    equation = Eq(r[i,j], k * a[i,j] + b[i,j])
    
    print(f"2D equation: {equation}")
    
    # Generate function
    func = genFunc(equation)
    
    # Test data
    rows, cols = 3, 4
    size = rows * cols
    va = np.arange(1.0, size + 1)  # [1, 2, 3, ..., 12]
    vb = np.ones(size) * 0.5       # [0.5, 0.5, ..., 0.5]
    vr = np.zeros(size)
    k_val = 2.0
    
    # Call function with shape parameters
    func(vr, va, vb, rows, cols, k_val)
    
    # Verify results
    expected = k_val * va + vb
    success = np.allclose(vr, expected)
    
    print(f"Grid size: {rows} x {cols}")
    print(f"Input a (flattened): {va}")
    print(f"Input b (flattened): {vb}")
    print(f"k = {k_val}")
    print(f"Result: {vr}")
    print(f"Expected: {expected}")
    print(f"Result: {'✓ PASS' if success else '✗ FAIL'}")
    return success

def test_2d_stencil():
    """Test 2D stencil operations."""
    print("\n=== Test 3: 2D Stencil Operations ===")
    
    # 2D symbols
    i, j = symbols('i j', integer=True)
    u = IndexedBase("u")
    laplacian = IndexedBase("laplacian")
    
    # 2D Laplacian with 5-point stencil
    # laplacian[i,j] = u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]
    equation = Eq(laplacian[i,j], u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j])
    
    print(f"2D Laplacian equation: {equation}")
    
    # Generate function
    func = genFunc(equation)
    
    # Test data - create a simple 2D function
    rows, cols = 5, 5
    size = rows * cols
    vu = np.zeros(size)
    vlaplacian = np.zeros(size)
    
    # Initialize u with a quadratic function: u(x,y) = x^2 + y^2
    for i_idx in range(rows):
        for j_idx in range(cols):
            idx = i_idx * cols + j_idx
            x, y = float(i_idx), float(j_idx)
            vu[idx] = x*x + y*y
    
    # Call function
    func(vlaplacian, vu, rows, cols)
    
    # For the function u(x,y) = x^2 + y^2, the Laplacian should be 4 everywhere
    # Check interior points (stencil operates on interior only)
    success = True
    for i_idx in range(1, rows-1):
        for j_idx in range(1, cols-1):
            idx = i_idx * cols + j_idx
            expected_val = 4.0  # Laplacian of x^2 + y^2 is 2 + 2 = 4
            if abs(vlaplacian[idx] - expected_val) > 1e-10:
                success = False
                print(f"Mismatch at ({i_idx},{j_idx}): expected {expected_val}, got {vlaplacian[idx]}")
                break
        if not success:
            break
    
    print(f"Grid size: {rows} x {cols}")
    print(f"Test function: u(x,y) = x² + y²")
    print(f"Expected Laplacian: 4.0 (interior points)")
    print(f"Sample interior result: {vlaplacian[1*cols + 1]:.2f}")
    print(f"Sample boundary result: {vlaplacian[0]:.2f} (should be 0)")
    print(f"Result: {'✓ PASS' if success else '✗ FAIL'}")
    return success

def test_2d_multi_equation():
    """Test 2D multi-equation system."""
    print("\n=== Test 4: 2D Multi-Equation System ===")
    
    # 2D symbols
    i, j = symbols('i j', integer=True)
    u = IndexedBase("u")
    grad_x = IndexedBase("grad_x")
    grad_y = IndexedBase("grad_y")
    grad_mag = IndexedBase("grad_mag")
    dx = Symbol("dx")
    
    # Multi-equation gradient system
    equations = [
        Eq(grad_x[i,j], (u[i+1,j] - u[i-1,j]) / (2*dx)),           # Gradient in x
        Eq(grad_y[i,j], (u[i,j+1] - u[i,j-1]) / (2*dx)),           # Gradient in y
        Eq(grad_mag[i,j], (grad_x[i,j]**2 + grad_y[i,j]**2)**0.5)  # Magnitude
    ]
    
    print(f"Multi-equation gradient system:")
    for k, eq in enumerate(equations):
        print(f"  {k+1}: {eq}")
    
    # Generate function
    func = genFunc(equations)
    
    # Test data
    rows, cols = 4, 4
    size = rows * cols
    vu = np.zeros(size)
    vgrad_x = np.zeros(size)
    vgrad_y = np.zeros(size)
    vgrad_mag = np.zeros(size)
    
    # Initialize u with a linear function: u(x,y) = 2*x + 3*y
    for i_idx in range(rows):
        for j_idx in range(cols):
            idx = i_idx * cols + j_idx
            x, y = float(i_idx), float(j_idx)
            vu[idx] = 2.0*x + 3.0*y
    
    dx_val = 1.0
    
    # Call function
    func(vgrad_x, vgrad_y, vgrad_mag, vu, rows, cols, dx_val)
    
    # For u(x,y) = 2*x + 3*y, grad_x should be 2, grad_y should be 3, grad_mag should be sqrt(13)
    expected_grad_x = 2.0
    expected_grad_y = 3.0
    expected_grad_mag = np.sqrt(13.0)
    
    # Check interior points
    success = True
    for i_idx in range(1, rows-1):
        for j_idx in range(1, cols-1):
            idx = i_idx * cols + j_idx
            if (abs(vgrad_x[idx] - expected_grad_x) > 1e-10 or 
                abs(vgrad_y[idx] - expected_grad_y) > 1e-10 or
                abs(vgrad_mag[idx] - expected_grad_mag) > 1e-10):
                success = False
                break
        if not success:
            break
    
    print(f"Grid size: {rows} x {cols}")
    print(f"Test function: u(x,y) = 2x + 3y")
    print(f"Expected grad_x: {expected_grad_x}")
    print(f"Expected grad_y: {expected_grad_y}")
    print(f"Expected grad_mag: {expected_grad_mag:.3f}")
    
    center_idx = (rows//2) * cols + (cols//2)
    print(f"Sample results at center:")
    print(f"  grad_x: {vgrad_x[center_idx]:.3f}")
    print(f"  grad_y: {vgrad_y[center_idx]:.3f}")
    print(f"  grad_mag: {vgrad_mag[center_idx]:.3f}")
    
    print(f"Result: {'✓ PASS' if success else '✗ FAIL'}")
    return success

def test_3d_simple():
    """Test simple 3D operations."""
    print("\n=== Test 5: 3D Simple Operations ===")
    
    # 3D symbols
    i, j, k = symbols('i j k', integer=True)
    a = IndexedBase("a")
    r = IndexedBase("r")
    c = Symbol("c")
    
    # Simple 3D equation: r[i,j,k] = c * a[i,j,k]
    equation = Eq(r[i,j,k], c * a[i,j,k])
    
    print(f"3D equation: {equation}")
    
    # Generate function
    func = genFunc(equation)
    
    # Test data
    rows, cols, depth = 2, 3, 2
    size = rows * cols * depth
    va = np.arange(1.0, size + 1)  # [1, 2, 3, ..., 12]
    vr = np.zeros(size)
    c_val = 5.0
    
    # Call function with 3D shape parameters
    func(vr, va, rows, cols, depth, c_val)
    
    # Verify results
    expected = c_val * va
    success = np.allclose(vr, expected)
    
    print(f"Grid size: {rows} x {cols} x {depth}")
    print(f"Input a (flattened): {va}")
    print(f"c = {c_val}")
    print(f"Result: {vr}")
    print(f"Expected: {expected}")
    print(f"Result: {'✓ PASS' if success else '✗ FAIL'}")
    return success

def test_mixed_dimensions():
    """Test mixing 1D and 2D arrays in the same equation."""
    print("\n=== Test 6: Mixed Dimensions ===")
    
    # Mixed symbols
    i, j = symbols('i j', integer=True)
    field_2d = IndexedBase("field_2d")  # 2D array
    coeff_1d = IndexedBase("coeff_1d")  # 1D array (varies only with i)
    result = IndexedBase("result")      # 2D result
    
    # Equation mixing 2D field with 1D coefficients
    # result[i,j] = coeff_1d[i] * field_2d[i,j]
    equation = Eq(result[i,j], coeff_1d[i] * field_2d[i,j])
    
    print(f"Mixed dimension equation: {equation}")
    
    # Generate function
    func = genFunc(equation)
    
    # Test data
    rows, cols = 3, 4
    field_size = rows * cols
    coeff_size = rows
    
    vfield_2d = np.ones(field_size) * 2.0  # All 2's
    vcoeff_1d = np.array([1.0, 3.0, 5.0])  # Coefficients for each row
    vresult = np.zeros(field_size)
    
    # Call function
    func(vresult, vcoeff_1d, vfield_2d, rows, cols)
    
    # Verify results
    success = True
    for i_idx in range(rows):
        for j_idx in range(cols):
            idx = i_idx * cols + j_idx
            expected_val = vcoeff_1d[i_idx] * vfield_2d[idx]
            if abs(vresult[idx] - expected_val) > 1e-10:
                success = False
                break
        if not success:
            break
    
    print(f"Grid size: {rows} x {cols}")
    print(f"2D field (all 2.0): shape {field_size}")
    print(f"1D coefficients: {vcoeff_1d}")
    print(f"Result (first row): {vresult[:cols]}")
    print(f"Result (second row): {vresult[cols:2*cols]}")
    print(f"Result: {'✓ PASS' if success else '✗ FAIL'}")
    return success

def run_all_tests():
    """Run all tests and report results."""
    print("Running sympy-hpx v4 Multi-Dimensional Tests...\n")
    
    tests = [
        test_1d_backward_compatibility,
        test_2d_simple,
        test_2d_stencil,
        test_2d_multi_equation,
        test_3d_simple,
        test_mixed_dimensions
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
    
    print(f"\n{'='*60}")
    print(f"sympy-hpx v4 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Multi-dimensional functionality is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    run_all_tests() 