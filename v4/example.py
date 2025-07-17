#!/usr/bin/env python3
"""
Example script for sympy-hpx v4 - Multi-Dimensional Support
Demonstrates 2D and 3D array operations with stencils.
"""

from sympy import *
from sympy_codegen import genFunc
import numpy as np

print("=== sympy-hpx v4 Example: Multi-Dimensional Arrays ===")

# Example 1: 2D Heat Diffusion
print("\n1. 2D Heat Diffusion Example")
print("   Computing heat diffusion on a 2D grid")

# Create SymPy symbols for 2D case
i, j = symbols('i j', integer=True)
T = IndexedBase("T")      # Temperature field (2D)
T_new = IndexedBase("T_new")  # New temperature (2D)
alpha = Symbol("alpha")   # Diffusion coefficient
dt = Symbol("dt")         # Time step
dx = Symbol("dx")         # Grid spacing

# 2D heat diffusion equation with 5-point stencil
# T_new[i,j] = T[i,j] + alpha*dt/dx^2 * (T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1] - 4*T[i,j])
heat_eq_2d = Eq(T_new[i,j], T[i,j] + alpha*dt/(dx**2) * (
    T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1] - 4*T[i,j]
))

print(f"   2D Heat equation: {heat_eq_2d}")

# Generate the 2D function
print("   Generating 2D heat diffusion function...")
heat_2d_func = genFunc(heat_eq_2d)
print("   Function generated successfully!")

# Create 2D test data
rows, cols = 6, 8
T_field = np.zeros(rows * cols)  # Flattened 2D array
T_new_field = np.zeros(rows * cols)

# Initialize with a hot spot in the center
center_i, center_j = rows // 2, cols // 2
for i in range(rows):
    for j in range(cols):
        idx = i * cols + j
        if abs(i - center_i) <= 1 and abs(j - center_j) <= 1:
            T_field[idx] = 100.0  # Hot spot
        else:
            T_field[idx] = 20.0   # Background temperature

# Parameters
alpha_val = 0.1
dt_val = 0.01
dx_val = 1.0

print(f"   Grid size: {rows} x {cols}")
print(f"   Parameters: α={alpha_val}, dt={dt_val}, dx={dx_val}")
print(f"   Initial hot spot at center: T=100°C, background: T=20°C")

# Call the 2D function
# Arguments: result_array, input_array, shape_params (rows, cols), scalar_params
heat_2d_func(T_new_field, T_field, rows, cols, alpha_val, dt_val, dx_val)

print(f"   Temperature after one time step:")
print(f"   Center temperature: {T_new_field[center_i * cols + center_j]:.2f}°C")
print(f"   Corner temperature: {T_new_field[0]:.2f}°C")

# Example 2: 2D Gradient Calculation
print(f"\n2. 2D Gradient Calculation")
print("   Computing gradient magnitude on a 2D field")

# Symbols for gradient calculation
u = IndexedBase("u")      # Input field (2D)
grad_x = IndexedBase("grad_x")  # Gradient in x direction
grad_y = IndexedBase("grad_y")  # Gradient in y direction
grad_mag = IndexedBase("grad_mag")  # Gradient magnitude

# Multi-equation system for 2D gradient
gradient_equations = [
    Eq(grad_x[i,j], (u[i+1,j] - u[i-1,j]) / (2*dx)),           # Central difference in x
    Eq(grad_y[i,j], (u[i,j+1] - u[i,j-1]) / (2*dx)),           # Central difference in y
    Eq(grad_mag[i,j], (grad_x[i,j]**2 + grad_y[i,j]**2)**0.5)  # Magnitude
]

print(f"   Gradient equations:")
for k, eq in enumerate(gradient_equations):
    print(f"     {k+1}: {eq}")

# Generate the gradient function
print("   Generating 2D gradient function...")
grad_2d_func = genFunc(gradient_equations)
print("   Function generated successfully!")

# Create test data - a simple 2D function
rows, cols = 5, 5
u_field = np.zeros(rows * cols)
grad_x_field = np.zeros(rows * cols)
grad_y_field = np.zeros(rows * cols)
grad_mag_field = np.zeros(rows * cols)

# Initialize u with a quadratic function: u(x,y) = x^2 + y^2
for i in range(rows):
    for j in range(cols):
        idx = i * cols + j
        x, y = i * 1.0, j * 1.0  # Grid coordinates
        u_field[idx] = x*x + y*y

dx_val = 1.0

print(f"   Test function: u(x,y) = x² + y²")
print(f"   Grid size: {rows} x {cols}")

# Call the gradient function
# Arguments: result_arrays (grad_x, grad_y, grad_mag), input_array (u), shape_params, scalar_params
grad_2d_func(grad_x_field, grad_y_field, grad_mag_field, u_field, rows, cols, dx_val)

# Check results at center point
center_idx = (rows//2) * cols + (cols//2)
print(f"   Results at center point:")
print(f"   Gradient x: {grad_x_field[center_idx]:.2f}")
print(f"   Gradient y: {grad_y_field[center_idx]:.2f}")
print(f"   Gradient magnitude: {grad_mag_field[center_idx]:.2f}")

# Example 3: 1D compatibility (backward compatibility with v3)
print(f"\n3. 1D Backward Compatibility Test")
print("   Verifying that 1D equations still work")

# 1D symbols
i = Idx("i")
a = IndexedBase("a")
b = IndexedBase("b")
r = IndexedBase("r")
k = Symbol("k")

# Simple 1D equation
eq_1d = Eq(r[i], k * a[i] + b[i])

print(f"   1D equation: {eq_1d}")

# Generate 1D function
func_1d = genFunc(eq_1d)

# Test 1D data
size = 5
a_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
b_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
r_vals = np.zeros(size)
k_val = 2.0

# Call 1D function
# For v4, 1D functions expect the size parameter *last*
func_1d(r_vals, a_vals, b_vals, k_val, size)

print(f"   Input a: {a_vals}")
print(f"   Input b: {b_vals}")
print(f"   k = {k_val}")
print(f"   Result r: {r_vals}")
print(f"   Expected: {k_val * a_vals + b_vals}")

# Verify results
expected = k_val * a_vals + b_vals
if np.allclose(r_vals, expected):
    print("   ✓ 1D compatibility test PASSED")
else:
    print("   ✗ 1D compatibility test FAILED")

print(f"\n✓ Multi-dimensional examples completed successfully!")
print(f"\nKey features demonstrated:")
print(f"  • 2D heat diffusion with 5-point stencil")
print(f"  • Multi-equation 2D gradient calculation")
print(f"  • Automatic flattened array indexing")
print(f"  • Backward compatibility with 1D equations")
print(f"  • Unified stencil bounds for multi-dimensional operations") 