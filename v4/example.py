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

# Example 2: 2D Laplacian (Simple Stencil)
print(f"\n2. 2D Laplacian Operator")
print("   Computing discrete Laplacian on a 2D field")

# Symbols for Laplacian calculation
u = IndexedBase("u")      # Input field (2D)
laplacian = IndexedBase("laplacian")  # Laplacian result

# 2D Laplacian using 5-point stencil: ∇²u = (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]) / dx²
laplacian_eq_2d = Eq(laplacian[i,j], (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]) / (dx**2))

print(f"   Laplacian equation: {laplacian_eq_2d}")

# Generate the Laplacian function
print("   Generating 2D Laplacian function...")
laplacian_2d_func = genFunc(laplacian_eq_2d)
print("   Function generated successfully!")

# Create test data - a simple 2D Gaussian-like function
rows, cols = 6, 8
u_field = np.zeros(rows * cols)
laplacian_field = np.zeros(rows * cols)

# Initialize u with a smooth function: u(x,y) = exp(-(x-cx)²-(y-cy)²)
cx, cy = rows//2, cols//2
for i in range(rows):
    for j in range(cols):
        idx = i * cols + j
        x, y = i - cx, j - cy
        u_field[idx] = np.exp(-(x*x + y*y) * 0.1)

dx_val = 1.0

print(f"   Test function: u(x,y) = exp(-(x-cx)²-(y-cy)²)")
print(f"   Grid size: {rows} x {cols}")

# Call the Laplacian function
# Arguments: result_array, input_array, shape_params (rows, cols), scalar_params
laplacian_2d_func(laplacian_field, u_field, rows, cols, dx_val)

# Check results at center point
center_idx = cx * cols + cy
print(f"   Results at center point:")
print(f"   Original value: {u_field[center_idx]:.3f}")
print(f"   Laplacian: {laplacian_field[center_idx]:.3f}")

# Example 3: 3D Simple Operation
print(f"\n3. 3D Array Operation")
print("   Computing 3D weighted average")

# Create SymPy indices for 3D case (using Idx for proper array indexing)
i_3d = Idx("i")
j_3d = Idx("j") 
k_3d = Idx("k")
u3d = IndexedBase("u3d")      # 3D input field
result3d = IndexedBase("result3d")  # 3D result
weight = Symbol("weight")     # Scaling factor

# Simple 3D operation: result[i,j,k] = weight * u3d[i,j,k]
eq_3d = Eq(result3d[i_3d,j_3d,k_3d], weight * u3d[i_3d,j_3d,k_3d])

print(f"   3D equation: {eq_3d}")

# Generate the 3D function
print("   Generating 3D function...")
func_3d = genFunc(eq_3d)
print("   Function generated successfully!")

# Create 3D test data
rows, cols, depth = 4, 5, 3
u3d_field = np.zeros(rows * cols * depth)
result3d_field = np.zeros(rows * cols * depth)

# Initialize with some pattern
for i in range(rows):
    for j in range(cols):
        for k in range(depth):
            idx = i * cols * depth + j * depth + k
            u3d_field[idx] = float(i + j + k + 1)

weight_val = 2.5

print(f"   Grid size: {rows} x {cols} x {depth}")
print(f"   Weight: {weight_val}")

# Call the 3D function
# Arguments: result_array, input_array, shape_params (rows, cols, depth), scalar_params
func_3d(result3d_field, u3d_field, rows, cols, depth, weight_val)

# Check results at a sample point
sample_idx = 1 * cols * depth + 2 * depth + 1  # [1,2,1]
print(f"   Sample point [1,2,1]:")
print(f"   Input: {u3d_field[sample_idx]:.1f}")
print(f"   Output: {result3d_field[sample_idx]:.1f}")
print(f"   Expected: {weight_val * u3d_field[sample_idx]:.1f}")

# Example 4: 1D compatibility (backward compatibility with v3)
print(f"\n4. 1D Backward Compatibility Test")
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
print(f"  • 2D Laplacian operator with stencil patterns")
print(f"  • 3D array operations with flattened indexing")
print(f"  • Automatic flattened array indexing (row-major)")
print(f"  • Backward compatibility with 1D equations")
print(f"  • HPX parallel execution for multi-dimensional arrays") 