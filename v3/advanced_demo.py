#!/usr/bin/env python3
"""
Advanced Demo for sympy-hpx v3 - Complex Multi-Equation System
Demonstrates advanced multi-equation capabilities with realistic computational scenarios.
"""

from sympy import *
from sympy_codegen import genFunc
import numpy as np
import time
import matplotlib.pyplot as plt

print("=== sympy-hpx v3 Advanced Demo: Multi-Equation Systems ===\n")

# Example 1: Numerical differentiation system
print("1. Numerical Differentiation System")
print("   Computing gradient, Laplacian, and combined result")

i = Idx("i")
u = IndexedBase("u")  # Input field
grad = IndexedBase("grad")    # Gradient (first derivative)
lapl = IndexedBase("lapl")    # Laplacian (second derivative)  
result = IndexedBase("result") # Combined result
dx = Symbol("dx")  # Grid spacing

# Multi-equation system for numerical differentiation
diff_equations = [
    Eq(grad[i], (u[i+1] - u[i-1]) / (2*dx)),           # Central difference gradient
    Eq(lapl[i], (u[i-1] - 2*u[i] + u[i+1]) / (dx**2)), # Second derivative (Laplacian)
    Eq(result[i], grad[i] + 0.1*lapl[i])               # Combined: gradient + damped Laplacian
]

print(f"   Equations:")
for j, eq in enumerate(diff_equations):
    print(f"     {j+1}: {eq}")

# Generate function
diff_func = genFunc(diff_equations)

# Test with a sine wave
size = 20
x_vals = np.linspace(0, 2*np.pi, size)
u_field = np.sin(x_vals)  # Input: sine wave
dx_val = x_vals[1] - x_vals[0]  # Grid spacing

# Prepare output arrays
grad_result = np.zeros(size)
lapl_result = np.zeros(size)
combined_result = np.zeros(size)

print(f"   Input: sine wave with {size} points")
print(f"   Grid spacing dx = {dx_val:.4f}")

# Call multi-equation function
start_time = time.time()
diff_func(grad_result, lapl_result, combined_result, u_field, dx_val)
end_time = time.time()

print(f"   Computation time: {(end_time - start_time)*1000:.2f} ms")
print(f"   Valid computation range: indices 1 to {size-2}")
print(f"   Gradient at middle: {grad_result[size//2]:.4f}")
print(f"   Laplacian at middle: {lapl_result[size//2]:.4f}")
print(f"   Combined result at middle: {combined_result[size//2]:.4f}")

# Example 2: Multi-physics coupling
print(f"\n2. Multi-Physics Coupling System")
print("   Temperature diffusion with velocity coupling")

# Symbols for multi-physics
T = IndexedBase("T")      # Temperature
v = IndexedBase("v")      # Velocity  
T_new = IndexedBase("T_new")  # New temperature
v_new = IndexedBase("v_new")  # New velocity
flux = IndexedBase("flux")    # Heat flux
alpha = Symbol("alpha")   # Thermal diffusivity
beta = Symbol("beta")     # Coupling coefficient
dt = Symbol("dt")         # Time step

# Multi-physics equations
physics_equations = [
    Eq(flux[i], alpha * (T[i+1] - T[i-1]) / (2*dx)),           # Heat flux
    Eq(T_new[i], T[i] + dt * (flux[i+1] - flux[i-1]) / (2*dx)), # Temperature update
    Eq(v_new[i], v[i] + beta * dt * (T_new[i] - T[i]))          # Velocity coupling
]

print(f"   Equations:")
for j, eq in enumerate(physics_equations):
    print(f"     {j+1}: {eq}")

# Generate multi-physics function
physics_func = genFunc(physics_equations)

# Test data
size = 16
T_field = np.ones(size) * 300.0  # Initial temperature: 300K
T_field[size//2-2:size//2+2] = 350.0  # Hot spot in the middle
v_field = np.zeros(size)  # Initial velocity: zero

# Output arrays
flux_result = np.zeros(size)
T_new_result = np.zeros(size)
v_new_result = np.zeros(size)

# Parameters
alpha_val = 1e-4  # Thermal diffusivity
beta_val = 1e-3   # Coupling strength
dt_val = 0.1      # Time step
dx_val = 0.1      # Grid spacing

print(f"   Initial hot spot: T = 350K at center, T = 300K elsewhere")
print(f"   Parameters: α = {alpha_val}, β = {beta_val}, dt = {dt_val}")

# Run multi-physics simulation
start_time = time.time()
physics_func(flux_result, T_new_result, v_new_result, T_field, v_field, alpha_val, beta_val, dt_val, dx_val)
end_time = time.time()

print(f"   Computation time: {(end_time - start_time)*1000:.2f} ms")
print(f"   Max heat flux: {np.max(np.abs(flux_result)):.6f}")
print(f"   Temperature change: {np.max(T_new_result) - np.min(T_new_result):.2f} K")
print(f"   Max induced velocity: {np.max(np.abs(v_new_result)):.6f}")

# Example 3: Performance comparison
print(f"\n3. Performance Comparison")
print("   Multi-equation vs. separate function calls")

# Simple test case for timing
simple_equations = [
    Eq(result[i], 2.0*u[i] + grad[i]),
    Eq(grad[i], u[i+1] - u[i-1])
]

multi_func = genFunc(simple_equations)

# Large dataset for performance testing
large_size = 1000
large_u = np.random.random(large_size)
large_result = np.zeros(large_size)
large_grad = np.zeros(large_size)

print(f"   Testing with {large_size} elements")

# Time multi-equation approach
start_time = time.time()
for _ in range(100):  # 100 iterations
    multi_func(large_result, large_grad, large_u)
multi_time = time.time() - start_time

print(f"   Multi-equation approach: {multi_time*1000:.1f} ms (100 iterations)")
print(f"   Per iteration: {multi_time*10:.2f} ms")

print(f"\n✓ Advanced multi-equation demo completed successfully!")
print(f"\nKey advantages demonstrated:")
print(f"  • Unified processing of related equations")
print(f"  • Automatic stencil bounds management")
print(f"  • Efficient single-loop computation")
print(f"  • Support for complex multi-physics scenarios")
print(f"  • High performance with large datasets") 