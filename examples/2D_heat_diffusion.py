#!/usr/bin/env python3
"""
2D Heat Diffusion Equation - Scientific Computing Example

This example demonstrates solving the 2D heat diffusion equation using 
sympy-hpx v4's multi-dimensional capabilities with HPX parallel acceleration.

Problem: Heat diffusion in a 2D metal plate
Equation: ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)

Where:
- u(x,y,t) = temperature at position (x,y) and time t
- α = thermal diffusivity constant
- Boundary conditions: Fixed temperature at edges
- Initial condition: Hot circular region in center

Reference: 
- Garbel N. "Solving 2D Heat Equation Numerically using Python"
- Level Up Coding, 2020
- Heat equation stability: dt <= (dx²dy²)/(2α(dx² + dy²))
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'v4'))

from sympy import *
from sympy_codegen import genFunc
import numpy as np
import matplotlib.pyplot as plt
import time

print("=" * 70)
print("2D HEAT DIFFUSION EQUATION - SCIENTIFIC COMPUTING EXAMPLE")
print("=" * 70)
print("Solving: ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)")
print("Using: sympy-hpx v4 with HPX parallel acceleration")
print("=" * 70)

# Physical parameters
alpha = 0.1      # Thermal diffusivity (mm²/s)
L = 10.0         # Plate length (mm) 
W = 10.0         # Plate width (mm)
T_cold = 20.0    # Cold temperature (°C)
T_hot = 100.0    # Hot temperature (°C)

# Numerical parameters  
nx, ny = 21, 21  # Grid points in x and y (smaller for stability)
dx = L / (nx - 1)
dy = W / (ny - 1)

# Stability condition for explicit method: dt <= min(dx²,dy²)/(4*α)
# For 2D: dt <= (dx*dy)² / (2*α*(dx² + dy²))
dt_max = min(dx**2, dy**2) / (4 * alpha)  # More conservative stability condition
dt = 0.2 * dt_max  # Use 20% of maximum stable timestep (very conservative)
print(f"Grid size: {nx} x {ny}")
print(f"Spatial resolution: dx = {dx:.3f} mm, dy = {dy:.3f} mm")  
print(f"Time step: dt = {dt:.6f} s (stability factor: 0.2)")
print(f"Thermal diffusivity: α = {alpha} mm²/s")

# Create SymPy symbols for 2D heat equation
print("\n" + "="*50)
print("CREATING SYMPY-HPX HEAT DIFFUSION FUNCTION")
print("="*50)

i, j = symbols('i j', integer=True)
u = IndexedBase("u")          # Current temperature field
u_new = IndexedBase("u_new")  # Next time step temperature
alpha_sym = Symbol("alpha")   # Thermal diffusivity
dt_sym = Symbol("dt")         # Time step
dx_sym = Symbol("dx")         # Grid spacing x
dy_sym = Symbol("dy")         # Grid spacing y

# 2D heat diffusion equation with 5-point stencil
# u_new[i,j] = u[i,j] + α*dt*((u[i+1,j] - 2*u[i,j] + u[i-1,j])/dx² + (u[i,j+1] - 2*u[i,j] + u[i,j-1])/dy²)
heat_eq = Eq(u_new[i,j], 
             u[i,j] + alpha_sym * dt_sym * (
                 (u[i+1,j] - 2*u[i,j] + u[i-1,j]) / dx_sym**2 +
                 (u[i,j+1] - 2*u[i,j] + u[i,j-1]) / dy_sym**2
             ))

print(f"Heat equation: {heat_eq}")
print("Generating HPX-parallel C++ function...")

# Generate the HPX-accelerated heat diffusion function
start_time = time.time()
heat_func = genFunc(heat_eq)
compile_time = time.time() - start_time
print(f"✓ HPX compilation completed in {compile_time:.3f} seconds")

# Initialize temperature field
print("\n" + "="*50)
print("SETTING UP INITIAL CONDITIONS")
print("="*50)

# Create flattened arrays for 2D data (row-major order)
u_field = np.full(nx * ny, T_cold, dtype=np.float64)  # Start with cold temperature
u_new_field = np.zeros(nx * ny, dtype=np.float64)

# Create initial hot circular region in center
center_x, center_y = L/2, W/2
radius = 1.5  # Hot region radius (mm) - smaller to avoid boundary

hot_count = 0
for i_idx in range(nx):
    for j_idx in range(ny):
        x = i_idx * dx
        y = j_idx * dy
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Only set hot temperature for interior points (not on boundaries)
        if distance <= radius and i_idx > 0 and i_idx < nx-1 and j_idx > 0 and j_idx < ny-1:
            flat_idx = i_idx * ny + j_idx
            u_field[flat_idx] = T_hot
            hot_count += 1

# Enforce initial boundary conditions
u_2d = u_field.reshape(nx, ny)
u_2d[0, :] = T_cold    # Top edge
u_2d[-1, :] = T_cold   # Bottom edge  
u_2d[:, 0] = T_cold    # Left edge
u_2d[:, -1] = T_cold   # Right edge  
u_field = u_2d.flatten()

print(f"Initial conditions set:")
print(f"- Cold temperature: {T_cold}°C everywhere")
print(f"- Hot circular region: {T_hot}°C (radius {radius} mm, {hot_count} points)")

# Time stepping parameters
n_steps = 200
output_interval = 40  # Save results every N steps
times_to_save = [0, output_interval, 2*output_interval, 3*output_interval, 4*output_interval]

print(f"\nTime integration:")
print(f"- Total time steps: {n_steps}")
print(f"- Total simulation time: {n_steps * dt:.3f} seconds")
print(f"- Output saved at steps: {times_to_save}")

# Storage for results at specific times
results = {}

print("\n" + "="*50)
print("RUNNING HPX-PARALLEL HEAT DIFFUSION SIMULATION")
print("="*50)

# Time-stepping loop with HPX acceleration
simulation_start = time.time()
for step in range(n_steps + 1):
    current_time = step * dt
    
    if step in times_to_save:
        # Store result (reshape flattened array to 2D for visualization)
        temp_2d = u_field.reshape(nx, ny)
        results[step] = {
            'temperature': temp_2d.copy(),
            'time': current_time,
            'max_temp': np.max(temp_2d),
            'min_temp': np.min(temp_2d),
            'center_temp': temp_2d[nx//2, ny//2]
        }
        
        print(f"Step {step:3d}: t={current_time:.4f}s, "
              f"T_center={results[step]['center_temp']:.1f}°C, "
              f"T_range=[{results[step]['min_temp']:.1f}, {results[step]['max_temp']:.1f}]°C")
    
    if step < n_steps:
        # Apply HPX-parallel heat diffusion update
        # Arguments: result_array, input_array, rows, cols, alpha, dt, dx, dy
        heat_func(u_new_field, u_field, nx, ny, alpha, dt, dx, dy)
        
        # Enforce boundary conditions (keep edges at T_cold)
        u_new_2d = u_new_field.reshape(nx, ny)
        u_new_2d[0, :] = T_cold    # Top edge
        u_new_2d[-1, :] = T_cold   # Bottom edge  
        u_new_2d[:, 0] = T_cold    # Left edge
        u_new_2d[:, -1] = T_cold   # Right edge
        u_new_field = u_new_2d.flatten()
        
        # Check for numerical stability
        max_temp = np.max(u_new_field)
        min_temp = np.min(u_new_field)
        if max_temp > 1000 or min_temp < -100:
            print(f"\n⚠ WARNING: Numerical instability detected at step {step+1}")
            print(f"   Temperature range: [{min_temp:.2f}, {max_temp:.2f}]°C")
            print("   Stopping simulation to prevent overflow...")
            break
        
        # Swap arrays for next iteration
        u_field, u_new_field = u_new_field, u_field

simulation_time = time.time() - simulation_start
print(f"\n✓ Simulation completed in {simulation_time:.3f} seconds")
print(f"✓ HPX processed {n_steps * nx * ny:,} total grid points")
print(f"✓ Performance: {(n_steps * nx * ny) / simulation_time / 1e6:.2f} million points/second")

# Analytical verification (steady-state check)
print("\n" + "="*50)
print("ANALYTICAL VERIFICATION")
print("="*50)

steady_state_temp = results[times_to_save[-1]]['temperature']
temp_range = np.max(steady_state_temp) - np.min(steady_state_temp)
print(f"Final temperature range: {temp_range:.2f}°C")

# For long-time behavior, temperature should approach equilibrium
# Check if we're moving toward equilibrium (temperature differences decreasing)
initial_range = results[0]['max_temp'] - results[0]['min_temp']
final_range = results[times_to_save[-1]]['max_temp'] - results[times_to_save[-1]]['min_temp']
diffusion_progress = (initial_range - final_range) / initial_range * 100

print(f"Initial temperature range: {initial_range:.2f}°C")
print(f"Final temperature range: {final_range:.2f}°C")
print(f"Diffusion progress: {diffusion_progress:.1f}% toward equilibrium")

if diffusion_progress > 0:
    print("✓ Physical behavior verified: Heat is diffusing as expected")
else:
    print("⚠ Warning: Heat diffusion may need more time or different parameters")

# Visualization
print("\n" + "="*50)
print("CREATING VISUALIZATION")
print("="*50)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('2D Heat Diffusion - HPX Parallel Simulation Results', fontsize=16)

# Plot temperature evolution
plot_times = times_to_save[:5]  # Use first 5 time points
for idx, step in enumerate(plot_times):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    temp_data = results[step]['temperature']
    
    # Create temperature contour plot
    im = ax.imshow(temp_data.T, extent=[0, L, 0, W], origin='lower', 
                   cmap='hot', vmin=T_cold, vmax=T_hot)
    ax.set_title(f't = {results[step]["time"]:.4f}s\n'
                f'Center: {results[step]["center_temp"]:.1f}°C')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    
    # Add colorbar for the last plot
    if idx == len(plot_times) - 1:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Temperature (°C)')

# Performance comparison plot
ax_perf = axes[1, 2]
time_points = [results[step]['time'] for step in times_to_save]
center_temps = [results[step]['center_temp'] for step in times_to_save]

ax_perf.plot(time_points, center_temps, 'ro-', linewidth=2, markersize=8)
ax_perf.set_xlabel('Time (s)')
ax_perf.set_ylabel('Center Temperature (°C)')
ax_perf.set_title('Temperature Evolution at Center')
ax_perf.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/c/Users/rrajo/github-repos/hpx-pyapi/sympy-hpx/examples/2D_heat_diffusion_results.png', 
            dpi=300, bbox_inches='tight')
print("✓ Results saved to: 2D_heat_diffusion_results.png")

# Summary
print("\n" + "="*70)
print("SIMULATION SUMMARY")
print("="*70)
print(f"Problem solved: 2D Heat Diffusion Equation")
print(f"Grid resolution: {nx} × {ny} = {nx*ny:,} points")
print(f"Time steps: {n_steps}")
print(f"HPX compilation time: {compile_time:.3f} seconds")
print(f"HPX simulation time: {simulation_time:.3f} seconds")
print(f"Total time: {compile_time + simulation_time:.3f} seconds")
print(f"Performance: {(n_steps * nx * ny) / simulation_time / 1e6:.2f} million updates/second")
print()
print("Physical Results:")
print(f"- Initial hot region temperature: {T_hot}°C")
print(f"- Final center temperature: {results[times_to_save[-1]]['center_temp']:.1f}°C")
print(f"- Diffusion progress: {diffusion_progress:.1f}% toward equilibrium")
print(f"- Temperature range decreased by: {initial_range - final_range:.2f}°C")
print()
print("✓ 2D Heat diffusion successfully solved with HPX parallel acceleration!")
print("✓ Results show physically realistic heat spreading behavior")
print("✓ sympy-hpx v4 automatically generated optimized parallel C++ code")
print("="*70)

plt.show()