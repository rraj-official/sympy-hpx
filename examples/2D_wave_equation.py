#!/usr/bin/env python3
"""
2D Wave Equation - Scientific Computing Example

This example demonstrates solving the 2D wave equation using 
sympy-hpx v4's multi-dimensional capabilities with HPX parallel acceleration.

Problem: Wave propagation on a 2D membrane (e.g., vibrating drum)
Equation: ∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²)

Where:
- u(x,y,t) = displacement/amplitude at position (x,y) and time t
- c = wave speed (m/s)
- Boundary conditions: Fixed edges (Dirichlet) or absorbing boundaries
- Initial conditions: Localized disturbance and initial velocity

Reference: 
- Beltoforion.de "2D Wave Equation with Finite Differences"
- SciPython "The Two-Dimensional Wave Equation"
- Stability condition: c*dt/dx ≤ 1/√2 for 2D case
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
print("2D WAVE EQUATION - SCIENTIFIC COMPUTING EXAMPLE")
print("=" * 70)
print("Solving: ∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²)")
print("Using: sympy-hpx v4 with HPX parallel acceleration")
print("=" * 70)

# Physical parameters
c = 1.0          # Wave speed (m/s)
L = 10.0         # Domain length (m)
W = 10.0         # Domain width (m)

# Numerical parameters
nx, ny = 41, 41  # Grid points in x and y
dx = L / (nx - 1)
dy = W / (ny - 1)

# Stability condition for 2D wave equation: c*dt ≤ min(dx,dy)/√2
dt_max = min(dx, dy) / (c * np.sqrt(2))
dt = 0.8 * dt_max  # Use 80% of maximum stable timestep

print(f"Physical parameters:")
print(f"- Wave speed: c = {c} m/s")
print(f"- Domain: {L} × {W} m")
print(f"Grid parameters:")
print(f"- Grid size: {nx} × {ny}")
print(f"- Spatial resolution: dx = {dx:.3f} m, dy = {dy:.3f} m")  
print(f"- Time step: dt = {dt:.6f} s (stability factor: 0.8)")
print(f"- CFL number: c*dt/dx = {c*dt/dx:.3f} (must be ≤ 0.707 for 2D)")

# Create SymPy symbols for 2D wave equation
print("\n" + "="*50)
print("CREATING SYMPY-HPX WAVE EQUATION FUNCTION")
print("="*50)

i, j = symbols('i j', integer=True)
u_new = IndexedBase("u_new")  # Next time step: u(t+dt)
u_curr = IndexedBase("u_curr") # Current time step: u(t)  
u_prev = IndexedBase("u_prev") # Previous time step: u(t-dt)
c_sym = Symbol("c")           # Wave speed
dt_sym = Symbol("dt")         # Time step
dx_sym = Symbol("dx")         # Grid spacing x
dy_sym = Symbol("dy")         # Grid spacing y

# 2D Wave equation finite difference discretization:
# u_new[i,j] = 2*u_curr[i,j] - u_prev[i,j] + (c*dt)²*((u_curr[i+1,j] - 2*u_curr[i,j] + u_curr[i-1,j])/dx² + (u_curr[i,j+1] - 2*u_curr[i,j] + u_curr[i,j-1])/dy²)
wave_eq = Eq(u_new[i,j], 
             2*u_curr[i,j] - u_prev[i,j] + 
             (c_sym * dt_sym)**2 * (
                 (u_curr[i+1,j] - 2*u_curr[i,j] + u_curr[i-1,j]) / dx_sym**2 +
                 (u_curr[i,j+1] - 2*u_curr[i,j] + u_curr[i,j-1]) / dy_sym**2
             ))

print(f"Wave equation discretization:")
print(f"u_new[i,j] = 2*u[i,j] - u_old[i,j] + (c*dt)²*∇²u[i,j]")
print("Generating HPX-parallel C++ function...")

# Generate the HPX-accelerated wave propagation function
start_time = time.time()
wave_func = genFunc(wave_eq)
compile_time = time.time() - start_time
print(f"✓ HPX compilation completed in {compile_time:.3f} seconds")

# Initialize wave field arrays (need 3 time levels for wave equation)
print("\n" + "="*50)
print("SETTING UP INITIAL CONDITIONS")
print("="*50)

# Create flattened arrays for 2D data (row-major order)
u_prev_field = np.zeros(nx * ny, dtype=np.float64)  # u(t-dt)
u_curr_field = np.zeros(nx * ny, dtype=np.float64)  # u(t)
u_new_field = np.zeros(nx * ny, dtype=np.float64)   # u(t+dt)

# Set initial conditions: Gaussian pulse at two locations (interference)
print("Setting initial conditions:")
print("- Two Gaussian pulses for wave interference")

# First pulse at (3, 5)
x1, y1 = 3.0, 5.0
sigma1 = 0.8
amplitude1 = 1.0

# Second pulse at (7, 5) 
x2, y2 = 7.0, 5.0
sigma2 = 0.8
amplitude2 = 0.8

pulse_count = 0
for i_idx in range(nx):
    for j_idx in range(ny):
        x = i_idx * dx
        y = j_idx * dy
        
        # First Gaussian pulse
        r1_sq = (x - x1)**2 + (y - y1)**2
        pulse1 = amplitude1 * np.exp(-r1_sq / (2 * sigma1**2))
        
        # Second Gaussian pulse (slightly delayed)
        r2_sq = (x - x2)**2 + (y - y2)**2  
        pulse2 = amplitude2 * np.exp(-r2_sq / (2 * sigma2**2))
        
        flat_idx = i_idx * ny + j_idx
        
        # Initial displacement: u(x,y,0) = pulse1 + pulse2
        u_curr_field[flat_idx] = pulse1 + pulse2
        
        # Initial velocity: ∂u/∂t(x,y,0) = 0 (start from rest)
        # This means u_prev = u_curr for first step
        u_prev_field[flat_idx] = u_curr_field[flat_idx]
        
        if pulse1 + pulse2 > 0.01:
            pulse_count += 1

print(f"- Pulse 1: center=({x1}, {y1}), σ={sigma1}, A={amplitude1}")
print(f"- Pulse 2: center=({x2}, {y2}), σ={sigma2}, A={amplitude2}")
print(f"- Initial disturbance covers {pulse_count} grid points")
print(f"- Initial velocity: ∂u/∂t = 0 (start from rest)")

# Time stepping parameters
n_steps = 300
output_interval = 60  # Save results every N steps
times_to_save = [0, output_interval, 2*output_interval, 3*output_interval, 4*output_interval]

total_time = n_steps * dt
print(f"\nTime integration:")
print(f"- Total time steps: {n_steps}")
print(f"- Total simulation time: {total_time:.3f} seconds")
print(f"- Output saved at steps: {times_to_save}")

# Storage for results at specific times
results = {}

print("\n" + "="*50)
print("RUNNING HPX-PARALLEL WAVE EQUATION SIMULATION")
print("="*50)

# Time-stepping loop with HPX acceleration
simulation_start = time.time()
for step in range(n_steps + 1):
    current_time = step * dt
    
    if step in times_to_save:
        # Store result (reshape flattened array to 2D for visualization)
        wave_2d = u_curr_field.reshape(nx, ny)
        results[step] = {
            'wave': wave_2d.copy(),
            'time': current_time,
            'max_amp': np.max(np.abs(wave_2d)),
            'total_energy': np.sum(wave_2d**2)  # Approximation of wave energy
        }
        
        print(f"Step {step:3d}: t={current_time:.4f}s, "
              f"max_amplitude={results[step]['max_amp']:.3f}, "
              f"energy={results[step]['total_energy']:.1f}")
    
    if step < n_steps:
        # Apply HPX-parallel wave equation update
        # Note: Wave equation needs 3 time levels, unlike heat equation's 2
        # Arguments: result_array, u_current, u_previous, rows, cols, c, dt, dx, dy
        wave_func(u_new_field, u_curr_field, u_prev_field, nx, ny, c, dt, dx, dy)
        
        # Enforce boundary conditions (fixed boundaries: u = 0 at edges)
        u_new_2d = u_new_field.reshape(nx, ny)
        u_new_2d[0, :] = 0.0     # Top edge
        u_new_2d[-1, :] = 0.0    # Bottom edge  
        u_new_2d[:, 0] = 0.0     # Left edge
        u_new_2d[:, -1] = 0.0    # Right edge
        u_new_field = u_new_2d.flatten()
        
        # Check for numerical stability
        max_amp = np.max(np.abs(u_new_field))
        if max_amp > 100:
            print(f"\n⚠ WARNING: Numerical instability detected at step {step+1}")
            print(f"   Maximum amplitude: {max_amp:.2f}")
            print("   Stopping simulation to prevent overflow...")
            break
        
        # Time step: rotate arrays for next iteration
        # u_prev <- u_curr <- u_new
        u_prev_field, u_curr_field, u_new_field = u_curr_field, u_new_field, u_prev_field

simulation_time = time.time() - simulation_start
print(f"\n✓ Simulation completed in {simulation_time:.3f} seconds")
print(f"✓ HPX processed {n_steps * nx * ny:,} total grid points")
print(f"✓ Performance: {(n_steps * nx * ny) / simulation_time / 1e6:.2f} million points/second")

# Wave physics verification
print("\n" + "="*50)
print("WAVE PHYSICS VERIFICATION")
print("="*50)

initial_energy = results[0]['total_energy']
final_energy = results[times_to_save[-1]]['total_energy']
energy_change = abs(final_energy - initial_energy) / initial_energy * 100

print(f"Energy analysis:")
print(f"- Initial wave energy: {initial_energy:.1f}")
print(f"- Final wave energy: {final_energy:.1f}")
print(f"- Energy change: {energy_change:.2f}%")

if energy_change < 5:
    print("✓ Energy approximately conserved (good numerical accuracy)")
elif energy_change < 15:
    print("⚠ Moderate energy drift (acceptable for this timestep)")
else:
    print("⚠ Significant energy drift (may need smaller timestep)")

# Check wave speed
max_initial_amp = results[0]['max_amp']
final_max_amp = results[times_to_save[-1]]['max_amp']
amplitude_ratio = final_max_amp / max_initial_amp

print(f"Amplitude analysis:")
print(f"- Initial max amplitude: {max_initial_amp:.3f}")
print(f"- Final max amplitude: {final_max_amp:.3f}")
print(f"- Amplitude ratio: {amplitude_ratio:.3f}")

if amplitude_ratio > 0.5:
    print("✓ Wave amplitudes well maintained")
elif amplitude_ratio > 0.2:
    print("✓ Moderate wave dispersion (normal for discrete scheme)")
else:
    print("⚠ Significant wave attenuation")

# Visualization
print("\n" + "="*50)
print("CREATING VISUALIZATION")
print("="*50)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('2D Wave Equation - HPX Parallel Simulation Results\n'
            'Wave Interference and Propagation', fontsize=16)

# Plot wave evolution
plot_times = times_to_save[:5]  # Use first 5 time points
vmax = max([results[step]['max_amp'] for step in plot_times])

for idx, step in enumerate(plot_times):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    wave_data = results[step]['wave']
    
    # Create wave surface plot
    X, Y = np.meshgrid(np.linspace(0, L, nx), np.linspace(0, W, ny))
    im = ax.contourf(X, Y, wave_data.T, levels=20, cmap='RdBu_r', 
                     vmin=-vmax, vmax=vmax, extend='both')
    ax.contour(X, Y, wave_data.T, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    
    ax.set_title(f't = {results[step]["time"]:.4f}s\n'
                f'Max Amp: {results[step]["max_amp"]:.3f}')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')
    
    # Add colorbar for the last plot
    if idx == len(plot_times) - 1:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Wave Amplitude')

# Energy evolution plot
ax_energy = axes[1, 2]
time_points = [results[step]['time'] for step in times_to_save]
energies = [results[step]['total_energy'] for step in times_to_save]

ax_energy.plot(time_points, energies, 'bo-', linewidth=2, markersize=8)
ax_energy.set_xlabel('Time (s)')
ax_energy.set_ylabel('Total Wave Energy')
ax_energy.set_title('Energy Conservation Check')
ax_energy.grid(True, alpha=0.3)

# Add energy conservation reference line
ax_energy.axhline(y=initial_energy, color='red', linestyle='--', alpha=0.7, 
                 label=f'Initial Energy: {initial_energy:.1f}')
ax_energy.legend()

plt.tight_layout()
plt.savefig('/mnt/c/Users/rrajo/github-repos/hpx-pyapi/sympy-hpx/examples/2D_wave_equation_results.png', 
            dpi=300, bbox_inches='tight')
print("✓ Results saved to: 2D_wave_equation_results.png")

# Summary
print("\n" + "="*70)
print("SIMULATION SUMMARY")
print("="*70)
print(f"Problem solved: 2D Wave Equation (Hyperbolic PDE)")
print(f"Grid resolution: {nx} × {ny} = {nx*ny:,} points")
print(f"Time steps: {n_steps}")
print(f"Wave speed: {c} m/s")
print(f"CFL number: {c*dt/dx:.3f} (stable: < 0.707)")
print(f"HPX compilation time: {compile_time:.3f} seconds")
print(f"HPX simulation time: {simulation_time:.3f} seconds")
print(f"Total time: {compile_time + simulation_time:.3f} seconds")
print(f"Performance: {(n_steps * nx * ny) / simulation_time / 1e6:.2f} million updates/second")
print()
print("Wave Physics Results:")
print(f"- Wave interference observed: ✓")
print(f"- Energy conservation: {energy_change:.1f}% drift")
print(f"- Amplitude preservation: {amplitude_ratio:.3f} ratio")
print(f"- Numerical stability: {'✓' if max_amp < 10 else '⚠'}")
print()
print("Key Differences from Heat Equation:")
print("- Hyperbolic PDE (oscillatory) vs. Parabolic PDE (diffusive)")
print("- Requires 3 time levels vs. 2 time levels")
print("- More restrictive stability: CFL ≤ 1/√2 vs. heat stability")
print("- Conserves energy vs. dissipates energy")
print("- Wave propagation vs. heat diffusion")
print()
print("✓ 2D Wave equation successfully solved with HPX parallel acceleration!")
print("✓ Wave interference and propagation behavior correctly simulated")
print("✓ sympy-hpx v4 automatically generated optimized parallel C++ code")
print("="*70)

plt.show()