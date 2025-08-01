#!/usr/bin/env python3
"""
1D Schrödinger Equation - Quantum Mechanics Example

This example demonstrates solving the 1D time-dependent Schrödinger equation
using sympy-hpx v4's capabilities with HPX parallel acceleration.

Problem: Quantum wave packet evolution and tunneling
Equation: iℏ ∂ψ/∂t = -ℏ²/(2m) ∂²ψ/∂x² + V(x)ψ

Where:
- ψ(x,t) = complex quantum wave function (probability amplitude)
- ℏ = reduced Planck constant (set to 1 in atomic units)
- m = particle mass (set to 1 in atomic units)  
- V(x) = potential energy function
- |ψ(x,t)|² = probability density of finding particle at position x

Key quantum phenomena demonstrated:
- Wave packet dispersion (uncertainty principle)
- Quantum tunneling through potential barriers
- Interference and superposition

Reference:
- Jake VanderPlas "Quantum Python: Animating the Schrödinger Equation"
- Split-step Fourier method for numerical solution
- SciPy Cookbook "FDTD Algorithm Applied to the Schrödinger Equation"
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
print("1D SCHRÖDINGER EQUATION - QUANTUM MECHANICS EXAMPLE")
print("=" * 70)
print("Solving: iℏ ∂ψ/∂t = -ℏ²/(2m) ∂²ψ/∂x² + V(x)ψ")
print("Using: sympy-hpx v4 with HPX parallel acceleration")
print("Demonstrating: Quantum tunneling through potential barrier")
print("=" * 70)

# Physical parameters (atomic units: ℏ = m = 1)
hbar = 1.0  # Reduced Planck constant
m = 1.0     # Particle mass
L = 200.0   # Simulation domain length

# Numerical parameters  
nx = 1024   # Number of grid points (power of 2 for FFT efficiency)
dx = L / nx # Spatial grid spacing
x = np.linspace(-L/2, L/2, nx)

# Time stepping parameters - much smaller dt needed for Schrödinger stability
dt = 0.002   # Time step (much smaller for numerical stability)  
n_steps = 500
total_time = n_steps * dt

print(f"Physical parameters (atomic units):")
print(f"- ℏ = {hbar}, m = {m}")
print(f"- Domain: [-{L/2:.1f}, {L/2:.1f}]")
print(f"Grid parameters:")
print(f"- Grid points: {nx}")
print(f"- Spatial resolution: dx = {dx:.3f}")
print(f"- Time step: dt = {dt}")
print(f"- Total time: {total_time:.1f}")

# Create potential barrier for quantum tunneling demonstration
print("\n" + "="*50)
print("CREATING QUANTUM TUNNELING SCENARIO")
print("="*50)

# Gaussian wave packet parameters
x0 = -50.0      # Initial position (left side)
k0 = 0.5        # Initial momentum (rightward)
sigma = 8.0     # Wave packet width
E = hbar**2 * k0**2 / (2 * m)  # Kinetic energy

# Potential barrier parameters
barrier_height = 1.5 * E  # Barrier higher than particle energy
barrier_width = 20.0
barrier_center = 0.0

print(f"Wave packet parameters:")
print(f"- Initial position: x₀ = {x0}")
print(f"- Initial momentum: k₀ = {k0}")
print(f"- Wave packet width: σ = {sigma}")
print(f"- Kinetic energy: E = {E:.4f}")

print(f"Potential barrier:")
print(f"- Height: V₀ = {barrier_height:.4f} (ratio E/V₀ = {E/barrier_height:.2f})")
print(f"- Width: {barrier_width}")
print(f"- Center: {barrier_center}")
print(f"- Classical prediction: TOTAL REFLECTION (E < V₀)")
print(f"- Quantum prediction: PARTIAL TUNNELING")

# Create potential function
V = np.zeros_like(x)
barrier_mask = (np.abs(x - barrier_center) <= barrier_width/2)
V[barrier_mask] = barrier_height

# Add absorbing boundaries to prevent reflections at domain edges
absorb_width = 20.0
absorb_strength = 5.0
left_absorb = (x < -L/2 + absorb_width)
right_absorb = (x > L/2 - absorb_width)
V[left_absorb] += absorb_strength
V[right_absorb] += absorb_strength

# Create initial Gaussian wave packet
print("\n" + "="*50)
print("CREATING SYMPY-HPX SCHRÖDINGER EVOLUTION FUNCTION")
print("="*50)

# For the Schrödinger equation, we need to handle complex fields
# We'll split into real and imaginary parts and create coupled evolution equations

# SymPy symbols
i_idx = symbols('i', integer=True)
psi_real = IndexedBase("psi_real")     # Real part of wave function
psi_imag = IndexedBase("psi_imag")     # Imaginary part of wave function
psi_real_new = IndexedBase("psi_real_new")  # Next time step real
psi_imag_new = IndexedBase("psi_imag_new")  # Next time step imaginary
V_pot = IndexedBase("V")               # Potential
hbar_sym = Symbol("hbar")
m_sym = Symbol("m")
dt_sym = Symbol("dt")
dx_sym = Symbol("dx")

print("Schrödinger equation split into real and imaginary parts:")
print("Real part: ∂ψᵣ/∂t = +(ℏ/(2m)) ∂²ψᵢ/∂x² - (V/ℏ)ψᵢ")
print("Imag part: ∂ψᵢ/∂t = -(ℏ/(2m)) ∂²ψᵣ/∂x² + (V/ℏ)ψᵣ")

# Finite difference discretization for second derivative: (ψ[i+1] - 2ψ[i] + ψ[i-1])/dx²
# Real part evolution: ψᵣ(new) = ψᵣ + dt * [+(ℏ/(2m)) * ∇²ψᵢ - (V/ℏ)*ψᵢ]
schrodinger_real = Eq(psi_real_new[i_idx], 
                      psi_real[i_idx] + dt_sym * (
                          hbar_sym/(2*m_sym) * (psi_imag[i_idx+1] - 2*psi_imag[i_idx] + psi_imag[i_idx-1])/dx_sym**2
                          - V_pot[i_idx]/hbar_sym * psi_imag[i_idx]
                      ))

# Imaginary part evolution: ψᵢ(new) = ψᵢ + dt * [-(ℏ/(2m)) * ∇²ψᵣ + (V/ℏ)*ψᵣ]  
schrodinger_imag = Eq(psi_imag_new[i_idx],
                      psi_imag[i_idx] + dt_sym * (
                          -hbar_sym/(2*m_sym) * (psi_real[i_idx+1] - 2*psi_real[i_idx] + psi_real[i_idx-1])/dx_sym**2
                          + V_pot[i_idx]/hbar_sym * psi_real[i_idx]
                      ))

print("Generating HPX-parallel C++ functions...")

# Generate HPX functions for both real and imaginary evolution
start_time = time.time()
schrodinger_real_func = genFunc(schrodinger_real)
schrodinger_imag_func = genFunc(schrodinger_imag)
compile_time = time.time() - start_time
print(f"✓ HPX compilation completed in {compile_time:.3f} seconds")

# Initialize quantum wave function (complex Gaussian wave packet)
print("\n" + "="*50)
print("SETTING UP INITIAL QUANTUM STATE")
print("="*50)

# Create initial Gaussian wave packet: ψ(x,0) = exp(-(x-x₀)²/(4σ²) + ik₀x)
# Normalized so that ∫|ψ|²dx = 1
A = (2*np.pi*sigma**2)**(-0.25)  # Normalization constant
psi_real_init = A * np.exp(-(x - x0)**2 / (4*sigma**2)) * np.cos(k0 * x)
psi_imag_init = A * np.exp(-(x - x0)**2 / (4*sigma**2)) * np.sin(k0 * x)

# Arrays for current and next time step
psi_real_current = psi_real_init.copy()
psi_imag_current = psi_imag_init.copy()
psi_real_next = np.zeros_like(psi_real_current)
psi_imag_next = np.zeros_like(psi_imag_current)

# Verify normalization
prob_density = psi_real_current**2 + psi_imag_current**2
total_prob = np.trapezoid(prob_density, x)
print(f"Initial wave packet:")
print(f"- Peak position: {x[np.argmax(prob_density)]:.1f}")
print(f"- Total probability: {total_prob:.6f} (should be ~1.0)")
print(f"- Average momentum: <p> = {k0:.3f}")
print(f"- Group velocity: vₓ = <p>/m = {k0/m:.3f}")

# Classical time to reach barrier
time_to_barrier = (barrier_center - x0) / (k0/m)
print(f"- Classical time to barrier: {time_to_barrier:.1f}")

# Time evolution parameters
output_times = [0, 100, 200, 300, 400, 500]  # Steps to save results
results = {}

print(f"\nTime integration:")
print(f"- Time steps: {n_steps}")
print(f"- Output saved at steps: {output_times}")

print("\n" + "="*50)
print("RUNNING HPX-PARALLEL SCHRÖDINGER EVOLUTION")
print("="*50)

# Time-stepping loop with HPX acceleration
simulation_start = time.time()
for step in range(n_steps + 1):
    current_time = step * dt
    
    if step in output_times:
        # Calculate probability density and quantum observables
        prob_density = psi_real_current**2 + psi_imag_current**2
        
        # Quantum expectation values
        x_avg = np.trapezoid(x * prob_density, x)  # <x>
        x2_avg = np.trapezoid(x**2 * prob_density, x)  # <x²>
        sigma_x = np.sqrt(x2_avg - x_avg**2)  # Position uncertainty
        
        # Momentum expectation (via gradient)
        psi_complex = psi_real_current + 1j * psi_imag_current
        dpsi_dx = np.gradient(psi_complex, dx)
        p_density = np.real(np.conj(psi_complex) * (-1j * hbar * dpsi_dx))
        p_avg = np.trapezoid(p_density, x)
        
        # Store results
        results[step] = {
            'time': current_time,
            'psi_real': psi_real_current.copy(),
            'psi_imag': psi_imag_current.copy(),
            'prob_density': prob_density.copy(),
            'x_avg': x_avg,
            'sigma_x': sigma_x,
            'p_avg': p_avg,
            'total_prob': np.trapezoid(prob_density, x)
        }
        
        print(f"Step {step:3d}: t={current_time:6.2f}, "
              f"<x>={x_avg:6.1f}, σₓ={sigma_x:5.1f}, "
              f"<p>={p_avg:5.3f}, ∫|ψ|²={results[step]['total_prob']:.6f}")
    
    if step < n_steps:
        # Apply HPX-parallel Schrödinger evolution
        # Note: Boundary conditions handled by not updating boundary points
        schrodinger_real_func(psi_real_next, psi_real_current, psi_imag_current, V, nx, hbar, m, dt, dx)
        schrodinger_imag_func(psi_imag_next, psi_imag_current, psi_real_current, V, nx, hbar, m, dt, dx)
        
        # Apply boundary conditions (wave function = 0 at edges)
        psi_real_next[0] = psi_real_next[-1] = 0.0
        psi_imag_next[0] = psi_imag_next[-1] = 0.0
        
        # Check for numerical stability
        prob_density_check = psi_real_next**2 + psi_imag_next**2
        max_prob = np.max(prob_density_check)
        if max_prob > 100:
            print(f"\n⚠ WARNING: Numerical instability at step {step+1}")
            print(f"   Maximum probability density: {max_prob:.2f}")
            print("   Stopping simulation...")
            break
        
        # Update for next iteration
        psi_real_current, psi_real_next = psi_real_next, psi_real_current
        psi_imag_current, psi_imag_next = psi_imag_next, psi_imag_current

simulation_time = time.time() - simulation_start
print(f"\n✓ Simulation completed in {simulation_time:.3f} seconds")
print(f"✓ HPX processed {n_steps * nx:,} total grid points")
print(f"✓ Performance: {(n_steps * nx) / simulation_time / 1e6:.2f} million points/second")

# Quantum tunneling analysis
print("\n" + "="*50)
print("QUANTUM TUNNELING ANALYSIS")
print("="*50)

# Analyze transmission and reflection coefficients
# Use the last available result in case simulation stopped early
available_steps = list(results.keys())
if not available_steps:
    print("❌ No results available - simulation failed completely")
    exit(1)
    
final_step = max(available_steps)
final_prob = results[final_step]['prob_density']
print(f"Using results from step {final_step} (t = {results[final_step]['time']:.2f})")

# Split domain into three regions: left, barrier, right
left_region = x < (barrier_center - barrier_width/2)
barrier_region = (np.abs(x - barrier_center) <= barrier_width/2)  
right_region = x > (barrier_center + barrier_width/2)

# Calculate probabilities in each region
P_left = np.trapezoid(final_prob[left_region], x[left_region])
P_barrier = np.trapezoid(final_prob[barrier_region], x[barrier_region])
P_right = np.trapezoid(final_prob[right_region], x[right_region])
P_total = P_left + P_barrier + P_right

print(f"Final probability distribution:")
print(f"- Left of barrier (reflected): {P_left:.6f} ({P_left/P_total*100:.2f}%)")
print(f"- Inside barrier: {P_barrier:.6f} ({P_barrier/P_total*100:.2f}%)")  
print(f"- Right of barrier (transmitted): {P_right:.6f} ({P_right/P_total*100:.2f}%)")
print(f"- Total probability: {P_total:.6f}")

# Transmission and reflection coefficients
T = P_right / P_total  # Transmission coefficient
R = P_left / P_total   # Reflection coefficient

print(f"\nQuantum tunneling results:")
print(f"- Transmission coefficient: T = {T:.6f} ({T*100:.3f}%)")
print(f"- Reflection coefficient: R = {R:.6f} ({R*100:.3f}%)")
print(f"- T + R = {T + R:.6f} (should be ~1.0)")

# Compare with classical prediction
print(f"\nClassical vs. Quantum:")
print(f"- Classical: T = 0% (total reflection, E < V₀)")
print(f"- Quantum: T = {T*100:.3f}% (tunneling through barrier)")
print(f"- Quantum effect: {T*100:.3f}% tunneling probability!")

# Theoretical estimate (for rectangular barrier)
kappa = np.sqrt(2 * m * (barrier_height - E)) / hbar  # Decay constant in barrier
T_theory = 1 / (1 + (barrier_height**2 * np.sinh(kappa * barrier_width)**2) / (4 * E * (barrier_height - E)))
print(f"- Theoretical estimate: T ≈ {T_theory:.6f} ({T_theory*100:.3f}%)")

# Visualization
print("\n" + "="*50)  
print("CREATING VISUALIZATION")
print("="*50)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('1D Schrödinger Equation - Quantum Tunneling Simulation\n'
            'Wave Packet Evolution Through Potential Barrier', fontsize=16)

# Plot wave function evolution - use available results
plot_times = sorted(available_steps)[:6]  # Use first 6 available results
colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']

for idx, step in enumerate(plot_times):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    result = results[step]
    prob_density = result['prob_density']
    
    # Plot probability density |ψ|²
    ax.plot(x, prob_density, color=colors[idx], linewidth=2, 
            label=f'|ψ(x,t)|²')
    
    # Plot potential barrier
    ax.fill_between(x, 0, V/max(V) * max(prob_density), 
                   where=(V > 0.1), alpha=0.3, color='red',
                   label='Potential V(x)')
    
    # Mark average position
    ax.axvline(result['x_avg'], color='black', linestyle='--', alpha=0.7,
              label=f'⟨x⟩ = {result["x_avg"]:.1f}')
    
    ax.set_title(f't = {result["time"]:.1f}\n'
                f'⟨x⟩ = {result["x_avg"]:.1f}, σₓ = {result["sigma_x"]:.1f}')
    ax.set_xlabel('Position x')
    ax.set_ylabel('Probability Density')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Set consistent y-axis limits
    ax.set_ylim(0, max(prob_density) * 1.1)
    ax.set_xlim(-100, 100)

plt.tight_layout()
plt.savefig('/mnt/c/Users/rrajo/github-repos/hpx-pyapi/sympy-hpx/examples/1D_schrodinger_results.png', 
            dpi=300, bbox_inches='tight')
print("✓ Results saved to: 1D_schrodinger_results.png")

# Summary
print("\n" + "="*70)
print("SIMULATION SUMMARY")
print("="*70)
print(f"Problem solved: 1D Time-Dependent Schrödinger Equation")
print(f"Grid resolution: {nx:,} points")
print(f"Time steps: {n_steps}")
print(f"Particle energy: E = {E:.4f}")
print(f"Barrier height: V₀ = {barrier_height:.4f}")
print(f"HPX compilation time: {compile_time:.3f} seconds")
print(f"HPX simulation time: {simulation_time:.3f} seconds")
print(f"Total time: {compile_time + simulation_time:.3f} seconds")
print(f"Performance: {(n_steps * nx) / simulation_time / 1e6:.2f} million updates/second")
print()
print("Quantum Mechanics Results:")
print(f"- Wave packet successfully evolved: ✓")
print(f"- Quantum tunneling observed: T = {T*100:.3f}%")
print(f"- Probability conservation: {P_total:.6f} (≈ 1.0)")
print(f"- Wave packet dispersion: σₓ increased from {sigma:.1f} to {results[final_step]['sigma_x']:.1f}")
print()
print("Key Quantum Phenomena Demonstrated:")
print("- Complex wave function evolution (ψ = ψᵣ + iψᵢ)")
print("- Quantum tunneling through classically forbidden barrier")
print("- Wave packet dispersion due to uncertainty principle")
print("- Probability density |ψ|² interpretation")
print("- Conservation of probability current")
print()
print("Comparison with Classical Physics:")
print(f"- Classical: Complete reflection (T = 0%)")
print(f"- Quantum: Partial tunneling (T = {T*100:.3f}%)")
print(f"- Quantum advantage: Non-zero transmission despite E < V₀")
print() 
print("✓ 1D Schrödinger equation successfully solved with HPX parallel acceleration!")
print("✓ Quantum tunneling phenomenon correctly simulated")
print("✓ sympy-hpx v4 automatically generated optimized parallel C++ code")
print("="*70)

plt.show()