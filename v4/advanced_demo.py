#!/usr/bin/env python3
"""
Advanced Demo for sympy-hpx v4 - Multi-Dimensional Support
Showcases sophisticated scientific computing applications with 2D/3D arrays.
"""

from sympy import *
from sympy_codegen import genFunc
import numpy as np
import matplotlib.pyplot as plt

def demo_2d_heat_simulation():
    """Advanced 2D heat diffusion simulation with visualization."""
    print("=== Advanced Demo 1: 2D Heat Diffusion Simulation ===")
    
    # Set up 2D heat equation
    i, j = symbols('i j', integer=True)
    T = IndexedBase("T")
    T_new = IndexedBase("T_new")
    alpha = Symbol("alpha")
    dt = Symbol("dt")
    dx = Symbol("dx")
    
    # 2D heat equation with 5-point stencil
    heat_eq = Eq(T_new[i,j], T[i,j] + alpha*dt/(dx**2) * (
        T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1] - 4*T[i,j]
    ))
    
    print(f"2D Heat Equation: {heat_eq}")
    
    # Generate function
    heat_func = genFunc(heat_eq)
    print("✓ Function compiled successfully")
    
    # Simulation parameters
    rows, cols = 50, 60
    size = rows * cols
    alpha_val = 0.25
    dt_val = 0.1
    dx_val = 1.0
    
    # Initialize temperature field
    T_field = np.ones(size) * 20.0  # Background temperature
    T_new_field = np.zeros(size)
    
    # Create initial hot spots
    hot_spots = [
        (10, 15, 100.0),  # (i, j, temperature)
        (30, 40, 80.0),
        (20, 25, 120.0)
    ]
    
    for hi, hj, temp in hot_spots:
        # Create 3x3 hot spot
        for di in range(-1, 2):
            for dj in range(-1, 2):
                if 0 <= hi+di < rows and 0 <= hj+dj < cols:
                    idx = (hi+di) * cols + (hj+dj)
                    T_field[idx] = temp
    
    print(f"Grid size: {rows} x {cols}")
    print(f"Parameters: α={alpha_val}, dt={dt_val}, dx={dx_val}")
    print(f"Initial hot spots: {len(hot_spots)} locations")
    
    # Run simulation for multiple time steps
    num_steps = 20
    print(f"Running {num_steps} time steps...")
    
    for step in range(num_steps):
        heat_func(T_new_field, T_field, rows, cols, alpha_val, dt_val, dx_val)
        T_field, T_new_field = T_new_field, T_field  # Swap arrays
        
        if step % 5 == 0:
            max_temp = np.max(T_field)
            min_temp = np.min(T_field)
            print(f"  Step {step:2d}: T_max={max_temp:.1f}°C, T_min={min_temp:.1f}°C")
    
    # Analyze final state
    final_max = np.max(T_field)
    final_min = np.min(T_field)
    final_avg = np.mean(T_field)
    
    print(f"Final state:")
    print(f"  Max temperature: {final_max:.2f}°C")
    print(f"  Min temperature: {final_min:.2f}°C")
    print(f"  Average temperature: {final_avg:.2f}°C")
    
    return T_field.reshape(rows, cols)

def demo_2d_wave_propagation():
    """2D wave propagation with multiple wave sources."""
    print("\n=== Advanced Demo 2: 2D Wave Propagation ===")
    
    # Set up 2D wave equation
    i, j = symbols('i j', integer=True)
    u = IndexedBase("u")          # Current wave field
    u_prev = IndexedBase("u_prev") # Previous time step
    u_next = IndexedBase("u_next") # Next time step
    c = Symbol("c")               # Wave speed
    dt = Symbol("dt")             # Time step
    dx = Symbol("dx")             # Grid spacing
    
    # 2D wave equation: u_next = 2*u - u_prev + c^2*dt^2*Laplacian(u)
    wave_eq = Eq(u_next[i,j], 2*u[i,j] - u_prev[i,j] + 
                 c**2 * dt**2 / dx**2 * (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]))
    
    print(f"2D Wave Equation: {wave_eq}")
    
    # Generate function
    wave_func = genFunc(wave_eq)
    print("✓ Function compiled successfully")
    
    # Simulation parameters
    rows, cols = 40, 40
    size = rows * cols
    c_val = 1.0
    dt_val = 0.1
    dx_val = 1.0
    
    # Initialize wave fields
    u_field = np.zeros(size)
    u_prev_field = np.zeros(size)
    u_next_field = np.zeros(size)
    
    # Create initial wave sources (Gaussian pulses)
    wave_sources = [
        (10, 10, 2.0),  # (i, j, amplitude)
        (30, 30, -1.5),
        (20, 10, 1.0)
    ]
    
    for wi, wj, amp in wave_sources:
        for di in range(-2, 3):
            for dj in range(-2, 3):
                if 0 <= wi+di < rows and 0 <= wj+dj < cols:
                    idx = (wi+di) * cols + (wj+dj)
                    r_sq = di*di + dj*dj
                    u_field[idx] += amp * np.exp(-r_sq / 2.0)
    
    print(f"Grid size: {rows} x {cols}")
    print(f"Parameters: c={c_val}, dt={dt_val}, dx={dx_val}")
    print(f"Initial wave sources: {len(wave_sources)} locations")
    
    # Run wave simulation
    num_steps = 25
    print(f"Running {num_steps} time steps...")
    
    for step in range(num_steps):
        wave_func(u_next_field, u_field, u_prev_field, rows, cols, c_val, dt_val, dx_val)
        u_prev_field, u_field, u_next_field = u_field, u_next_field, u_prev_field
        
        if step % 5 == 0:
            max_amp = np.max(np.abs(u_field))
            energy = np.sum(u_field**2)
            print(f"  Step {step:2d}: Max amplitude={max_amp:.3f}, Energy={energy:.2f}")
    
    return u_field.reshape(rows, cols)

def demo_3d_diffusion():
    """3D diffusion in a cubic domain."""
    print("\n=== Advanced Demo 3: 3D Diffusion ===")
    
    # Set up 3D diffusion equation
    i, j, k = symbols('i j k', integer=True)
    phi = IndexedBase("phi")
    phi_new = IndexedBase("phi_new")
    D = Symbol("D")
    dt = Symbol("dt")
    dx = Symbol("dx")
    
    # 3D diffusion equation with 7-point stencil
    diffusion_eq = Eq(phi_new[i,j,k], phi[i,j,k] + D*dt/(dx**2) * (
        phi[i+1,j,k] + phi[i-1,j,k] + 
        phi[i,j+1,k] + phi[i,j-1,k] + 
        phi[i,j,k+1] + phi[i,j,k-1] - 6*phi[i,j,k]
    ))
    
    print(f"3D Diffusion Equation: {diffusion_eq}")
    
    # Generate function
    diffusion_func = genFunc(diffusion_eq)
    print("✓ Function compiled successfully")
    
    # Simulation parameters
    rows, cols, depth = 20, 20, 20
    size = rows * cols * depth
    D_val = 0.1
    dt_val = 0.01
    dx_val = 1.0
    
    # Initialize concentration field
    phi_field = np.zeros(size)
    phi_new_field = np.zeros(size)
    
    # Create initial concentration at center
    center_i, center_j, center_k = rows//2, cols//2, depth//2
    for di in range(-2, 3):
        for dj in range(-2, 3):
            for dk in range(-2, 3):
                if (0 <= center_i+di < rows and 0 <= center_j+dj < cols and 0 <= center_k+dk < depth):
                    idx = (center_i+di) * cols * depth + (center_j+dj) * depth + (center_k+dk)
                    r_sq = di*di + dj*dj + dk*dk
                    phi_field[idx] = 100.0 * np.exp(-r_sq / 4.0)
    
    print(f"Grid size: {rows} x {cols} x {depth}")
    print(f"Parameters: D={D_val}, dt={dt_val}, dx={dx_val}")
    print(f"Initial concentration: Gaussian at center")
    
    # Run 3D diffusion simulation
    num_steps = 50
    print(f"Running {num_steps} time steps...")
    
    for step in range(num_steps):
        diffusion_func(phi_new_field, phi_field, rows, cols, depth, D_val, dt_val, dx_val)
        phi_field, phi_new_field = phi_new_field, phi_field
        
        if step % 10 == 0:
            max_conc = np.max(phi_field)
            total_mass = np.sum(phi_field)
            print(f"  Step {step:2d}: Max concentration={max_conc:.3f}, Total mass={total_mass:.1f}")
    
    # Analyze final 3D distribution
    final_max = np.max(phi_field)
    final_total = np.sum(phi_field)
    
    print(f"Final 3D state:")
    print(f"  Max concentration: {final_max:.3f}")
    print(f"  Total mass: {final_total:.1f}")
    
    # Return central slice for visualization
    central_slice = np.zeros(rows * cols)
    for i in range(rows):
        for j in range(cols):
            idx_3d = i * cols * depth + j * depth + (depth//2)
            idx_2d = i * cols + j
            central_slice[idx_2d] = phi_field[idx_3d]
    
    return central_slice.reshape(rows, cols)

def demo_multi_physics_coupling():
    """Multi-physics coupling: heat and concentration."""
    print("\n=== Advanced Demo 4: Multi-Physics Coupling ===")
    
    # Set up coupled heat-concentration system
    i, j = symbols('i j', integer=True)
    T = IndexedBase("T")          # Temperature
    C = IndexedBase("C")          # Concentration
    T_new = IndexedBase("T_new")  # New temperature
    C_new = IndexedBase("C_new")  # New concentration
    
    alpha = Symbol("alpha")       # Thermal diffusivity
    D = Symbol("D")              # Mass diffusivity
    beta = Symbol("beta")        # Coupling coefficient
    dt = Symbol("dt")
    dx = Symbol("dx")
    
    # Coupled equations
    equations = [
        # Heat equation with concentration-dependent source
        Eq(T_new[i,j], T[i,j] + alpha*dt/(dx**2) * (
            T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1] - 4*T[i,j]
        ) + beta*dt*C[i,j]),
        
        # Concentration equation with temperature-dependent diffusion
        Eq(C_new[i,j], C[i,j] + D*dt/(dx**2) * (1 + 0.1*T[i,j]) * (
            C[i+1,j] + C[i-1,j] + C[i,j+1] + C[i,j-1] - 4*C[i,j]
        ))
    ]
    
    print(f"Coupled system equations:")
    for k, eq in enumerate(equations):
        print(f"  {k+1}: {eq}")
    
    # Generate function
    coupled_func = genFunc(equations)
    print("✓ Multi-physics function compiled successfully")
    
    # Simulation parameters
    rows, cols = 30, 30
    size = rows * cols
    alpha_val = 0.2
    D_val = 0.1
    beta_val = 0.05
    dt_val = 0.01
    dx_val = 1.0
    
    # Initialize fields
    T_field = np.ones(size) * 25.0      # Background temperature
    C_field = np.zeros(size)            # Zero concentration
    T_new_field = np.zeros(size)
    C_new_field = np.zeros(size)
    
    # Create initial conditions
    center_i, center_j = rows//2, cols//2
    
    # Hot spot for temperature
    for di in range(-3, 4):
        for dj in range(-3, 4):
            if 0 <= center_i+di < rows and 0 <= center_j+dj < cols:
                idx = (center_i+di) * cols + (center_j+dj)
                r_sq = di*di + dj*dj
                T_field[idx] = 25.0 + 50.0 * np.exp(-r_sq / 6.0)
    
    # Concentration source at different location
    source_i, source_j = rows//4, 3*cols//4
    for di in range(-2, 3):
        for dj in range(-2, 3):
            if 0 <= source_i+di < rows and 0 <= source_j+dj < cols:
                idx = (source_i+di) * cols + (source_j+dj)
                r_sq = di*di + dj*dj
                C_field[idx] = 10.0 * np.exp(-r_sq / 3.0)
    
    print(f"Grid size: {rows} x {cols}")
    print(f"Parameters: α={alpha_val}, D={D_val}, β={beta_val}")
    print(f"Initial conditions: Hot spot + concentration source")
    
    # Run coupled simulation
    num_steps = 100
    print(f"Running {num_steps} coupled time steps...")
    
    for step in range(num_steps):
        coupled_func(T_new_field, C_new_field, T_field, C_field, rows, cols, 
                    alpha_val, D_val, beta_val, dt_val, dx_val)
        T_field, T_new_field = T_new_field, T_field
        C_field, C_new_field = C_new_field, C_field
        
        if step % 20 == 0:
            max_T = np.max(T_field)
            max_C = np.max(C_field)
            avg_T = np.mean(T_field)
            total_C = np.sum(C_field)
            print(f"  Step {step:3d}: T_max={max_T:.1f}°C, C_max={max_C:.3f}, T_avg={avg_T:.1f}°C, C_total={total_C:.1f}")
    
    print(f"Multi-physics coupling completed successfully!")
    
    return T_field.reshape(rows, cols), C_field.reshape(rows, cols)

def visualize_results(results_dict):
    """Create visualizations of the simulation results."""
    print("\n=== Visualization ===")
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('sympy-hpx v4 Multi-Dimensional Simulation Results', fontsize=16)
        
        # Heat diffusion
        if 'heat' in results_dict:
            im1 = axes[0,0].imshow(results_dict['heat'], cmap='hot', interpolation='bilinear')
            axes[0,0].set_title('2D Heat Diffusion')
            axes[0,0].set_xlabel('x')
            axes[0,0].set_ylabel('y')
            plt.colorbar(im1, ax=axes[0,0])
        
        # Wave propagation
        if 'wave' in results_dict:
            im2 = axes[0,1].imshow(results_dict['wave'], cmap='RdBu', interpolation='bilinear')
            axes[0,1].set_title('2D Wave Propagation')
            axes[0,1].set_xlabel('x')
            axes[0,1].set_ylabel('y')
            plt.colorbar(im2, ax=axes[0,1])
        
        # 3D diffusion (central slice)
        if 'diffusion_3d' in results_dict:
            im3 = axes[0,2].imshow(results_dict['diffusion_3d'], cmap='viridis', interpolation='bilinear')
            axes[0,2].set_title('3D Diffusion (Central Slice)')
            axes[0,2].set_xlabel('x')
            axes[0,2].set_ylabel('y')
            plt.colorbar(im3, ax=axes[0,2])
        
        # Multi-physics: Temperature
        if 'temperature' in results_dict:
            im4 = axes[1,0].imshow(results_dict['temperature'], cmap='hot', interpolation='bilinear')
            axes[1,0].set_title('Coupled: Temperature')
            axes[1,0].set_xlabel('x')
            axes[1,0].set_ylabel('y')
            plt.colorbar(im4, ax=axes[1,0])
        
        # Multi-physics: Concentration
        if 'concentration' in results_dict:
            im5 = axes[1,1].imshow(results_dict['concentration'], cmap='Blues', interpolation='bilinear')
            axes[1,1].set_title('Coupled: Concentration')
            axes[1,1].set_xlabel('x')
            axes[1,1].set_ylabel('y')
            plt.colorbar(im5, ax=axes[1,1])
        
        # Summary plot
        axes[1,2].text(0.1, 0.8, 'sympy-hpx v4 Features:', fontsize=12, fontweight='bold')
        axes[1,2].text(0.1, 0.7, '✓ Multi-dimensional arrays', fontsize=10)
        axes[1,2].text(0.1, 0.6, '✓ Multi-dimensional stencils', fontsize=10)
        axes[1,2].text(0.1, 0.5, '✓ Multi-equation systems', fontsize=10)
        axes[1,2].text(0.1, 0.4, '✓ Multi-physics coupling', fontsize=10)
        axes[1,2].text(0.1, 0.3, '✓ 2D/3D computations', fontsize=10)
        axes[1,2].text(0.1, 0.2, '✓ Backward compatibility', fontsize=10)
        axes[1,2].set_xlim(0, 1)
        axes[1,2].set_ylim(0, 1)
        axes[1,2].axis('off')
        
        plt.tight_layout()
        plt.savefig('sympy_hpx_v4_results.png', dpi=150, bbox_inches='tight')
        print("✓ Results saved to 'sympy_hpx_v4_results.png'")
        
    except ImportError:
        print("Matplotlib not available - skipping visualization")
        print("Install with: pip install matplotlib")

def main():
    """Run all advanced demos."""
    print("🚀 sympy-hpx v4 Advanced Multi-Dimensional Demos")
    print("=" * 60)
    
    results = {}
    
    try:
        # Demo 1: 2D Heat Diffusion
        heat_result = demo_2d_heat_simulation()
        results['heat'] = heat_result
        
        # Demo 2: 2D Wave Propagation
        wave_result = demo_2d_wave_propagation()
        results['wave'] = wave_result
        
        # Demo 3: 3D Diffusion
        diffusion_result = demo_3d_diffusion()
        results['diffusion_3d'] = diffusion_result
        
        # Demo 4: Multi-Physics Coupling
        temp_result, conc_result = demo_multi_physics_coupling()
        results['temperature'] = temp_result
        results['concentration'] = conc_result
        
        # Visualization
        visualize_results(results)
        
        print("\n" + "=" * 60)
        print("🎉 All advanced demos completed successfully!")
        print("\nsympy-hpx v4 demonstrates:")
        print("  • Multi-dimensional array processing (1D, 2D, 3D)")
        print("  • Complex stencil operations across dimensions")
        print("  • Multi-equation systems with inter-dependencies")
        print("  • Multi-physics coupling scenarios")
        print("  • High-performance scientific computing applications")
        print("  • Seamless integration with NumPy and visualization tools")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 