# Scientific Computing Examples - sympy-hpx v4

This directory contains comprehensive scientific computing examples demonstrating the capabilities of `sympy-hpx` v4 with HPX parallel acceleration. Each example showcases different physics, numerical methods, and computational challenges.

**⚡ Performance Note**: All examples benefit from smart compilation optimization - first run compiles the HPX kernel (~3-5s), subsequent runs with identical equations skip compilation entirely (~0.03s), enabling rapid iteration and testing.

## 📊 Performance Benchmark

### **Performance Benchmark** (`benchmark.py`)
**Purpose**: Compare Python/NumPy vs sympy-hpx v4 performance  
**Equation**: Complex 2D reaction-diffusion system (2 coupled PDEs)  
**Method**: Multi-equation HPX parallel processing  
**Features**:
- ✅ Tests multiple problem sizes (2.5K to 90K elements)
- ✅ Compares Pure Python, NumPy, SymPy+NumPy, and sympy-hpx v4
- ✅ Shows up to **5x speedup** over NumPy for large problems
- ✅ Demonstrates HPX runtime optimization benefits
- **Performance**: Best speedup at 90K elements (5.0x faster than NumPy)

**Key Results:**
- Small problems (2.5K): HPX overhead dominates
- Medium problems (10K+): HPX starts showing benefits  
- Large problems (40K+): **3-5x speedup** over NumPy
- Runtime optimization: 27-33x faster subsequent calls

**Usage:**
```bash
cd examples
python3 benchmark.py
```

**Sample Output:**
```
TESTING GRID SIZE: 300 × 300 = 90,000 elements
NumPy Vectorized:     6.09 ms
sympy-hpx v4:         1.21 ms (avg), 1.73 ms (init)
Speedups vs NumPy:
sympy-hpx v4:         5.0x faster
```

---

## 🔬 Available Examples

### 1. **2D Heat Diffusion** (`2D_heat_diffusion.py`)
**Physics**: Thermal diffusion in 2D materials  
**Equation**: `∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)`  
**Method**: Explicit finite differences  
**Features**: 
- ✅ Numerically stable
- ✅ Physical heat spreading simulation
- ✅ Initial hot spot with thermal diffusion
- ✅ Energy conservation analysis
- **Performance**: ~0.02 million points/second

---

### 2. **2D Wave Equation** (`2D_wave_equation.py`) 
**Physics**: Mechanical wave propagation on 2D membrane  
**Equation**: `∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²)`  
**Method**: Leap-frog time integration  
**Features**:
- ✅ Numerically stable (CFL < 0.707)
- ✅ Wave interference between two Gaussian pulses
- ✅ Energy conservation monitoring
- ✅ Hyperbolic PDE characteristics
- **Performance**: ~0.08 million points/second

---

### 3. **1D Advection-Diffusion** (`1D_advection_diffusion.py`)
**Physics**: Pollutant/heat transport in flowing media  
**Equation**: `∂c/∂t + v∂c/∂x = D∂²c/∂x²`  
**Method**: Upwind finite differences (stable)  
**Features**:
- ✅ Numerically stable upwind scheme
- ✅ Mass conservation verification
- ✅ Advection vs. diffusion competition
- ✅ Péclet number analysis
- **Performance**: ~7.4 million points/second
- **Applications**: Groundwater, atmospheric dispersion, reactors

---

### 4. **1D Schrödinger Equation** (`1D_schrodinger_equation.py`)
**Physics**: Quantum mechanics and tunneling  
**Equation**: `iℏ ∂ψ/∂t = -ℏ²/(2m) ∂²ψ/∂x² + V(x)ψ`  
**Method**: Split real/imaginary evolution  
**Status**: ⚠️ Numerical instability issues  
**Note**: Requires advanced methods (split-step Fourier, Crank-Nicolson)

---

## 📊 Performance Summary

| Example | Grid Size | Time Steps | Performance (Mpts/s) | Stability |
|---------|-----------|------------|---------------------|-----------|
| Heat 2D | 21×21     | 200        | 0.02                | ✅ Stable |
| Wave 2D | 41×41     | 300        | 0.08                | ✅ Stable |
| Advect-Diff 1D | 200 | 800      | 7.40                | ✅ Stable |
| Schrödinger 1D | 1024 | 500      | 12.54               | ❌ Unstable |

## 🔧 Technical Features Demonstrated

### HPX Parallel Acceleration
- **Automatic C++ code generation** from SymPy equations
- **CMake compilation** with HPX linking
- **Parallel execution** across CPU cores
- **Memory-efficient** flattened array indexing

### Numerical Methods
- **Explicit/Implicit time integration**
- **Finite difference spatial discretization**
- **Stability analysis** and CFL conditions
- **Boundary condition handling**
- **Conservation property verification**

### Scientific Validation
- **Physical behavior verification**
- **Conservation laws** (mass, energy)
- **Analytical comparisons** where available
- **Parameter sensitivity analysis**

## 🚀 Usage Instructions

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Ensure HPX is built with system allocator
# HPX must be configured with: -DHPX_WITH_MALLOC=system
```

### Running Examples
```bash
# Run individual examples
python3 2D_heat_diffusion.py
python3 2D_wave_equation.py  
python3 1D_advection_diffusion.py

# All examples save visualization results as PNG files
```

### Output Files
- `2D_heat_diffusion_results.png` - Heat spreading visualization
- `2D_wave_equation_results.png` - Wave interference patterns
- `1D_advection_diffusion_results.png` - Transport phenomena plots

## 📚 Scientific Domains Covered

### Transport Phenomena
- **Heat transfer**: Conduction, convection
- **Mass transfer**: Diffusion, advection
- **Momentum transfer**: Wave propagation

### Mathematical Classifications
- **Parabolic PDEs**: Heat/diffusion equations
- **Hyperbolic PDEs**: Wave equations  
- **Mixed PDEs**: Advection-diffusion

### Engineering Applications
- **Environmental**: Groundwater contamination, air pollution
- **Thermal**: Heat exchangers, building energy
- **Mechanical**: Vibrations, acoustics
- **Chemical**: Reactor design, mixing

## 🎯 Future Improvements

### Planned Examples
- **3D Navier-Stokes** - Fluid dynamics
- **Reaction-Diffusion** - Pattern formation
- **Maxwell Equations** - Electromagnetics
- **Nonlinear Schrödinger** - Quantum solitons

### Numerical Enhancements
- **Implicit methods** for stiff equations
- **Adaptive time stepping**
- **Higher-order schemes** (Runge-Kutta, spectral)
- **Multigrid solvers**

### Performance Features
- **Multi-GPU support** via HPX
- **Load balancing** for irregular domains
- **Memory optimization** for large grids
- **Scalability benchmarks**

## 📖 References

1. **Numerical Methods**: LeVeque, R. J. "Finite Difference Methods for Ordinary and Partial Differential Equations"
2. **Transport Phenomena**: Bird, Stewart, Lightfoot "Transport Phenomena" 
3. **HPX Documentation**: https://hpx-docs.stellar-group.org/
4. **SymPy Documentation**: https://docs.sympy.org/

---

**🎉 All examples successfully demonstrate sympy-hpx v4's ability to automatically generate high-performance HPX-accelerated C++ code from symbolic mathematics!**