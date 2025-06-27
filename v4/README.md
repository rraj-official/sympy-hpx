# sympy-hpx v4 - Multi-Dimensional Support

This version extends v3 with comprehensive support for multi-dimensional arrays (2D and 3D) while maintaining full backward compatibility.

## Key Features

- **Multi-Dimensional Arrays**: Support for 1D, 2D, and 3D arrays with automatic indexing
- **Multi-Dimensional Stencils**: Handle stencil patterns across multiple dimensions
- **Flattened Array Storage**: Efficient memory layout using row-major flattened arrays
- **Unified Multi-Equation Processing**: Process multiple equations together across dimensions
- **Backward Compatibility**: Full compatibility with v1, v2, and v3 functionality
- **Automatic Bounds Calculation**: Safe multi-dimensional stencil bounds

## Basic Usage

### 2D Heat Diffusion Example

```python
from sympy import *
from sympy_codegen import genFunc
import numpy as np

# Create 2D symbols
i, j = symbols('i j', integer=True)
T = IndexedBase("T")          # Temperature field (2D)
T_new = IndexedBase("T_new")  # New temperature (2D)
alpha = Symbol("alpha")       # Diffusion coefficient
dt = Symbol("dt")             # Time step
dx = Symbol("dx")             # Grid spacing

# 2D heat diffusion with 5-point stencil
heat_eq = Eq(T_new[i,j], T[i,j] + alpha*dt/(dx**2) * (
    T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1] - 4*T[i,j]
))

# Generate function
heat_func = genFunc(heat_eq)

# Prepare 2D data (stored as flattened arrays)
rows, cols = 10, 12
T_field = np.zeros(rows * cols)      # Flattened 2D array
T_new_field = np.zeros(rows * cols)

# Initialize temperature field
# ... set initial conditions ...

# Call function with shape parameters
heat_func(T_new_field, T_field, rows, cols, alpha_val, dt_val, dx_val)
```

### 2D Multi-Equation Gradient System

```python
# Multi-equation 2D gradient calculation
i, j = symbols('i j', integer=True)
u = IndexedBase("u")
grad_x = IndexedBase("grad_x")
grad_y = IndexedBase("grad_y")
grad_mag = IndexedBase("grad_mag")
dx = Symbol("dx")

equations = [
    Eq(grad_x[i,j], (u[i+1,j] - u[i-1,j]) / (2*dx)),           # x-gradient
    Eq(grad_y[i,j], (u[i,j+1] - u[i,j-1]) / (2*dx)),           # y-gradient
    Eq(grad_mag[i,j], (grad_x[i,j]**2 + grad_y[i,j]**2)**0.5)  # magnitude
]

grad_func = genFunc(equations)

# Call with multiple result arrays
grad_func(grad_x_field, grad_y_field, grad_mag_field, u_field, rows, cols, dx_val)
```

## Function Signature Convention

### 2D Arrays
```python
func(result_arrays..., input_arrays..., rows, cols, scalar_params...)
```

### 3D Arrays
```python
func(result_arrays..., input_arrays..., rows, cols, depth, scalar_params...)
```

### 1D Arrays (backward compatible)
```python
func(result_arrays..., input_arrays..., scalar_params...)
```

## Multi-Dimensional Array Storage

Arrays are stored as **flattened 1D NumPy arrays** using row-major ordering:

### 2D Array Indexing
```python
# For 2D array A[i,j] with shape (rows, cols):
flat_index = i * cols + j
value = A_flat[flat_index]
```

### 3D Array Indexing
```python
# For 3D array A[i,j,k] with shape (rows, cols, depth):
flat_index = i * cols * depth + j * depth + k
value = A_flat[flat_index]
```

## Generated C++ Code Structure

### 2D Example
```cpp
#include <vector>
#include <cassert>
#include <cmath>

extern "C" {
void cpp_multidim_12345678(std::vector<double>& vT_new,
                           const std::vector<double>& vT,
                           const int& rows,
                           const int& cols,
                           const double& salpha,
                           const double& sdt,
                           const double& sdx)
{
    // Multi-dimensional array handling
    const int total_size = rows * cols;
    assert(vT_new.size() == total_size);
    assert(vT.size() == total_size);

    // Multi-dimensional stencil bounds
    const int min_i = 1;      // from i-1 stencil
    const int max_i = rows - 1; // from i+1 stencil
    const int min_j = 1;      // from j-1 stencil
    const int max_j = cols - 1; // from j+1 stencil

    // Generated multi-dimensional loop
    for(int i = min_i; i < max_i; i++) {
        for(int j = min_j; j < max_j; j++) {
            vT_new[i * cols + j] = vT[i * cols + j] + salpha*sdt/pow(sdx, 2) * 
                (vT[(i + 1) * cols + j] + vT[(i - 1) * cols + j] + 
                 vT[i * cols + (j + 1)] + vT[i * cols + (j - 1)] - 
                 4*vT[i * cols + j]);
        }
    }
}
}
```

## Stencil Pattern Support

### 2D Stencil Patterns

#### 5-Point Stencil (Laplacian)
```python
Eq(result[i,j], u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j])
# Bounds: i ∈ [1, rows-2], j ∈ [1, cols-2]
```

#### 9-Point Stencil
```python
Eq(result[i,j], u[i-1,j-1] + u[i-1,j] + u[i-1,j+1] +
                u[i,j-1]   + u[i,j]   + u[i,j+1] +
                u[i+1,j-1] + u[i+1,j] + u[i+1,j+1])
# Bounds: i ∈ [1, rows-2], j ∈ [1, cols-2]
```

#### Gradient Stencils
```python
# Central difference in x-direction
Eq(grad_x[i,j], (u[i+1,j] - u[i-1,j]) / (2*dx))

# Central difference in y-direction  
Eq(grad_y[i,j], (u[i,j+1] - u[i,j-1]) / (2*dx))
```

### 3D Stencil Patterns

#### 7-Point Stencil (3D Laplacian)
```python
Eq(result[i,j,k], u[i+1,j,k] + u[i-1,j,k] + 
                  u[i,j+1,k] + u[i,j-1,k] + 
                  u[i,j,k+1] + u[i,j,k-1] - 6*u[i,j,k])
```

## Examples

### Example 1: 2D Wave Equation
```python
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

wave_func = genFunc(wave_eq)
```

### Example 2: 3D Diffusion
```python
i, j, k = symbols('i j k', integer=True)
phi = IndexedBase("phi")
phi_new = IndexedBase("phi_new")
D = Symbol("D")  # Diffusion coefficient

# 3D diffusion equation
diffusion_3d = Eq(phi_new[i,j,k], phi[i,j,k] + D*dt/(dx**2) * (
    phi[i+1,j,k] + phi[i-1,j,k] + 
    phi[i,j+1,k] + phi[i,j-1,k] + 
    phi[i,j,k+1] + phi[i,j,k-1] - 6*phi[i,j,k]
))

diffusion_func = genFunc(diffusion_3d)
```

### Example 3: Mixed Dimensions
```python
# Combine 2D field with 1D coefficients
i, j = symbols('i j', integer=True)
field_2d = IndexedBase("field_2d")  # 2D field
coeff_1d = IndexedBase("coeff_1d")  # 1D coefficients (varies with i only)
result = IndexedBase("result")

# Apply row-dependent scaling
mixed_eq = Eq(result[i,j], coeff_1d[i] * field_2d[i,j])

mixed_func = genFunc(mixed_eq)
# Call: mixed_func(result_2d, coeff_1d, field_2d, rows, cols)
```

## Technical Implementation

### Core Logic Overview

sympy-hpx v4 implements sophisticated multi-dimensional processing with unified analysis:

#### 1. Multi-Dimensional Analysis (`_analyze_equations`)
- **Dimensionality Detection**: Automatically determines array dimensions from index count
- **Variable Classification**: Separates 1D, 2D, 3D arrays and scalar parameters
- **Unified Stencil Analysis**: Merges stencil patterns across all dimensions
- **Shape Parameter Management**: Handles rows, cols, depth parameters

#### 2. Multi-Dimensional Stencil Analysis (`MultiDimStencilInfo`)
- **Per-Dimension Offsets**: Tracks stencil offsets for each dimension separately
- **Unified Bounds Calculation**: Computes safe loop bounds considering all dimensions
- **Stencil Pattern Merging**: Combines different dimensional patterns safely

#### 3. Enhanced C++ Code Generation
- **Nested Loop Generation**: Creates appropriate nested loops for each dimension
- **Flattened Index Calculation**: Converts multi-dimensional indices to flat array access
- **Shape Parameter Handling**: Includes rows, cols, depth in function signature
- **Multi-Dimensional Bounds**: Generates bounds-safe loops for each dimension

#### 4. Array Indexing Algorithms

##### 2D Indexing Conversion
```python
def convert_2d_to_flat(i_expr, j_expr, cols):
    return f"({i_expr}) * cols + ({j_expr})"
```

##### 3D Indexing Conversion
```python
def convert_3d_to_flat(i_expr, j_expr, k_expr, cols, depth):
    return f"({i_expr}) * cols * depth + ({j_expr}) * depth + ({k_expr})"
```

#### 5. Multi-Dimensional Bounds Algorithm
```python
def calculate_multidim_bounds(stencil_info, dimensions):
    bounds = []
    for dim in range(dimensions):
        min_offset = stencil_info.min_offsets[dim]
        max_offset = stencil_info.max_offsets[dim]
        
        min_bound = max(0, -min_offset)
        max_bound = f"shape[{dim}] - {max_offset}" if max_offset > 0 else f"shape[{dim}]"
        
        bounds.append((min_bound, max_bound))
    return bounds
```

### Key Algorithmic Innovations

#### 1. Automatic Dimensionality Detection
```python
def detect_array_dimensions(indexed_expressions):
    dimensions = {}
    for indexed in indexed_expressions:
        array_name = str(indexed.base)
        num_indices = len(indexed.indices)
        dimensions[array_name] = num_indices
    return dimensions
```

#### 2. Unified Multi-Dimensional Stencil Analysis
```python
def analyze_multidim_stencils(equations):
    max_dim = determine_max_dimensions(equations)
    unified_stencil = MultiDimStencilInfo(max_dim)
    
    for eq in equations:
        for indexed in eq.atoms(sp.Indexed):
            for dim, idx_expr in enumerate(indexed.indices):
                offset = parse_index_offset(idx_expr)
                unified_stencil.add_offset(dim, offset)
    
    return unified_stencil
```

#### 3. Flattened Array Access Pattern Generation
```python
def generate_flat_access(var_name, indices, array_dims):
    if array_dims[var_name] == 1:
        return f"v{var_name}[{indices[0]}]"
    elif array_dims[var_name] == 2:
        return f"v{var_name}[({indices[0]}) * cols + ({indices[1]})]"
    elif array_dims[var_name] == 3:
        return f"v{var_name}[({indices[0]}) * cols * depth + ({indices[1]}) * depth + ({indices[2]})]"
```

### Performance Characteristics

- **Setup Time**: ~150-300ms for multi-dimensional analysis and compilation
- **Memory Layout**: Optimal cache performance with row-major storage
- **Stencil Efficiency**: Bounds-safe access without runtime checks
- **Multi-Equation Optimization**: Single nested loop for multiple calculations
- **Scalability**: O(N^d) where N is grid size and d is dimensionality

## Generated Code Inspection

All generated C++ code is saved to files with names like `generated_cpp_multidim_12345678.cpp`. The multi-dimensional code shows:
- Nested loop structures for each dimension
- Flattened array indexing calculations
- Unified stencil bounds for all dimensions
- Multi-equation processing within inner loops

## Files

- `sympy_codegen.py`: Enhanced multi-dimensional implementation
- `example.py`: Comprehensive 2D and 3D examples
- `test_script.py`: Multi-dimensional test suite
- `README.md`: This documentation

## Running the Examples

```bash
cd sympy-hpx/v4
source ../../venv/bin/activate

# Run multi-dimensional examples
python3 example.py

# Run comprehensive tests
python3 test_script.py
```

## Improvements over v3

- ✅ **Multi-Dimensional Arrays**: Support for 2D and 3D arrays
- ✅ **Multi-Dimensional Stencils**: Handle complex stencil patterns across dimensions
- ✅ **Flattened Storage**: Efficient memory layout for multi-dimensional data
- ✅ **Unified Processing**: Multi-equation support across all dimensions
- ✅ **Automatic Indexing**: Seamless conversion between multi-dimensional and flat indices
- ✅ **Backward Compatibility**: Full compatibility with v1, v2, v3

## Applications

### Scientific Computing
- **Finite Difference Methods**: 2D/3D PDEs on structured grids
- **Image Processing**: 2D convolution and filtering operations
- **Computational Fluid Dynamics**: Multi-dimensional flow simulations
- **Heat Transfer**: 2D/3D thermal diffusion problems

### Numerical Methods
- **Laplace/Poisson Solvers**: Multi-dimensional elliptic PDEs
- **Wave Propagation**: 2D/3D wave equation simulations
- **Reaction-Diffusion**: Multi-species systems on 2D/3D domains
- **Level Set Methods**: Interface tracking in multiple dimensions

## Limitations (v4)

- Limited to rectangular/cubic grids (no irregular meshes)
- Maximum 3 dimensions currently supported
- Uses direct computation fallback (full ctypes integration pending)
- Memory usage scales as O(N^d) for d-dimensional problems

## Future Enhancements

- **Irregular Meshes**: Support for unstructured grids
- **Higher Dimensions**: Extension to 4D+ arrays
- **GPU Acceleration**: CUDA/OpenCL code generation
- **Adaptive Grids**: Dynamic mesh refinement support
- **Parallel Processing**: Multi-threaded execution for large grids

## Next Steps

This multi-dimensional foundation enables:
- High-performance PDE solvers
- Advanced scientific simulations
- Real-time computational applications
- Integration with existing numerical libraries 