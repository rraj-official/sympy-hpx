# sympy-hpx v2: Stencil Operations

This version extends the basic code generation to handle stencil patterns commonly used in numerical computations.

## Overview

sympy-hpx v2 enhances the SymPy code generation to handle stencil patterns commonly used in numerical computations, finite difference methods, and signal processing.

## Key Features

- **Stencil Pattern Support**: Handle expressions with offset indices like `a[i+1]`, `b[i-2]`
- **Automatic Bounds Calculation**: Generate safe loop bounds based on stencil offsets
- **Optimized C++ Code**: Generate efficient loops with proper boundary handling
- **Multiple Stencil Types**: Support forward, backward, and symmetric stencil patterns

## Basic Usage

```python
from sympy import *
from sympy_codegen import genFunc
import numpy as np

# Define symbolic variables
i = Idx("i")
a = IndexedBase("a")
b = IndexedBase("b")
c = IndexedBase("c")
r = IndexedBase("r")
d = Symbol("d")  # scalar

# Create stencil equation: r[i] = d*a[i] + b[i+1]*c[i-2]
equation = Eq(r[i], d*a[i] + b[i+1]*c[i-2])

# Generate compiled function
a_bc = genFunc(equation)

# Use with NumPy arrays
va = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
vb = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
vc = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
vr = np.zeros(8)
d_val = 2.0

# Call the generated function
a_bc(vr, va, vb, vc, d_val)
```

## Generated C++ Code Structure

For the stencil equation `r[i] = d*a[i] + b[i+1]*c[i-2]`, the generated C++ code is:

```cpp
#include <vector>
#include <cassert>
#include <cmath>

extern "C" {
void cpp_stencil_12345678(std::vector<double>& vr,
                         const std::vector<double>& va,
                         const std::vector<double>& vb,
                         const std::vector<double>& vc,
                         const double& sd)
{
    const int n = va.size();
    assert(n == vb.size());
    assert(n == vc.size());
    assert(n == vr.size());

    const int min_index = 2; // from stencil pattern (i-2 >= 0)
    const int max_index = n - 1; // from stencil pattern (i+1 < n)

    // Generated stencil loop
    for(int i = min_index; i < max_index; i++) {
        vr[i] = sd*va[i] + vb[i + 1]*vc[i - 2];
    }
}
}
```

## Stencil Bounds Calculation

The system automatically analyzes offset patterns and calculates safe loop bounds:

- **Min offset analysis**: For `i-2`, minimum safe index is `2`
- **Max offset analysis**: For `i+1`, maximum safe index is `n-2` (loop `i < n-1`)
- **Boundary handling**: Elements outside the valid range remain unchanged (typically zero)

## Supported Stencil Patterns

### 1. Forward Stencil
```python
# r[i] = a[i+1]
equation = Eq(r[i], a[i+1])
# Valid range: i = 0 to n-2
```

### 2. Backward Stencil  
```python
# r[i] = a[i-1]
equation = Eq(r[i], a[i-1])
# Valid range: i = 1 to n-1
```

### 3. Symmetric Stencil
```python
# r[i] = a[i-1] + a[i] + a[i+1] (3-point stencil)
equation = Eq(r[i], a[i-1] + a[i] + a[i+1])
# Valid range: i = 1 to n-2
```

### 4. Complex Stencil
```python
# r[i] = d*a[i] + b[i+1]*c[i-2] (mixed offsets)
equation = Eq(r[i], d*a[i] + b[i+1]*c[i-2])
# Valid range: i = 2 to n-2
```

### 5. No Stencil (Regular)
```python
# r[i] = d*a[i] + b[i] (same as v1)
equation = Eq(r[i], d*a[i] + b[i])
# Valid range: i = 0 to n-1
```

## Examples

### Example 1: Requirements Stencil
```python
from sympy import *
import numpy as np
from sympy_codegen import genFunc

i = Idx("i")
a = IndexedBase("a")
b = IndexedBase("b")
c = IndexedBase("c")
r = IndexedBase("r")
d = Symbol("d")

# Stencil equation from requirements
a_bc = genFunc(Eq(r[i], d*a[i] + b[i+1]*c[i-2]))

# Test data
va = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
vb = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
vc = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
vr = np.zeros(5)

a_bc(vr, va, vb, vc, 2.0)
# Result: vr = [0.0, 0.0, 8.0, 14.0, 0.0]
# Only indices 2 and 3 are computed (valid stencil range)
```

### Example 2: Finite Difference
```python
# Central difference approximation: f'[i] ≈ (f[i+1] - f[i-1]) / (2*h)
h = Symbol("h")
f = IndexedBase("f")
df = IndexedBase("df")

derivative_func = genFunc(Eq(df[i], (f[i+1] - f[i-1]) / (2*h)))

# Apply to data
x = np.linspace(0, 2*np.pi, 10)
f_vals = np.sin(x)
df_vals = np.zeros(10)

derivative_func(df_vals, f_vals, 0.1)  # h = 0.1
```

## Files

- `sympy_codegen.py`: Enhanced implementation with stencil support
- `example.py`: Demonstration of requirements stencil pattern
- `test_script.py`: Comprehensive test suite for various stencil patterns
- `README.md`: This documentation

## Running the Examples

```bash
cd sympy-hpx/v2
source ../../venv/bin/activate

# Run the requirements example
python3 example.py

# Run comprehensive tests
python3 test_script.py
```

## Improvements over v1

- ✅ **Stencil Support**: Handle offset indices like `i+1`, `i-2`
- ✅ **Automatic Bounds**: Calculate safe loop bounds from stencil patterns
- ✅ **Boundary Safety**: Prevent out-of-bounds array access
- ✅ **Multiple Patterns**: Support various stencil configurations
- ✅ **Backward Compatibility**: Still works with regular (non-stencil) expressions

## Technical Implementation

### Core Logic Overview

sympy-hpx v2 extends v1 with sophisticated stencil pattern analysis and boundary handling:

#### 1. Stencil Pattern Analysis (`StencilInfo` class)
- **Purpose**: Analyze index expressions to identify stencil access patterns
- **Key Methods**:
  - `add_offset(offset)`: Records index offsets like +1, -2
  - `get_loop_bounds(n)`: Calculates safe iteration bounds
- **Pattern Detection**: Parses expressions like `i+1`, `i-2` to extract offsets
- **Bounds Calculation**: 
  ```python
  min_index = max(0, -min_offset)  # Ensure i-offset >= 0
  max_index = n - max_offset       # Ensure i+offset < n
  ```

#### 2. Enhanced Expression Analysis (`_analyze_expression`)
- **Extends v1**: Builds on basic variable identification
- **Stencil Detection**: Finds all indexed expressions with offsets
- **Offset Parsing**: Uses `_parse_index_offset()` to handle complex index expressions
- **Index Types Supported**:
  - Simple: `i` → offset 0
  - Forward: `i+1`, `i+2` → positive offsets
  - Backward: `i-1`, `i-2` → negative offsets
  - Complex: `i+3-1` → parsed to net offset

#### 3. Advanced C++ Code Generation (`_generate_cpp_code`)
- **Stencil-Aware Loops**: Generates bounds-safe iteration
  ```cpp
  const int min_index = 2;    // from i-2
  const int max_index = n-1;  // from i+1
  for(int i = min_index; i < max_index; i++) {
      vr[i] = va[i] + vb[i+1]*vc[i-2];  // Safe stencil access
  }
  ```
- **Boundary Safety**: Automatically prevents out-of-bounds access
- **Offset Translation**: Converts SymPy offsets to C++ array indexing

#### 4. Stencil Expression Translation (`_convert_expression_to_cpp`)
- **Pattern Matching**: Uses regex to find indexed expressions
- **Offset Handling**: Translates `a[i+1]` → `va[i + 1]`
- **Complex Patterns**: Supports multiple offsets in single expression
- **Safety Checks**: Ensures all accesses are within computed bounds

### Key Algorithmic Innovations

#### Stencil Bounds Algorithm
```python
def calculate_bounds(stencil_pattern):
    min_offset = min(all_offsets)  # Most negative offset
    max_offset = max(all_offsets)  # Most positive offset
    
    # Safe bounds ensuring all accesses are valid
    min_index = max(0, -min_offset)
    max_index = n - max_offset if max_offset > 0 else n
    
    return min_index, max_index
```

#### Index Offset Parser
- **Handles Addition**: `i+3` → offset +3
- **Handles Subtraction**: `i-2` → offset -2  
- **Handles Complex**: `i+5-2` → offset +3
- **Fallback**: Unknown patterns → offset 0 (safe default)

### Stencil Pattern Examples

1. **Forward Stencil**: `b[i+1]`
   - Offset: +1
   - Bounds: `i` from 0 to `n-2`

2. **Backward Stencil**: `a[i-1]`
   - Offset: -1
   - Bounds: `i` from 1 to `n-1`

3. **Symmetric Stencil**: `a[i-1] + a[i] + a[i+1]`
   - Offsets: -1, 0, +1
   - Bounds: `i` from 1 to `n-2`

4. **Complex Stencil**: `b[i+1]*c[i-2]`
   - Offsets: +1, -2
   - Bounds: `i` from 2 to `n-2`

### Performance Optimizations

- **Bounds Pre-calculation**: Computed once, used for entire loop
- **Offset Analysis**: Cached stencil patterns avoid recomputation
- **Safe Indexing**: No runtime bounds checking needed in generated code
- **Compiler Optimization**: Generated C++ code optimizes well with `-O3`

## Generated Code Inspection

All generated C++ code is automatically saved to files with names like `generated_cpp_stencil_12345678.cpp` in the current directory. The stencil-aware code shows:
- Calculated loop bounds based on stencil pattern
- Safe array access patterns
- Optimized loop structure for performance

## Limitations (v2)

- Uses direct computation fallback instead of full ctypes integration
- Limited to integer constant offsets (no variable offsets)
- Basic error handling for complex stencil patterns

## Future Versions

- v3: Multiple equations processed together
- v4: Advanced optimizations and multi-dimensional stencils 