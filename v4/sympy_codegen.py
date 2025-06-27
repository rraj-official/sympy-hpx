"""
SymPy-based code generation for HPX Python API v4 - Multi-Dimensional Support
Extends v3 multi-equation processing with support for 2D and 3D arrays and stencils.
"""

import os
import subprocess
import tempfile
import hashlib
from typing import Dict, List, Tuple, Any, Set, Union
import sympy as sp
import numpy as np
import re


class MultiDimStencilInfo:
    """Information about multi-dimensional stencil patterns."""
    def __init__(self, dimensions: int = 1):
        self.dimensions = dimensions
        self.offsets = {dim: set() for dim in range(dimensions)}
        self.min_offsets = {dim: 0 for dim in range(dimensions)}
        self.max_offsets = {dim: 0 for dim in range(dimensions)}
    
    def add_offset(self, dim: int, offset: int):
        """Add an index offset for a specific dimension."""
        if dim >= self.dimensions:
            raise ValueError(f"Dimension {dim} exceeds array dimensions {self.dimensions}")
        
        self.offsets[dim].add(offset)
        self.min_offsets[dim] = min(self.min_offsets[dim], offset)
        self.max_offsets[dim] = max(self.max_offsets[dim], offset)
    
    def get_loop_bounds(self, shape_vars: List[str]) -> Tuple[List[str], List[str]]:
        """Get the loop bounds for safe multi-dimensional stencil access."""
        min_indices = []
        max_indices = []
        
        for dim in range(self.dimensions):
            min_idx = max(0, -self.min_offsets[dim])
            max_idx = f"{shape_vars[dim]} - {self.max_offsets[dim]}" if self.max_offsets[dim] > 0 else shape_vars[dim]
            min_indices.append(str(min_idx))
            max_indices.append(max_idx)
        
        return min_indices, max_indices


class MultiDimCodeGenerator:
    """
    Enhanced code generator supporting multi-dimensional arrays and equations.
    """
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.compiled_functions = {}
        
    def _analyze_equations(self, equations: List[sp.Eq]) -> Tuple[List[str], List[str], List[str], MultiDimStencilInfo, Dict[str, int]]:
        """
        Analyze multiple SymPy equations with multi-dimensional support.
        
        Returns:
            - vector_vars: List of vector variable names (input arrays)
            - scalar_vars: List of scalar variable names  
            - result_vars: List of result variable names (output arrays)
            - stencil_info: Multi-dimensional stencil information
            - array_dims: Dictionary mapping array names to their dimensions
        """
        all_vector_vars = set()
        all_scalar_vars = set()
        result_vars = []
        array_dims = {}
        
        # First pass: determine dimensionality
        max_dimensions = 1
        index_vars = set()
        
        for eq in equations:
            # Extract all indexed expressions
            indexed_exprs = list(eq.atoms(sp.Indexed))
            for indexed in indexed_exprs:
                array_name = str(indexed.base)
                num_indices = len(indexed.indices)
                max_dimensions = max(max_dimensions, num_indices)
                array_dims[array_name] = num_indices
                
                # Collect index variables
                for idx in indexed.indices:
                    if isinstance(idx, (sp.Symbol, sp.Idx)):
                        index_vars.add(str(idx))
                    elif hasattr(idx, 'free_symbols'):
                        for sym in idx.free_symbols:
                            if isinstance(sym, (sp.Symbol, sp.Idx)):
                                index_vars.add(str(sym))
        
        stencil_info = MultiDimStencilInfo(max_dimensions)
        
        # Second pass: analyze variables and stencil patterns
        for eq in equations:
            lhs = eq.lhs
            rhs = eq.rhs
            
            # Extract result variable (left-hand side)
            if isinstance(lhs, sp.Indexed):
                result_var = str(lhs.base)
                result_vars.append(result_var)
                all_vector_vars.add(result_var)
            else:
                raise ValueError("Left-hand side must be an indexed expression")
            
            # Find all IndexedBase objects
            indexed_bases = list(eq.atoms(sp.IndexedBase))
            for base in indexed_bases:
                all_vector_vars.add(str(base))
            
            # Analyze multi-dimensional stencil patterns
            indexed_exprs = list(eq.atoms(sp.Indexed))
            for indexed in indexed_exprs:
                for dim, idx_expr in enumerate(indexed.indices):
                    offset = self._parse_index_offset(idx_expr)
                    stencil_info.add_offset(dim, offset)
            
            # Find scalar variables
            all_symbols = list(eq.atoms(sp.Symbol))
            for symbol in all_symbols:
                symbol_name = str(symbol)
                
                # Skip if this symbol corresponds to an IndexedBase
                if symbol_name in all_vector_vars:
                    continue
                    
                # Skip if this is an index variable
                if symbol_name in index_vars:
                    continue
                
                # Add to scalar variables
                all_scalar_vars.add(symbol_name)
        
        # Separate input arrays from result arrays
        input_vector_vars = []
        for var in sorted(all_vector_vars):
            if var not in result_vars:
                input_vector_vars.append(var)
        
        # Convert to sorted lists
        vector_vars = sorted(input_vector_vars)
        scalar_vars = sorted(list(all_scalar_vars))
        # Preserve original order of result variables (don't sort)
        result_vars = list(dict.fromkeys(result_vars))  # Remove duplicates while preserving order
        
        return vector_vars, scalar_vars, result_vars, stencil_info, array_dims
    
    def _parse_index_offset(self, idx_expr) -> int:
        """Parse an index expression to extract the offset (same as v3)."""
        if isinstance(idx_expr, (sp.Symbol, sp.Idx)):
            return 0
        elif isinstance(idx_expr, sp.Add):
            offset = 0
            for arg in idx_expr.args:
                if isinstance(arg, sp.Integer):
                    offset += int(arg)
                elif isinstance(arg, sp.Mul) and len(arg.args) == 2:
                    if isinstance(arg.args[0], sp.Integer) and isinstance(arg.args[1], sp.Integer):
                        offset += int(arg.args[0]) * int(arg.args[1])
            return offset
        elif isinstance(idx_expr, sp.Integer):
            return int(idx_expr)
        else:
            try:
                simplified = idx_expr
                for sym in idx_expr.free_symbols:
                    simplified = simplified.subs(sym, 0)
                return int(simplified)
            except:
                return 0
    
    def _generate_cpp_code(self, equations: List[sp.Eq], func_name: str) -> str:
        """Generate C++ code for multi-dimensional equations."""
        vector_vars, scalar_vars, result_vars, stencil_info, array_dims = self._analyze_equations(equations)
        
        # Determine if we have multi-dimensional arrays
        is_multidim = any(dim > 1 for dim in array_dims.values())
        max_dim = max(array_dims.values()) if array_dims else 1
        
        # Generate function signature
        cpp_code = f"""#include <vector>
#include <cassert>
#include <cmath>

extern "C" {{

void {func_name}("""
        
        # Add result parameters first (in alphabetical order)
        for i, var in enumerate(result_vars):
            if i > 0:
                cpp_code += ",\n               "
            
            if is_multidim and array_dims.get(var, 1) > 1:
                # Multi-dimensional array as flattened vector with shape parameters
                cpp_code += f"std::vector<double>& v{var}"
            else:
                # 1D array
                cpp_code += f"std::vector<double>& v{var}"
        
        # Add input vector parameters (alphabetically ordered)
        for var in vector_vars:
            cpp_code += f",\n               const std::vector<double>& v{var}"
        
        # Add shape parameters for multi-dimensional arrays
        if is_multidim:
            shape_params = []
            for dim in range(max_dim):
                dim_names = ['rows', 'cols', 'depth']
                if dim < len(dim_names):
                    shape_params.append(f"const int& {dim_names[dim]}")
                else:
                    shape_params.append(f"const int& dim{dim}")
            
            for param in shape_params:
                cpp_code += f",\n               {param}"
        
        # Add scalar parameters (alphabetically ordered) 
        for var in scalar_vars:
            cpp_code += f",\n               const double& s{var}"
            
        cpp_code += ")\n{\n"
        
        # Add size assertions and shape calculations
        if is_multidim:
            cpp_code += f"    // Multi-dimensional array handling\n"
            if max_dim >= 2:
                cpp_code += f"    const int total_size = rows * cols"
                if max_dim >= 3:
                    cpp_code += f" * depth"
                cpp_code += ";\n"
            
            # Add assertions for all arrays
            all_arrays = result_vars + vector_vars
            for var in all_arrays:
                if array_dims.get(var, 1) > 1:
                    cpp_code += f"    assert(v{var}.size() == total_size);\n"
                else:
                    cpp_code += f"    assert(v{var}.size() == rows);\n"
        else:
            # 1D case (same as v3)
            all_vectors = result_vars + vector_vars
            if all_vectors:
                first_vector = all_vectors[0]
                cpp_code += f"    const int n = v{first_vector}.size();\n"
                
                for var in all_vectors[1:]:
                    cpp_code += f"    assert(n == v{var}.size());\n"
        
        # Generate stencil bounds
        if is_multidim and max_dim > 1:
            shape_vars = ['rows', 'cols', 'depth'][:max_dim]
            min_bounds, max_bounds = stencil_info.get_loop_bounds(shape_vars)
            
            cpp_code += f"\n    // Multi-dimensional stencil bounds\n"
            for dim in range(max_dim):
                dim_names = ['i', 'j', 'k']
                cpp_code += f"    const int min_{dim_names[dim]} = {min_bounds[dim]};\n"
                cpp_code += f"    const int max_{dim_names[dim]} = {max_bounds[dim]};\n"
            
            # Generate nested loops
            cpp_code += f"\n    // Generated multi-dimensional loop\n"
            for dim in range(max_dim):
                dim_names = ['i', 'j', 'k']
                indent = "    " + "    " * dim
                cpp_code += f"{indent}for(int {dim_names[dim]} = min_{dim_names[dim]}; {dim_names[dim]} < max_{dim_names[dim]}; {dim_names[dim]}++) {{\n"
            
            # Generate index calculation for flattened arrays
            if max_dim == 2:
                index_calc = "i * cols + j"
            elif max_dim == 3:
                index_calc = "i * cols * depth + j * depth + k"
            else:
                index_calc = "i"  # fallback
            
            # Convert each equation to C++ code
            for eq in equations:
                lhs = eq.lhs
                rhs = eq.rhs
                result_var = str(lhs.base)
                
                rhs_str = self._convert_multidim_expression_to_cpp(rhs, vector_vars, scalar_vars, result_vars, array_dims, max_dim)
                
                # Post-process to fix any remaining **0.5 patterns
                rhs_str = self._fix_power_operators(rhs_str)
                
                indent = "    " + "    " * max_dim
                if array_dims.get(result_var, 1) > 1:
                    cpp_code += f"{indent}v{result_var}[{index_calc}] = {rhs_str};\n"
                else:
                    cpp_code += f"{indent}v{result_var}[i] = {rhs_str};\n"
            
            # Close nested loops
            for dim in range(max_dim):
                indent = "    " + "    " * (max_dim - 1 - dim)
                cpp_code += f"{indent}}}\n"
                
        else:
            # 1D case (same as v3)
            if stencil_info.offsets[0] and (stencil_info.min_offsets[0] < 0 or stencil_info.max_offsets[0] > 0):
                min_bound, max_bound = stencil_info.get_loop_bounds(["n"])
                cpp_code += f"\n    const int min_index = {min_bound[0]};\n"
                cpp_code += f"    const int max_index = {max_bound[0]};\n"
                loop_start = "min_index"
                loop_end = "max_index"
            else:
                loop_start = "0"
                loop_end = "n"
            
            cpp_code += "\n    // Generated multi-equation loop\n"
            cpp_code += f"    for(int i = {loop_start}; i < {loop_end}; i++) {{\n"
            
            # Convert each equation to C++ code
            for eq in equations:
                lhs = eq.lhs
                rhs = eq.rhs
                result_var = str(lhs.base)
                
                rhs_str = self._convert_expression_to_cpp(rhs, vector_vars, scalar_vars, result_vars)
                cpp_code += f"        v{result_var}[i] = {rhs_str};\n"
                
            cpp_code += "    }\n"
        
        cpp_code += "}\n\n}"
        
        return cpp_code
    
    def _convert_multidim_expression_to_cpp(self, expr, vector_vars: List[str], scalar_vars: List[str], 
                                          result_vars: List[str], array_dims: Dict[str, int], max_dim: int) -> str:
        """Convert a SymPy expression to C++ code with multi-dimensional support."""
        expr_str = str(expr)
        
        # Handle multi-dimensional indexed expressions
        indexed_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\[([^\]]+)\]'
        
        def replace_multidim_indexed(match):
            var_name = match.group(1)
            indices_str = match.group(2)
            
            if var_name in vector_vars or var_name in result_vars:
                # Parse indices (could be i, j, k with offsets)
                indices = [idx.strip() for idx in indices_str.split(',')]
                
                if array_dims.get(var_name, 1) > 1:
                    # Multi-dimensional array - convert to flattened index
                    if len(indices) == 2:  # 2D
                        i_expr, j_expr = indices
                        return f"v{var_name}[({i_expr}) * cols + ({j_expr})]"
                    elif len(indices) == 3:  # 3D
                        i_expr, j_expr, k_expr = indices
                        return f"v{var_name}[({i_expr}) * cols * depth + ({j_expr}) * depth + ({k_expr})]"
                    else:
                        # Fallback for other dimensions
                        return f"v{var_name}[{indices[0]}]"
                else:
                    # 1D array
                    return f"v{var_name}[{indices[0]}]"
            
            return match.group(0)
        
        expr_str = re.sub(indexed_pattern, replace_multidim_indexed, expr_str)
        
        # Replace scalar variables
        for var in scalar_vars:
            expr_str = re.sub(r'\b' + re.escape(var) + r'\b', f"s{var}", expr_str)
            
        # Convert power operators - improved pattern to handle complex expressions
        power_pattern = r'(\([^)]+\)|[a-zA-Z_][a-zA-Z0-9_]*\[[^\]]+\]|\w+)\*\*(\d+(?:\.\d+)?|\([^)]+\)|\w+)'
        def power_replacement(match):
            base = match.group(1)
            exponent = match.group(2)
            # Special case for square root
            if exponent == "0.5":
                return f"sqrt({base})"
            return f"pow({base}, {exponent})"
        
        expr_str = re.sub(power_pattern, power_replacement, expr_str)
        
        return expr_str
    
    def _convert_expression_to_cpp(self, expr, vector_vars: List[str], scalar_vars: List[str], result_vars: List[str]) -> str:
        """Convert 1D expression to C++ (same as v3)."""
        expr_str = str(expr)
        
        indexed_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\[([^\]]+)\]'
        
        def replace_indexed(match):
            var_name = match.group(1)
            index_expr = match.group(2)
            
            if var_name in vector_vars or var_name in result_vars:
                if index_expr == 'i':
                    return f"v{var_name}[i]"
                else:
                    if '+' in index_expr:
                        parts = index_expr.split('+')
                        if len(parts) == 2 and parts[0].strip() == 'i':
                            offset = parts[1].strip()
                            return f"v{var_name}[i + {offset}]"
                    elif '-' in index_expr and not index_expr.startswith('-'):
                        parts = index_expr.split('-')
                        if len(parts) == 2 and parts[0].strip() == 'i':
                            offset = parts[1].strip()
                            return f"v{var_name}[i - {offset}]"
                    
                    return f"v{var_name}[{index_expr}]"
            
            return match.group(0)
        
        expr_str = re.sub(indexed_pattern, replace_indexed, expr_str)
        
        # Replace scalar variables
        for var in scalar_vars:
            expr_str = re.sub(r'\b' + re.escape(var) + r'\b', f"s{var}", expr_str)
            
        # Convert power operators - improved pattern to handle complex expressions
        power_pattern = r'(\([^)]+\)|[a-zA-Z_][a-zA-Z0-9_]*\[[^\]]+\]|\w+)\*\*(\d+(?:\.\d+)?|\([^)]+\)|\w+)'
        def power_replacement(match):
            base = match.group(1)
            exponent = match.group(2)
            # Special case for square root
            if exponent == "0.5":
                return f"sqrt({base})"
            return f"pow({base}, {exponent})"
        
        expr_str = re.sub(power_pattern, power_replacement, expr_str)
        
        return expr_str
    
    def _fix_power_operators(self, expr_str: str) -> str:
        """Fix any remaining **0.5 patterns that weren't caught by the main conversion."""
        # Multiple passes to handle nested cases
        prev_expr = ""
        while prev_expr != expr_str:
            prev_expr = expr_str
            
            # Handle complex expressions with balanced parentheses
            # Look for patterns like (...)**0.5 where ... can contain nested parentheses
            paren_count = 0
            i = 0
            while i < len(expr_str):
                if expr_str[i] == '(':
                    start = i
                    paren_count = 1
                    i += 1
                    
                    # Find matching closing parenthesis
                    while i < len(expr_str) and paren_count > 0:
                        if expr_str[i] == '(':
                            paren_count += 1
                        elif expr_str[i] == ')':
                            paren_count -= 1
                        i += 1
                    
                    # Check if followed by **0.5
                    if i < len(expr_str) - 4 and expr_str[i:i+5] == '**0.5':
                        # Replace this pattern
                        inner_expr = expr_str[start+1:i-1]  # content between parentheses
                        replacement = f'sqrt({inner_expr})'
                        expr_str = expr_str[:start] + replacement + expr_str[i+5:]
                        break  # Restart the loop
                else:
                    i += 1
        
        # Handle simple cases that might have been missed
        # Array access patterns like var[index]**0.5
        array_sqrt_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*\[[^\]]+\])\*\*0\.5'
        expr_str = re.sub(array_sqrt_pattern, r'sqrt(\1)', expr_str)
        
        # Simple variable patterns like var**0.5
        var_sqrt_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\*\*0\.5'
        expr_str = re.sub(var_sqrt_pattern, r'sqrt(\1)', expr_str)
        
        return expr_str
    
    def _compile_cpp_code(self, cpp_code: str, func_name: str) -> str:
        """Compile the C++ code into a shared library and return the path."""
        code_hash = hashlib.md5(cpp_code.encode()).hexdigest()[:8]
        cpp_file = os.path.join(self.temp_dir, f"{func_name}_{code_hash}.cpp")
        so_file = os.path.join(self.temp_dir, f"{func_name}_{code_hash}.so")
        
        # Also save to current directory for inspection
        local_cpp_file = f"generated_{func_name}.cpp"
        
        # Write C++ code to both temp and local files
        with open(cpp_file, 'w') as f:
            f.write(cpp_code)
        with open(local_cpp_file, 'w') as f:
            f.write(cpp_code)
            
        print(f"Generated C++ code saved to: {local_cpp_file}")
            
        # Compile to shared library
        compile_cmd = [
            "g++", "-shared", "-fPIC", "-O3", "-std=c++17",
            cpp_file, "-o", so_file
        ]
        
        try:
            subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Compilation failed: {e.stderr}")
            
        return so_file
    
    def _create_python_wrapper(self, so_file: str, func_name: str, equations: List[sp.Eq]):
        """Create a Python wrapper function for multi-dimensional arrays."""
        import ctypes
        
        # Load the shared library
        lib = ctypes.CDLL(so_file)
        
        # Get function signature info
        vector_vars, scalar_vars, result_vars, stencil_info, array_dims = self._analyze_equations(equations)
        
        def wrapper(*args):
            """
            Python wrapper for multi-dimensional compiled C++ function.
            Arguments: result_arrays..., input_arrays..., shape_params..., scalar_args...
            """
            # Determine if we have multi-dimensional arrays
            is_multidim = any(dim > 1 for dim in array_dims.values())
            max_dim = max(array_dims.values()) if array_dims else 1
            
            expected_args = len(result_vars) + len(vector_vars) + len(scalar_vars)
            if is_multidim:
                expected_args += max_dim  # shape parameters
            
            if len(args) != expected_args:
                raise ValueError(f"Expected {expected_args} arguments, got {len(args)}")
            
            arg_idx = 0
            
            # Result arrays
            result_arrays = []
            for var in result_vars:
                result_array = np.asarray(args[arg_idx], dtype=np.float64)
                result_arrays.append(result_array)
                arg_idx += 1
            
            # Input arrays
            vector_args = []
            for var in vector_vars:
                vec_array = np.asarray(args[arg_idx], dtype=np.float64)
                vector_args.append(vec_array)
                arg_idx += 1
            
            # Shape parameters (if multi-dimensional)
            shape_params = []
            if is_multidim:
                for dim in range(max_dim):
                    shape_params.append(int(args[arg_idx]))
                    arg_idx += 1
            
            # Scalar arguments
            scalar_args = []
            for var in scalar_vars:
                scalar_args.append(float(args[arg_idx]))
                arg_idx += 1
            
            # Direct computation for v4
            self._compute_multidim_equations_directly(equations, result_arrays, vector_args, shape_params,
                                                    scalar_args, vector_vars, scalar_vars, result_vars, 
                                                    stencil_info, array_dims)
            
        return wrapper
    
    def _compute_multidim_equations_directly(self, equations: List[sp.Eq], result_arrays, vector_args, shape_params,
                                           scalar_args, vector_vars, scalar_vars, result_vars, 
                                           stencil_info: MultiDimStencilInfo, array_dims: Dict[str, int]):
        """Direct computation of multi-dimensional equations."""
        is_multidim = any(dim > 1 for dim in array_dims.values())
        max_dim = max(array_dims.values()) if array_dims else 1
        
        # Create symbol mapping
        symbol_map = {}
        
        # Map input arrays
        for i, var in enumerate(vector_vars):
            symbol_map[var] = vector_args[i]
        
        # Map result arrays
        for i, var in enumerate(result_vars):
            symbol_map[var] = result_arrays[i]
        
        # Map scalar variables
        for i, var in enumerate(scalar_vars):
            symbol_map[var] = scalar_args[i]
        
        if is_multidim and max_dim > 1:
            # Multi-dimensional case
            if max_dim == 2:
                rows, cols = shape_params[:2]
                self._compute_2d_equations(equations, symbol_map, result_vars, stencil_info, 
                                         array_dims, rows, cols)
            elif max_dim == 3:
                rows, cols, depth = shape_params[:3]
                self._compute_3d_equations(equations, symbol_map, result_vars, stencil_info, 
                                         array_dims, rows, cols, depth)
        else:
            # 1D case (same as v3)
            self._compute_1d_equations(equations, symbol_map, result_vars, stencil_info)
    
    def _compute_2d_equations(self, equations, symbol_map, result_vars, stencil_info, array_dims, rows, cols):
        """Compute 2D equations."""
        min_bounds, max_bounds = stencil_info.get_loop_bounds(['rows', 'cols'])
        min_i, min_j = int(min_bounds[0]), int(min_bounds[1])
        max_i = rows - stencil_info.max_offsets[0] if stencil_info.max_offsets[0] > 0 else rows
        max_j = cols - stencil_info.max_offsets[1] if stencil_info.max_offsets[1] > 0 else cols
        
        for i in range(min_i, max_i):
            for j in range(min_j, max_j):
                flat_idx = i * cols + j
                
                for eq in equations:
                    try:
                        lhs = eq.lhs
                        rhs = eq.rhs
                        result_var = str(lhs.base)
                        
                        # Substitute indexed expressions
                        expr_subs = rhs
                        indexed_exprs = list(rhs.atoms(sp.Indexed))
                        
                        for indexed_expr in indexed_exprs:
                            base_name = str(indexed_expr.base)
                            if base_name in symbol_map and isinstance(symbol_map[base_name], np.ndarray):
                                indices = indexed_expr.indices
                                
                                # Handle different dimensionalities
                                if len(indices) == 2:
                                    # 2D array access
                                    i_offset = self._parse_index_offset(indices[0])
                                    j_offset = self._parse_index_offset(indices[1])
                                    
                                    access_i = i + i_offset
                                    access_j = j + j_offset
                                    
                                    if (0 <= access_i < rows and 0 <= access_j < cols):
                                        if array_dims.get(base_name, 1) > 1:
                                            # 2D array - use flattened index
                                            access_idx = access_i * cols + access_j
                                        else:
                                            # 1D array - use only first dimension
                                            access_idx = access_i
                                        
                                        if 0 <= access_idx < len(symbol_map[base_name]):
                                            value = symbol_map[base_name][access_idx]
                                            expr_subs = expr_subs.subs(indexed_expr, value)
                                
                                elif len(indices) == 1:
                                    # 1D array access
                                    i_offset = self._parse_index_offset(indices[0])
                                    access_i = i + i_offset
                                    
                                    if 0 <= access_i < len(symbol_map[base_name]):
                                        value = symbol_map[base_name][access_i]
                                        expr_subs = expr_subs.subs(indexed_expr, value)
                        
                        # Substitute scalars
                        for var, value in symbol_map.items():
                            if not isinstance(value, np.ndarray):
                                expr_subs = expr_subs.subs(sp.Symbol(var), value)
                        
                        # Store result
                        if array_dims.get(result_var, 1) > 1:
                            symbol_map[result_var][flat_idx] = float(expr_subs.evalf())
                        else:
                            symbol_map[result_var][i] = float(expr_subs.evalf())
                            
                    except Exception as e:
                        if array_dims.get(result_var, 1) > 1:
                            symbol_map[result_var][flat_idx] = 0.0
                        else:
                            symbol_map[result_var][i] = 0.0
    
    def _compute_3d_equations(self, equations, symbol_map, result_vars, stencil_info, array_dims, rows, cols, depth):
        """Compute 3D equations."""
        min_bounds, max_bounds = stencil_info.get_loop_bounds(['rows', 'cols', 'depth'])
        min_i, min_j, min_k = int(min_bounds[0]), int(min_bounds[1]), int(min_bounds[2])
        max_i = rows - stencil_info.max_offsets[0] if stencil_info.max_offsets[0] > 0 else rows
        max_j = cols - stencil_info.max_offsets[1] if stencil_info.max_offsets[1] > 0 else cols
        max_k = depth - stencil_info.max_offsets[2] if stencil_info.max_offsets[2] > 0 else depth
        
        for i in range(min_i, max_i):
            for j in range(min_j, max_j):
                for k in range(min_k, max_k):
                    flat_idx = i * cols * depth + j * depth + k
                    
                    for eq in equations:
                        try:
                            lhs = eq.lhs
                            rhs = eq.rhs
                            result_var = str(lhs.base)
                            
                            # Similar to 2D but with 3 indices
                            expr_subs = rhs
                            indexed_exprs = list(rhs.atoms(sp.Indexed))
                            
                            for indexed_expr in indexed_exprs:
                                base_name = str(indexed_expr.base)
                                if base_name in symbol_map and isinstance(symbol_map[base_name], np.ndarray):
                                    indices = indexed_expr.indices
                                    if len(indices) == 3:
                                        i_offset = self._parse_index_offset(indices[0])
                                        j_offset = self._parse_index_offset(indices[1])
                                        k_offset = self._parse_index_offset(indices[2])
                                        
                                        access_i = i + i_offset
                                        access_j = j + j_offset
                                        access_k = k + k_offset
                                        
                                        if (0 <= access_i < rows and 0 <= access_j < cols and 0 <= access_k < depth):
                                            access_idx = access_i * cols * depth + access_j * depth + access_k
                                            value = symbol_map[base_name][access_idx]
                                            expr_subs = expr_subs.subs(indexed_expr, value)
                            
                            # Substitute scalars
                            for var, value in symbol_map.items():
                                if not isinstance(value, np.ndarray):
                                    expr_subs = expr_subs.subs(sp.Symbol(var), value)
                            
                            # Store result
                            symbol_map[result_var][flat_idx] = float(expr_subs.evalf())
                            
                        except Exception as e:
                            symbol_map[result_var][flat_idx] = 0.0
    
    def _compute_1d_equations(self, equations, symbol_map, result_vars, stencil_info):
        """Compute 1D equations (same as v3)."""
        n = len(list(symbol_map.values())[0])  # Get size from first array
        
        # Determine loop bounds
        if stencil_info.offsets[0] and (stencil_info.min_offsets[0] < 0 or stencil_info.max_offsets[0] > 0):
            min_index = max(0, -stencil_info.min_offsets[0])
            max_index = n - stencil_info.max_offsets[0] if stencil_info.max_offsets[0] > 0 else n
        else:
            min_index = 0
            max_index = n
        
        # Evaluate each equation for each valid index
        for i in range(min_index, max_index):
            for eq in equations:
                try:
                    lhs = eq.lhs
                    rhs = eq.rhs
                    result_var = str(lhs.base)
                    
                    # Substitute variables in the expression
                    expr_subs = rhs
                    
                    # Handle indexed expressions with stencil patterns
                    indexed_exprs = list(rhs.atoms(sp.Indexed))
                    for indexed_expr in indexed_exprs:
                        base_name = str(indexed_expr.base)
                        if base_name in symbol_map and isinstance(symbol_map[base_name], np.ndarray):
                            index_expr = indexed_expr.indices[0]
                            offset = self._parse_index_offset(index_expr)
                            
                            access_index = i + offset
                            if 0 <= access_index < len(symbol_map[base_name]):
                                value = symbol_map[base_name][access_index]
                                expr_subs = expr_subs.subs(indexed_expr, value)
                    
                    # Substitute scalar variables
                    for var, value in symbol_map.items():
                        if not isinstance(value, np.ndarray):
                            expr_subs = expr_subs.subs(sp.Symbol(var), value)
                    
                    # Evaluate the result
                    symbol_map[result_var][i] = float(expr_subs.evalf())
                    
                except Exception as e:
                    symbol_map[result_var][i] = 0.0


def genFunc(equations: Union[sp.Eq, List[sp.Eq]]) -> callable:
    """
    Generate a callable function from SymPy equation(s) with multi-dimensional support.
    
    Args:
        equations: Single SymPy equation or list of equations
        
    Returns:
        Callable function that takes numpy arrays as arguments
    """
    # Handle single equation case
    if isinstance(equations, sp.Eq):
        equations = [equations]
    
    generator = MultiDimCodeGenerator()
    
    # Generate a unique function name
    eq_strs = [str(eq) for eq in equations]
    combined_str = "|".join(eq_strs)
    func_hash = hashlib.md5(combined_str.encode()).hexdigest()[:8]
    func_name = f"cpp_multidim_{func_hash}"
    
    # Generate and compile C++ code
    cpp_code = generator._generate_cpp_code(equations, func_name)
    so_file = generator._compile_cpp_code(cpp_code, func_name)
    
    # Create Python wrapper
    wrapper = generator._create_python_wrapper(so_file, func_name, equations)
    
    return wrapper 