"""
SymPy-based code generation for HPX Python API v4 - Multi-Dimensional Support
Extends v3 with support for multi-dimensional arrays and advanced optimizations.
"""

import os
import subprocess
import hashlib
import atexit
import re
from typing import Dict, List, Tuple, Any, Union
import sympy as sp
import numpy as np


# Global set to track generated files for cleanup
_generated_files = set()


def _cleanup_generated_files():
    """Cleanup function to remove generated files on exit."""
    for file_path in _generated_files.copy():
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                _generated_files.discard(file_path)
        except OSError:
            pass  # File might already be deleted


# Register cleanup function to run on exit
atexit.register(_cleanup_generated_files)


class MultiDimStencilInfo:
    """Enhanced stencil info for multi-dimensional arrays."""
    def __init__(self, max_dimensions: int = 3):
        self.max_dimensions = max_dimensions
        self.offsets = {dim: set() for dim in range(max_dimensions)}
        self.min_offset = {dim: 0 for dim in range(max_dimensions)}
        self.max_offset = {dim: 0 for dim in range(max_dimensions)}
    
    def add_offset(self, dim: int, offset: int):
        """Add an offset for a specific dimension."""
        if dim < self.max_dimensions:
            self.offsets[dim].add(offset)
            self.min_offset[dim] = min(self.min_offset[dim], offset)
            self.max_offset[dim] = max(self.max_offset[dim], offset)
    
    def get_loop_bounds(self, dim: int, size_var: str) -> Tuple[str, str]:
        """Get loop bounds for a specific dimension."""
        min_bound = f"({-self.min_offset[dim]})" if self.min_offset[dim] < 0 else "0"
        max_bound = f"({size_var} - {self.max_offset[dim]})" if self.max_offset[dim] > 0 else size_var
        return min_bound, max_bound


class MultiDimCodeGenerator:
    """
    Enhanced code generator supporting multi-dimensional arrays and equations.
    """
    
    def __init__(self):
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
        
        # Second pass: analyze each equation
        for eq in equations:
            lhs = eq.lhs
            rhs = eq.rhs
            
            # Extract result variable (left-hand side)
            if isinstance(lhs, sp.Indexed):
                result_var = str(lhs.base)
                result_vars.append(result_var)
            else:
                raise ValueError("Left-hand side must be an indexed expression")
            
            # Find all IndexedBase objects in this equation
            indexed_bases = list(eq.atoms(sp.IndexedBase))
            equation_vectors = [str(base) for base in indexed_bases]
            all_vector_vars.update(equation_vectors)
            
            # Analyze multi-dimensional stencil patterns
            indexed_exprs = list(eq.atoms(sp.Indexed))
            for indexed in indexed_exprs:
                for dim_idx, idx_expr in enumerate(indexed.indices):
                    if dim_idx < max_dimensions:
                        offset = self._parse_index_offset(idx_expr)
                        stencil_info.add_offset(dim_idx, offset)
            
            # Find scalar variables in this equation
            all_symbols = list(eq.atoms(sp.Symbol))
            for symbol in all_symbols:
                symbol_name = str(symbol)
                if symbol_name not in equation_vectors and symbol_name not in index_vars:
                    all_scalar_vars.add(symbol_name)
        
        # Convert to sorted lists
        vector_vars = sorted(list(all_vector_vars))
        scalar_vars = sorted(list(all_scalar_vars))
        
        # Remove result variables from input vectors
        input_vector_vars = [var for var in vector_vars if var not in result_vars]
        
        return input_vector_vars, scalar_vars, result_vars, stencil_info, array_dims
    
    def _parse_index_offset(self, idx_expr) -> int:
        """Parse index expression to extract offset."""
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
            return 0
    
    def _generate_cpp_code(self, equations: List[sp.Eq], func_name: str) -> str:
        """Generate C++ code for multi-dimensional equations."""
        vector_vars, scalar_vars, result_vars, stencil_info, array_dims = self._analyze_equations(equations)
        
        # Determine if we have multi-dimensional arrays
        is_multidim = any(dim > 1 for dim in array_dims.values())
        max_dim = max(array_dims.values()) if array_dims else 1
        
        # Generate function signature
        cpp_code = f"""#include <cmath>

extern "C" {{

void {func_name}("""
        
        # Result parameters first
        for i, var in enumerate(result_vars):
            if i > 0:
                cpp_code += ",\n               "
            cpp_code += f"double* result_{var}"
        
        # Input vector parameters
        for var in vector_vars:
            cpp_code += f",\n               const double* {var}"
        
        # Dimension parameters (if multi-dimensional)
        if is_multidim:
            if max_dim >= 2:
                cpp_code += ",\n               const int rows"
                cpp_code += ",\n               const int cols"
            if max_dim >= 3:
                cpp_code += ",\n               const int depth"
        else:
            cpp_code += ",\n               const int n"
        
        # Scalar parameters
        for var in scalar_vars:
            cpp_code += f",\n               const double {var}"
        
        cpp_code += ")\n{\n"
        
        # Generate loop structure based on dimensionality
        if is_multidim and max_dim > 1:
            cpp_code += self._generate_multidim_loops(equations, vector_vars, scalar_vars, result_vars, 
                                                    stencil_info, array_dims, max_dim)
        else:
            cpp_code += self._generate_1d_loops(equations, vector_vars, scalar_vars, result_vars, stencil_info)
        
        cpp_code += "}\n\n}"
        
        return cpp_code
    
    def _generate_1d_loops(self, equations: List[sp.Eq], vector_vars: List[str], scalar_vars: List[str],
                          result_vars: List[str], stencil_info: MultiDimStencilInfo) -> str:
        """Generate 1D loop structure."""
        cpp_code = ""
        
        # Generate stencil bounds for 1D
        if stencil_info.offsets[0] and (stencil_info.min_offset[0] < 0 or stencil_info.max_offset[0] > 0):
            min_bound, max_bound = stencil_info.get_loop_bounds(0, "n")
            cpp_code += f"    const int min_index = {min_bound};\n"
            cpp_code += f"    const int max_index = {max_bound};\n"
            loop_start = "min_index"
            loop_end = "max_index"
        else:
            loop_start = "0"
            loop_end = "n"
        
        cpp_code += f"\n    // Generated 1D loop\n"
        cpp_code += f"    for(int i = {loop_start}; i < {loop_end}; i++) {{\n"
        
        # Process each equation
        for eq in equations:
            lhs = eq.lhs
            rhs = eq.rhs
            result_var = str(lhs.base)
            
            # Convert expression to C++ code
            rhs_str = self._convert_1d_expression_to_cpp(rhs, vector_vars, scalar_vars, result_vars)
            
            cpp_code += f"        result_{result_var}[i] = {rhs_str};\n"
        
        cpp_code += "    }\n"
        
        return cpp_code
    
    def _generate_multidim_loops(self, equations: List[sp.Eq], vector_vars: List[str], scalar_vars: List[str],
                               result_vars: List[str], stencil_info: MultiDimStencilInfo, 
                               array_dims: Dict[str, int], max_dim: int) -> str:
        """Generate multi-dimensional loop structure."""
        cpp_code = ""
        
        if max_dim == 2:
            # 2D nested loops
            cpp_code += "    // Generated 2D loops\n"
            cpp_code += "    for(int i = 0; i < rows; i++) {\n"
            cpp_code += "        for(int j = 0; j < cols; j++) {\n"
            
            # Process each equation
            for eq in equations:
                lhs = eq.lhs
                rhs = eq.rhs
                result_var = str(lhs.base)
                
                # Convert expression to C++ code
                rhs_str = self._convert_2d_expression_to_cpp(rhs, vector_vars, scalar_vars, result_vars, array_dims)
                
                cpp_code += f"            result_{result_var}[i * cols + j] = {rhs_str};\n"
            
            cpp_code += "        }\n"
            cpp_code += "    }\n"
            
        elif max_dim == 3:
            # 3D nested loops
            cpp_code += "    // Generated 3D loops\n"
            cpp_code += "    for(int i = 0; i < rows; i++) {\n"
            cpp_code += "        for(int j = 0; j < cols; j++) {\n"
            cpp_code += "            for(int k = 0; k < depth; k++) {\n"
            
            # Process each equation
            for eq in equations:
                lhs = eq.lhs
                rhs = eq.rhs
                result_var = str(lhs.base)
                
                # Convert expression to C++ code
                rhs_str = self._convert_3d_expression_to_cpp(rhs, vector_vars, scalar_vars, result_vars, array_dims)
                
                cpp_code += f"                result_{result_var}[i * cols * depth + j * depth + k] = {rhs_str};\n"
            
            cpp_code += "            }\n"
            cpp_code += "        }\n"
            cpp_code += "    }\n"
        
        return cpp_code
    
    def _convert_1d_expression_to_cpp(self, expr, vector_vars: List[str], scalar_vars: List[str], result_vars: List[str]) -> str:
        """Convert 1D expression to C++ code."""
        expr_str = str(expr)
        
        # Handle indexed expressions
        indexed_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\[([^\]]+)\]'
        
        def replace_indexed(match):
            var_name = match.group(1)
            index_expr = match.group(2)
            
            if var_name in vector_vars:
                return f"{var_name}[{index_expr}]"
            elif var_name in result_vars:
                return f"result_{var_name}[{index_expr}]"
            else:
                return match.group(0)
        
        expr_str = re.sub(indexed_pattern, replace_indexed, expr_str)
        
        # Replace scalar variables
        for var in scalar_vars:
            expr_str = re.sub(r'\b' + re.escape(var) + r'\b', var, expr_str)
        
        # Handle power operators
        expr_str = self._fix_power_operators(expr_str)
        
        return expr_str
    
    def _convert_2d_expression_to_cpp(self, expr, vector_vars: List[str], scalar_vars: List[str], 
                                    result_vars: List[str], array_dims: Dict[str, int]) -> str:
        """Convert 2D expression to C++ code."""
        expr_str = str(expr)
        
        # Handle indexed expressions
        indexed_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\[([^\]]+)\]'
        
        def replace_indexed(match):
            var_name = match.group(1)
            index_expr = match.group(2)
            
            if var_name in vector_vars or var_name in result_vars:
                # Parse multi-dimensional indices
                indices = [idx.strip() for idx in index_expr.split(',')]
                if len(indices) == 2:
                    # 2D access
                    flat_access = f"({indices[0]}) * cols + ({indices[1]})"
                    if var_name in vector_vars:
                        return f"{var_name}[{flat_access}]"
                    elif var_name in result_vars:
                        return f"result_{var_name}[{flat_access}]"
                else:
                    # 1D access
                    if var_name in vector_vars:
                        return f"{var_name}[{index_expr}]"
                    elif var_name in result_vars:
                        return f"result_{var_name}[{index_expr}]"
            
            return match.group(0)
        
        expr_str = re.sub(indexed_pattern, replace_indexed, expr_str)
        
        # Replace scalar variables
        for var in scalar_vars:
            expr_str = re.sub(r'\b' + re.escape(var) + r'\b', var, expr_str)
        
        # Handle power operators
        expr_str = self._fix_power_operators(expr_str)
        
        return expr_str
    
    def _convert_3d_expression_to_cpp(self, expr, vector_vars: List[str], scalar_vars: List[str], 
                                    result_vars: List[str], array_dims: Dict[str, int]) -> str:
        """Convert 3D expression to C++ code."""
        expr_str = str(expr)
        
        # Handle indexed expressions
        indexed_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\[([^\]]+)\]'
        
        def replace_indexed(match):
            var_name = match.group(1)
            index_expr = match.group(2)
            
            if var_name in vector_vars or var_name in result_vars:
                # Parse multi-dimensional indices
                indices = [idx.strip() for idx in index_expr.split(',')]
                if len(indices) == 3:
                    # 3D access
                    flat_access = f"({indices[0]}) * cols * depth + ({indices[1]}) * depth + ({indices[2]})"
                    if var_name in vector_vars:
                        return f"{var_name}[{flat_access}]"
                    elif var_name in result_vars:
                        return f"result_{var_name}[{flat_access}]"
                elif len(indices) == 2:
                    # 2D access
                    flat_access = f"({indices[0]}) * cols + ({indices[1]})"
                    if var_name in vector_vars:
                        return f"{var_name}[{flat_access}]"
                    elif var_name in result_vars:
                        return f"result_{var_name}[{flat_access}]"
                else:
                    # 1D access
                    if var_name in vector_vars:
                        return f"{var_name}[{index_expr}]"
                    elif var_name in result_vars:
                        return f"result_{var_name}[{index_expr}]"
            
            return match.group(0)
        
        expr_str = re.sub(indexed_pattern, replace_indexed, expr_str)
        
        # Replace scalar variables
        for var in scalar_vars:
            expr_str = re.sub(r'\b' + re.escape(var) + r'\b', var, expr_str)
        
        # Handle power operators
        expr_str = self._fix_power_operators(expr_str)
        
        return expr_str
    
    def _fix_power_operators(self, expr_str: str) -> str:
        """
        Replace Python-style '**' power operators with C++ functions (pow, sqrt).
        This function iteratively finds and replaces power operations, starting from
        the rightmost (which corresponds to the innermost in evaluation order)
        to handle complex nested expressions correctly.
        """
        while "**" in expr_str:
            idx = expr_str.rfind("**")

            # ---- Find Exponent ----
            exponent_part = expr_str[idx+2:].lstrip()
            exponent_str = ""
            # Exponent can be a number like 2, 0.5, or a parenthesized expression
            if exponent_part.startswith('('):
                level = 1
                for i in range(1, len(exponent_part)):
                    if exponent_part[i] == '(': level += 1
                    elif exponent_part[i] == ')': level -= 1
                    if level == 0:
                        exponent_str = exponent_part[:i+1]
                        break
            else:
                match = re.match(r"[0-9.]+|[a-zA-Z_]\w*", exponent_part)
                if match:
                    exponent_str = match.group(0)

            if not exponent_str: break 

            # ---- Find Base ----
            base_part = expr_str[:idx].rstrip()
            base_str = ""
            if base_part.endswith(')'):
                level = 1
                start_idx = -1
                for i in range(len(base_part)-2, -1, -1):
                    if base_part[i] == ')': level += 1
                    elif base_part[i] == '(': level -= 1
                    if level == 0:
                        start_idx = i
                        break
                if start_idx != -1:
                    base_str = base_part[start_idx:]
            else:
                match = re.search(r"([a-zA-Z_]\w*(?:\[[^\]]+\])?)$", base_part)
                if match:
                    base_str = match.group(1)

            if not base_str: break

            # ---- Construct Replacement ----
            replacement = ""
            if exponent_str in ["0.5", "(1/2)"]:
                 replacement = f"sqrt({base_str})"
            else:
                 replacement = f"pow({base_str}, {exponent_str})"

            # ---- Replace in Original String ----
            start_replace_idx = expr_str.rfind(base_str, 0, idx)
            end_replace_idx = idx + 2 + len(expr_str[idx+2:]) - len(exponent_part) + len(exponent_str)
            expr_str = expr_str[:start_replace_idx] + replacement + expr_str[end_replace_idx:]
        
        return expr_str
    
    def _compile_cpp_code(self, cpp_code: str, func_name: str) -> str:
        """Compile the C++ code into a shared library and return the path."""
        code_hash = hashlib.md5(cpp_code.encode()).hexdigest()[:8]
        cpp_file = f"generated_{func_name}_{code_hash}.cpp"
        so_file = f"generated_{func_name}_{code_hash}.so"
        
        # Write C++ code to local file
        with open(cpp_file, 'w') as f:
            f.write(cpp_code)
            
        print(f"Generated C++ code saved to: {cpp_file}")
        
        # Track files for cleanup
        _generated_files.add(cpp_file)
        _generated_files.add(so_file)
            
        # Compile to shared library
        compile_cmd = [
            "g++", "-shared", "-fPIC", "-O3", "-std=c++17",
            cpp_file, "-o", so_file
        ]
        
        try:
            result = subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
            print(f"Compilation successful for: {so_file}")
        except subprocess.CalledProcessError as e:
            print(f"Compilation command: {' '.join(compile_cmd)}")
            print(f"Compilation stderr: {e.stderr}")
            print(f"Compilation stdout: {e.stdout}")
            raise RuntimeError(f"Compilation failed: {e.stderr}")
            
        return so_file
    
    def _create_python_wrapper(self, so_file: str, func_name: str, equations: List[sp.Eq]):
        """Create a Python wrapper function for multi-dimensional arrays."""
        import ctypes
        from ctypes import POINTER, c_double, c_int
        
        # Load the shared library (use absolute path)
        abs_so_file = os.path.abspath(so_file)
        lib = ctypes.CDLL(abs_so_file)
        
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
            else:
                expected_args += 1  # size parameter for 1D
            
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
            
            # Call the compiled C++ function using ctypes
            try:
                # Get the C function
                c_func = getattr(lib, func_name)
                
                # Convert numpy arrays to ctypes pointers
                result_ptrs = [arr.ctypes.data_as(POINTER(c_double)) for arr in result_arrays]
                vector_ptrs = [vec.ctypes.data_as(POINTER(c_double)) for vec in vector_args]
                
                # Set up argument types for the C function
                argtypes = []
                # Result vector pointers
                for _ in result_arrays:
                    argtypes.append(POINTER(c_double))
                # Input vector pointers
                for _ in vector_args:
                    argtypes.append(POINTER(c_double))
                # Shape parameters (if multi-dimensional)
                if is_multidim:
                    for _ in range(max_dim):
                        argtypes.append(c_int)
                # Scalar arguments
                for _ in scalar_args:
                    argtypes.append(c_double)
                if not is_multidim:
                    argtypes.append(c_int)  # size parameter for 1D
                
                c_func.argtypes = argtypes
                c_func.restype = None
                
                # Call the C++ function
                if is_multidim:
                    call_args = result_ptrs + vector_ptrs + shape_params + scalar_args
                else:
                    # 1D case - pass array size *last*
                    n = len(result_arrays[0]) if result_arrays else len(vector_args[0])
                    call_args = result_ptrs + vector_ptrs + scalar_args + [n]
                
                c_func(*call_args)
                
            except Exception as e:
                print(f"Failed to call C++ function: {e}")
                print("Falling back to Python computation...")
                # Fallback to Python computation
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
    
    def _compute_1d_equations(self, equations: List[sp.Eq], symbol_map: Dict[str, Any], 
                            result_vars: List[str], stencil_info: MultiDimStencilInfo):
        """Compute 1D equations directly."""
        n = len(symbol_map[result_vars[0]])
        
        # Determine loop bounds
        if stencil_info.offsets[0] and (stencil_info.min_offset[0] < 0 or stencil_info.max_offset[0] > 0):
            min_index = max(0, -stencil_info.min_offset[0])
            max_index = n - stencil_info.max_offset[0] if stencil_info.max_offset[0] > 0 else n
        else:
            min_index = 0
            max_index = n
        
        # Evaluate equations
        for i in range(min_index, max_index):
            for eq in equations:
                try:
                    lhs = eq.lhs
                    rhs = eq.rhs
                    result_var = str(lhs.base)
                    
                    # Substitute variables
                    expr_subs = rhs
                    indexed_exprs = list(rhs.atoms(sp.Indexed))
                    for indexed_expr in indexed_exprs:
                        base_name = str(indexed_expr.base)
                        if base_name in symbol_map and isinstance(symbol_map[base_name], np.ndarray):
                            # Simple 1D index substitution
                            value = symbol_map[base_name][i]
                            expr_subs = expr_subs.subs(indexed_expr, value)
                    
                    # Substitute scalars
                    for var, value in symbol_map.items():
                        if not isinstance(value, np.ndarray):
                            expr_subs = expr_subs.subs(sp.Symbol(var), value)
                    
                    # Evaluate result
                    result_idx = result_vars.index(result_var)
                    symbol_map[result_var][i] = float(expr_subs.evalf())
                    
                except Exception:
                    result_idx = result_vars.index(result_var)
                    symbol_map[result_var][i] = 0.0
    
    def _compute_2d_equations(self, equations: List[sp.Eq], symbol_map: Dict[str, Any], 
                            result_vars: List[str], stencil_info: MultiDimStencilInfo, 
                            array_dims: Dict[str, int], rows: int, cols: int):
        """Compute 2D equations directly."""
        for i in range(rows):
            for j in range(cols):
                for eq in equations:
                    try:
                        lhs = eq.lhs
                        rhs = eq.rhs
                        result_var = str(lhs.base)
                        
                        # Substitute variables
                        expr_subs = rhs
                        indexed_exprs = list(rhs.atoms(sp.Indexed))
                        for indexed_expr in indexed_exprs:
                            base_name = str(indexed_expr.base)
                            if base_name in symbol_map and isinstance(symbol_map[base_name], np.ndarray):
                                # 2D index substitution
                                flat_index = i * cols + j
                                value = symbol_map[base_name][flat_index]
                                expr_subs = expr_subs.subs(indexed_expr, value)
                        
                        # Substitute scalars
                        for var, value in symbol_map.items():
                            if not isinstance(value, np.ndarray):
                                expr_subs = expr_subs.subs(sp.Symbol(var), value)
                        
                        # Evaluate result
                        flat_index = i * cols + j
                        symbol_map[result_var][flat_index] = float(expr_subs.evalf())
                        
                    except Exception:
                        flat_index = i * cols + j
                        symbol_map[result_var][flat_index] = 0.0
    
    def _compute_3d_equations(self, equations: List[sp.Eq], symbol_map: Dict[str, Any], 
                            result_vars: List[str], stencil_info: MultiDimStencilInfo, 
                            array_dims: Dict[str, int], rows: int, cols: int, depth: int):
        """Compute 3D equations directly."""
        for i in range(rows):
            for j in range(cols):
                for k in range(depth):
                    for eq in equations:
                        try:
                            lhs = eq.lhs
                            rhs = eq.rhs
                            result_var = str(lhs.base)
                            
                            # Substitute variables
                            expr_subs = rhs
                            indexed_exprs = list(rhs.atoms(sp.Indexed))
                            for indexed_expr in indexed_exprs:
                                base_name = str(indexed_expr.base)
                                if base_name in symbol_map and isinstance(symbol_map[base_name], np.ndarray):
                                    # 3D index substitution
                                    flat_index = i * cols * depth + j * depth + k
                                    value = symbol_map[base_name][flat_index]
                                    expr_subs = expr_subs.subs(indexed_expr, value)
                            
                            # Substitute scalars
                            for var, value in symbol_map.items():
                                if not isinstance(value, np.ndarray):
                                    expr_subs = expr_subs.subs(sp.Symbol(var), value)
                            
                            # Evaluate result
                            flat_index = i * cols * depth + j * depth + k
                            symbol_map[result_var][flat_index] = float(expr_subs.evalf())
                            
                        except Exception:
                            flat_index = i * cols * depth + j * depth + k
                            symbol_map[result_var][flat_index] = 0.0


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