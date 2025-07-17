"""
SymPy-based code generation for HPX Python API v3 - Multi-Equation Support
Extends v2 with support for processing multiple equations simultaneously.
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


class StencilInfo:
    """Information about stencil patterns in multi-equation expressions."""
    def __init__(self):
        self.min_offset = 0
        self.max_offset = 0
        self.offsets = set()
    
    def add_offset(self, offset: int):
        """Add an index offset to the stencil pattern."""
        self.offsets.add(offset)
        self.min_offset = min(self.min_offset, offset)
        self.max_offset = max(self.max_offset, offset)
    
    def get_loop_bounds(self, n: str) -> Tuple[str, str]:
        """Get the loop bounds for safe stencil access."""
        min_index = max(0, -self.min_offset)
        max_index = f"{n} - {self.max_offset}" if self.max_offset > 0 else n
        return str(min_index), max_index


class MultiEquationCodeGenerator:
    """
    Enhanced code generator that handles multiple equations processed together.
    """
    
    def __init__(self):
        self.compiled_functions = {}
        
    def _analyze_equations(self, equations: List[sp.Eq]) -> Tuple[List[str], List[str], List[str], StencilInfo]:
        """
        Analyze multiple SymPy equations to extract variables, types, and stencil patterns.
        
        Returns:
            - vector_vars: List of vector variable names (input vectors)
            - scalar_vars: List of scalar variable names  
            - result_vars: List of result variable names (output vectors)
            - stencil_info: Information about stencil patterns
        """
        all_vector_vars = set()
        all_scalar_vars = set()
        result_vars = []
        stencil_info = StencilInfo()
        
        # Analyze each equation
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
            
            # Analyze stencil patterns
            indexed_exprs = list(eq.atoms(sp.Indexed))
            for indexed in indexed_exprs:
                for idx_expr in indexed.indices:
                    offset = self._parse_index_offset(idx_expr)
                    stencil_info.add_offset(offset)
            
            # Find scalar variables in this equation
            all_symbols = list(eq.atoms(sp.Symbol))
            
            # Get index variables for this equation
            index_vars = set()
            for indexed in indexed_exprs:
                for idx in indexed.indices:
                    if isinstance(idx, (sp.Symbol, sp.Idx)):
                        index_vars.add(str(idx))
                    elif hasattr(idx, 'free_symbols'):
                        for sym in idx.free_symbols:
                            if isinstance(sym, (sp.Symbol, sp.Idx)):
                                index_vars.add(str(sym))
            
            # Add scalar variables from this equation
            for symbol in all_symbols:
                symbol_name = str(symbol)
                if symbol_name not in equation_vectors and symbol_name not in index_vars:
                    all_scalar_vars.add(symbol_name)
        
        # Convert to sorted lists
        vector_vars = sorted(list(all_vector_vars))
        scalar_vars = sorted(list(all_scalar_vars))
        
        # Remove result variables from input vectors
        input_vector_vars = [var for var in vector_vars if var not in result_vars]
        
        return input_vector_vars, scalar_vars, result_vars, stencil_info
    
    def _parse_index_offset(self, idx_expr) -> int:
        """Parse index expression to extract offset (same as v2)."""
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
        """
        Generate C++ code for multiple equations.
        """
        vector_vars, scalar_vars, result_vars, stencil_info = self._analyze_equations(equations)
        
        # Generate function signature
        cpp_code = f"""#include <cmath>

extern "C" {{

void {func_name}("""
        
        # Result parameters first (one for each output vector)
        for i, var in enumerate(result_vars):
            if i > 0:
                cpp_code += ",\n               "
            cpp_code += f"double* result_{var}"
        
        # Input vector parameters
        for var in vector_vars:
            cpp_code += f",\n               const double* {var}"
        
        # Scalar parameters
        for var in scalar_vars:
            cpp_code += f",\n               const double {var}"
            
        # Size parameter
        cpp_code += ",\n               const int n"
        cpp_code += ")\n{\n"
        
        # Generate stencil bounds
        if stencil_info.offsets and (stencil_info.min_offset < 0 or stencil_info.max_offset > 0):
            min_bound, max_bound = stencil_info.get_loop_bounds("n")
            cpp_code += f"    const int min_index = {min_bound};\n"
            cpp_code += f"    const int max_index = {max_bound};\n"
            loop_start = "min_index"
            loop_end = "max_index"
        else:
            loop_start = "0"
            loop_end = "n"
        
        # Generate the loop body
        cpp_code += f"\n    // Generated multi-equation loop\n"
        cpp_code += f"    for(int i = {loop_start}; i < {loop_end}; i++) {{\n"
        
        # Process each equation
        for eq in equations:
            lhs = eq.lhs
            rhs = eq.rhs
            result_var = str(lhs.base)
            
            # Convert expression to C++ code
            rhs_str = self._convert_expression_to_cpp(rhs, vector_vars, scalar_vars, result_vars)
            
            cpp_code += f"        result_{result_var}[i] = {rhs_str};\n"
        
        cpp_code += "    }\n"
        cpp_code += "}\n\n}"
        
        return cpp_code
    
    def _convert_expression_to_cpp(self, expr, vector_vars: List[str], scalar_vars: List[str], result_vars: List[str]) -> str:
        """Convert SymPy expression to C++ code."""
        expr_str = str(expr)
        
        # Handle indexed expressions
        indexed_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\[([^\]]+)\]'
        
        def replace_indexed(match):
            var_name = match.group(1)
            index_expr = match.group(2)
            
            if var_name in vector_vars:
                # Input vector
                return f"{var_name}[{index_expr}]"
            elif var_name in result_vars:
                # Output vector (for dependencies)
                return f"result_{var_name}[{index_expr}]"
            else:
                # Unknown variable, keep as-is
                return match.group(0)
        
        expr_str = re.sub(indexed_pattern, replace_indexed, expr_str)
        
        # Replace scalar variables
        for var in scalar_vars:
            expr_str = re.sub(r'\b' + re.escape(var) + r'\b', var, expr_str)
        
        # Handle power operators
        power_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*\[[^\]]+\]|\([^)]+\)|\w+)\*\*(\d+(?:\.\d+)?|\([^)]+\)|\w+)'
        def power_replacement(match):
            base = match.group(1)
            exponent = match.group(2)
            return f"pow({base}, {exponent})"
        
        expr_str = re.sub(power_pattern, power_replacement, expr_str)
        
        return expr_str
    
    def _compile_cpp_code(self, cpp_code: str, func_name: str) -> str:
        """
        Compile the C++ code into a shared library and return the path.
        """
        # Create unique filename based on code hash
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
        """
        Create a Python wrapper function that calls the compiled C++ function.
        """
        import ctypes
        from ctypes import POINTER, c_double, c_int
        
        # Load the shared library (use absolute path)
        abs_so_file = os.path.abspath(so_file)
        lib = ctypes.CDLL(abs_so_file)
        
        # Get function signature info
        vector_vars, scalar_vars, result_vars, stencil_info = self._analyze_equations(equations)
        
        def wrapper(*args):
            """
            Python wrapper for the compiled C++ multi-equation function.
            Arguments should be provided in the order: result_vectors..., input_vectors..., scalar_args...
            """
            expected_args = len(result_vars) + len(vector_vars) + len(scalar_vars)
            if len(args) != expected_args:
                raise ValueError(f"Expected {expected_args} arguments, got {len(args)}")
            
            arg_idx = 0
            
            # Result vectors (first arguments)
            result_arrays = []
            for var in result_vars:
                result_array = np.asarray(args[arg_idx], dtype=np.float64)
                result_arrays.append(result_array)
                arg_idx += 1
            
            # Input vector arguments
            vector_args = []
            for var in vector_vars:
                vec_array = np.asarray(args[arg_idx], dtype=np.float64)
                vector_args.append(vec_array)
                arg_idx += 1
            
            # Scalar arguments
            scalar_args = []
            for var in scalar_vars:
                scalar_args.append(float(args[arg_idx]))
                arg_idx += 1
            
            # Get the size and verify bounds
            n = len(result_arrays[0]) if result_arrays else len(vector_args[0])
            
            # Verify all vectors have the same size
            all_arrays = result_arrays + vector_args
            for arr in all_arrays:
                if len(arr) != n:
                    raise ValueError("All vectors must have the same size")
            
            # Check stencil bounds
            if stencil_info.min_offset < 0 or stencil_info.max_offset > 0:
                min_index = max(0, -stencil_info.min_offset)
                max_index = n - stencil_info.max_offset if stencil_info.max_offset > 0 else n
                
                if min_index >= max_index:
                    raise ValueError(f"Invalid stencil bounds: min_index={min_index}, max_index={max_index}, array_size={n}")
            
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
                # Scalar arguments
                for _ in scalar_args:
                    argtypes.append(c_double)
                # Size parameter
                argtypes.append(c_int)
                
                c_func.argtypes = argtypes
                c_func.restype = None
                
                # Call the C++ function
                call_args = result_ptrs + vector_ptrs + scalar_args + [n]
                c_func(*call_args)
                
            except Exception as e:
                print(f"Failed to call C++ function: {e}")
                print("Falling back to Python computation...")
                # Fallback to Python computation
                self._compute_multi_equations_directly(equations, result_arrays, vector_args, scalar_args, 
                                                     vector_vars, scalar_vars, result_vars, stencil_info)
            
        return wrapper
    
    def _compute_multi_equations_directly(self, equations: List[sp.Eq], result_arrays, vector_args, scalar_args, 
                                        vector_vars, scalar_vars, result_vars, stencil_info):
        """
        Direct computation of multiple equations.
        """
        n = len(result_arrays[0]) if result_arrays else len(vector_args[0])
        
        # Create symbol mapping
        symbol_map = {}
        
        # Map input vector variables
        for i, var in enumerate(vector_vars):
            symbol_map[var] = vector_args[i]
        
        # Map result vector variables (for dependencies between equations)
        for i, var in enumerate(result_vars):
            symbol_map[var] = result_arrays[i]
        
        # Map scalar variables
        for i, var in enumerate(scalar_vars):
            symbol_map[var] = scalar_args[i]
        
        # Determine loop bounds
        if stencil_info.min_offset < 0 or stencil_info.max_offset > 0:
            min_index = max(0, -stencil_info.min_offset)
            max_index = n - stencil_info.max_offset if stencil_info.max_offset > 0 else n
        else:
            min_index = 0
            max_index = n
        
        # Evaluate each equation for each valid index
        for i in range(min_index, max_index):
            for eq_idx, eq in enumerate(equations):
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
                            # Parse the index to get the offset
                            index_expr = indexed_expr.indices[0]
                            offset = self._parse_index_offset(index_expr)
                            
                            # Get the value at the offset position
                            access_index = i + offset
                            if 0 <= access_index < len(symbol_map[base_name]):
                                value = symbol_map[base_name][access_index]
                                expr_subs = expr_subs.subs(indexed_expr, value)
                    
                    # Substitute scalar variables
                    for var, value in symbol_map.items():
                        if not isinstance(value, np.ndarray):
                            expr_subs = expr_subs.subs(sp.Symbol(var), value)
                    
                    # Evaluate the result
                    result_idx = result_vars.index(result_var)
                    result_arrays[result_idx][i] = float(expr_subs.evalf())
                    
                except Exception:
                    # If evaluation fails, set to 0.0
                    result_idx = result_vars.index(result_var)
                    result_arrays[result_idx][i] = 0.0


def genFunc(equations: Union[sp.Eq, List[sp.Eq]]) -> callable:
    """
    Generate a callable function from SymPy equation(s) with multi-equation support.
    
    Args:
        equations: Single SymPy equation or list of equations
        
    Returns:
        Callable function that takes numpy arrays as arguments
    """
    # Handle single equation case
    if isinstance(equations, sp.Eq):
        equations = [equations]
    
    generator = MultiEquationCodeGenerator()
    
    # Generate a unique function name
    eq_strs = [str(eq) for eq in equations]
    combined_str = "|".join(eq_strs)
    func_hash = hashlib.md5(combined_str.encode()).hexdigest()[:8]
    func_name = f"cpp_multi_{func_hash}"
    
    # Generate and compile C++ code
    cpp_code = generator._generate_cpp_code(equations, func_name)
    so_file = generator._compile_cpp_code(cpp_code, func_name)
    
    # Create Python wrapper
    wrapper = generator._create_python_wrapper(so_file, func_name, equations)
    
    return wrapper 