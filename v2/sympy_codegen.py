"""
SymPy-based code generation for HPX Python API v2 - Stencil Operations
Generates C++ functions from SymPy expressions with support for stencil patterns (offset indices).
"""

import os
import subprocess
import tempfile
import hashlib
from typing import Dict, List, Tuple, Any, Set
import sympy as sp
import numpy as np
import re


class StencilInfo:
    """Information about stencil patterns in an expression."""
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


class SymPyStencilCodeGenerator:
    """
    Enhanced code generator that handles stencil operations with offset indices.
    """
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.compiled_functions = {}
        
    def _analyze_expression(self, eq: sp.Eq) -> Tuple[List[str], List[str], str, StencilInfo]:
        """
        Analyze a SymPy equation to extract variables, types, and stencil patterns.
        
        Returns:
            - vector_vars: List of vector variable names
            - scalar_vars: List of scalar variable names  
            - result_var: Name of the result variable
            - stencil_info: Information about stencil patterns
        """
        lhs = eq.lhs
        rhs = eq.rhs
        
        # Extract result variable (left-hand side)
        if isinstance(lhs, sp.Indexed):
            result_var = str(lhs.base)
        else:
            raise ValueError("Left-hand side must be an indexed expression")
            
        # Find all IndexedBase objects - these are our vectors
        indexed_bases = list(eq.atoms(sp.IndexedBase))
        vector_vars = [str(base) for base in indexed_bases]
        vector_var_names = set(vector_vars)
        
        # Analyze stencil patterns
        stencil_info = StencilInfo()
        indexed_exprs = list(eq.atoms(sp.Indexed))
        
        for indexed in indexed_exprs:
            for idx_expr in indexed.indices:
                # Parse index expressions like i, i+1, i-2, etc.
                offset = self._parse_index_offset(idx_expr)
                stencil_info.add_offset(offset)
        
        # Get all index variables
        index_vars = set()
        for indexed in indexed_exprs:
            for idx in indexed.indices:
                if isinstance(idx, (sp.Symbol, sp.Idx)):
                    index_vars.add(str(idx))
                elif hasattr(idx, 'free_symbols'):
                    # Handle expressions like i+1, i-2
                    for sym in idx.free_symbols:
                        if isinstance(sym, (sp.Symbol, sp.Idx)):
                            index_vars.add(str(sym))
        
        # Find all regular Symbol objects that are not IndexedBase and not indices
        all_symbols = list(eq.atoms(sp.Symbol))
        scalar_vars = []
        
        for symbol in all_symbols:
            symbol_name = str(symbol)
            
            # Skip if this symbol name corresponds to an IndexedBase
            if symbol_name in vector_var_names:
                continue
                
            # Skip if this is an index variable
            if symbol_name in index_vars:
                continue
            
            # Add to scalar variables
            scalar_vars.append(symbol_name)
        
        # Remove duplicates and sort
        vector_vars = sorted(list(set(vector_vars)))
        scalar_vars = sorted(list(set(scalar_vars)))
        
        return vector_vars, scalar_vars, result_var, stencil_info
    
    def _parse_index_offset(self, idx_expr) -> int:
        """
        Parse an index expression to extract the offset.
        Examples: i -> 0, i+1 -> 1, i-2 -> -2
        """
        if isinstance(idx_expr, (sp.Symbol, sp.Idx)):
            return 0
        elif isinstance(idx_expr, sp.Add):
            # Handle expressions like i+1, i-2
            offset = 0
            for arg in idx_expr.args:
                if isinstance(arg, sp.Integer):
                    offset += int(arg)
                elif isinstance(arg, sp.Mul) and len(arg.args) == 2:
                    # Handle expressions like -2 (which is Mul(-1, 2))
                    if isinstance(arg.args[0], sp.Integer) and isinstance(arg.args[1], sp.Integer):
                        offset += int(arg.args[0]) * int(arg.args[1])
            return offset
        elif isinstance(idx_expr, sp.Integer):
            return int(idx_expr)
        else:
            # For complex expressions, try to evaluate numerically
            try:
                # Replace any symbols with 0 to get the constant offset
                simplified = idx_expr
                for sym in idx_expr.free_symbols:
                    simplified = simplified.subs(sym, 0)
                return int(simplified)
            except:
                return 0
    
    def _generate_cpp_code(self, eq: sp.Eq, func_name: str) -> str:
        """
        Generate C++ code for the given SymPy equation with stencil support.
        """
        vector_vars, scalar_vars, result_var, stencil_info = self._analyze_expression(eq)
        
        # Generate function signature with raw pointers for easier ctypes integration
        cpp_code = f"""#include <cmath>

extern "C" {{

void {func_name}("""
        
        # Add result parameter first (raw pointer)
        cpp_code += f"double* result"
        
        # Add vector parameters (raw pointers, alphabetically ordered)
        for var in vector_vars:
            if var != result_var:  # Don't duplicate result variable
                cpp_code += f",\n               const double* {var}"
        
        # Add scalar parameters (alphabetically ordered) 
        for var in scalar_vars:
            cpp_code += f",\n               const double {var}"
            
        # Add size parameter
        cpp_code += ",\n               const int n"
        cpp_code += ")\n{\n"
        
        # Generate stencil bounds
        if stencil_info.offsets and (stencil_info.min_offset < 0 or stencil_info.max_offset > 0):
            min_bound, max_bound = stencil_info.get_loop_bounds("n")
            cpp_code += f"    const int min_index = {min_bound}; // from stencil pattern\n"
            cpp_code += f"    const int max_index = {max_bound}; // from stencil pattern\n"
            loop_start = "min_index"
            loop_end = "max_index"
        else:
            loop_start = "0"
            loop_end = "n"
        
        # Generate the loop body
        cpp_code += f"\n    // Generated stencil loop\n"
        cpp_code += f"    for(int i = {loop_start}; i < {loop_end}; i++) {{\n"
        
        # Convert SymPy expression to C++ code
        rhs = eq.rhs
        rhs_str = self._convert_expression_to_cpp(rhs, vector_vars, scalar_vars, result_var)
            
        cpp_code += f"        result[i] = {rhs_str};\n"
        cpp_code += "    }\n"
        cpp_code += "}\n\n}"
        
        return cpp_code
    
    def _convert_expression_to_cpp(self, expr, vector_vars: List[str], scalar_vars: List[str], result_var: str) -> str:
        """
        Convert a SymPy expression to C++ code, handling stencil patterns.
        """
        expr_str = str(expr)
        
        # Handle stencil patterns - replace indexed expressions with offsets
        # Pattern: variable[i+offset] or variable[i-offset] or variable[i]
        indexed_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\[([^\]]+)\]'
        
        def replace_indexed(match):
            var_name = match.group(1)
            index_expr = match.group(2)
            
            if var_name in vector_vars and var_name != result_var:
                # Parse the index expression
                if index_expr == 'i':
                    return f"{var_name}[i]"
                else:
                    # Handle expressions like i+1, i-2
                    # Simple parsing for common patterns
                    if '+' in index_expr:
                        parts = index_expr.split('+')
                        if len(parts) == 2 and parts[0].strip() == 'i':
                            offset = parts[1].strip()
                            return f"{var_name}[i + {offset}]"
                    elif '-' in index_expr and not index_expr.startswith('-'):
                        parts = index_expr.split('-')
                        if len(parts) == 2 and parts[0].strip() == 'i':
                            offset = parts[1].strip()
                            return f"{var_name}[i - {offset}]"
                    
                    # Fallback: use the expression as-is (might need more complex parsing)
                    return f"{var_name}[{index_expr}]"
            
            return match.group(0)  # Return unchanged if not a vector variable
        
        expr_str = re.sub(indexed_pattern, replace_indexed, expr_str)
        
        # Replace scalar variables with their parameter names
        for var in scalar_vars:
            expr_str = re.sub(r'\b' + re.escape(var) + r'\b', f"{var}", expr_str)
            
        # Convert Python operators to C++ equivalents
        # Handle power operator ** -> pow()
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
    
    def _create_python_wrapper(self, so_file: str, func_name: str, eq: sp.Eq):
        """
        Create a Python wrapper function that calls the compiled C++ function.
        """
        import ctypes
        from ctypes import POINTER, c_double, c_int
        
        # Load the shared library
        lib = ctypes.CDLL(so_file)
        
        # Get function signature info
        vector_vars, scalar_vars, result_var, stencil_info = self._analyze_expression(eq)
        
        def wrapper(*args):
            """
            Python wrapper for the compiled C++ stencil function.
            Arguments should be provided in the order: result_vector, vector_args..., scalar_args...
            """
            expected_args = 1 + len(vector_vars) - (1 if result_var in vector_vars else 0) + len(scalar_vars)
            if len(args) != expected_args:
                raise ValueError(f"Expected {expected_args} arguments, got {len(args)}")
            
            arg_idx = 0
            
            # Result vector (first argument)
            result_array = np.asarray(args[arg_idx], dtype=np.float64)
            arg_idx += 1
            
            # Vector arguments (excluding result if it's also in vector_vars)
            vector_args = []
            for var in vector_vars:
                if var != result_var:
                    vec_array = np.asarray(args[arg_idx], dtype=np.float64)
                    vector_args.append(vec_array)
                    arg_idx += 1
            
            # Scalar arguments
            scalar_args = []
            for var in scalar_vars:
                scalar_args.append(float(args[arg_idx]))
                arg_idx += 1
            
            # Get the size and verify bounds
            n = len(result_array)
            
            # Verify all vectors have the same size
            for vec in vector_args:
                if len(vec) != n:
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
                result_ptr = result_array.ctypes.data_as(POINTER(c_double))
                vector_ptrs = [vec.ctypes.data_as(POINTER(c_double)) for vec in vector_args]
                
                # Set up argument types for the C function
                argtypes = [POINTER(c_double)]  # result vector
                for _ in vector_args:
                    argtypes.append(POINTER(c_double))  # input vectors
                for _ in scalar_args:
                    argtypes.append(c_double)  # scalar arguments
                argtypes.append(c_int)  # size parameter
                
                c_func.argtypes = argtypes
                c_func.restype = None
                
                # Call the C++ function
                call_args = [result_ptr] + vector_ptrs + scalar_args + [n]
                c_func(*call_args)
                
            except Exception as e:
                print(f"Failed to call C++ function: {e}")
                print("Falling back to Python computation...")
                # Fallback to Python computation
                self._compute_stencil_directly(eq, result_array, vector_args, scalar_args, 
                                             vector_vars, scalar_vars, result_var, stencil_info)
            
        return wrapper
    
    def _compute_stencil_directly(self, eq: sp.Eq, result_array, vector_args, scalar_args, 
                                vector_vars, scalar_vars, result_var, stencil_info):
        """
        Direct computation of stencil operations.
        """
        n = len(result_array)
        
        # Create symbol mapping
        symbol_map = {}
        
        # Map vector variables
        vec_idx = 0
        for var in vector_vars:
            if var != result_var:
                symbol_map[var] = vector_args[vec_idx]
                vec_idx += 1
        
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
        
        # Evaluate the expression for each valid index
        rhs = eq.rhs
        
        for i in range(min_index, max_index):
            try:
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
                result_array[i] = float(expr_subs.evalf())
                
            except Exception as e:
                # Fallback for complex expressions
                result_array[i] = 0.0


def genFunc(equation: sp.Eq) -> callable:
    """
    Generate a callable function from a SymPy equation with stencil support.
    
    Args:
        equation: SymPy equation in the form Eq(result[i], expression)
        
    Returns:
        Callable function that takes numpy arrays as arguments
    """
    generator = SymPyStencilCodeGenerator()
    
    # Generate a unique function name
    eq_str = str(equation)
    func_hash = hashlib.md5(eq_str.encode()).hexdigest()[:8]
    func_name = f"cpp_stencil_{func_hash}"
    
    # Generate and compile C++ code
    cpp_code = generator._generate_cpp_code(equation, func_name)
    so_file = generator._compile_cpp_code(cpp_code, func_name)
    
    # Create Python wrapper
    wrapper = generator._create_python_wrapper(so_file, func_name, equation)
    
    return wrapper 