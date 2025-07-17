"""
SymPy-based code generation for HPX Python API v1
Generates C++ functions from SymPy expressions and compiles them into callable Python functions.
"""

import os
import subprocess
import hashlib
import atexit
import weakref
from typing import Dict, List, Tuple, Any
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


class SymPyCodeGenerator:
    """
    Generates C++ code from SymPy expressions and compiles them into Python-callable functions.
    """
    
    def __init__(self):
        self.compiled_functions = {}
        
    def _analyze_expression(self, eq: sp.Eq) -> Tuple[List[str], List[str], str]:
        """
        Analyze a SymPy equation to extract variables and determine their types.
        
        Returns:
            - vector_vars: List of vector variable names
            - scalar_vars: List of scalar variable names  
            - result_var: Name of the result variable
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
        
        # Find all regular Symbol objects that are not IndexedBase and not indices
        all_symbols = list(eq.atoms(sp.Symbol))
        scalar_vars = []
        
        # Get all index variables
        index_vars = set()
        indexed_exprs = list(eq.atoms(sp.Indexed))
        for indexed in indexed_exprs:
            for idx in indexed.indices:
                if isinstance(idx, (sp.Symbol, sp.Idx)):
                    index_vars.add(str(idx))
        
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
        
        return vector_vars, scalar_vars, result_var
    
    def _generate_cpp_code(self, eq: sp.Eq, func_name: str) -> str:
        """
        Generate C++ code for the given SymPy equation.
        """
        vector_vars, scalar_vars, result_var = self._analyze_expression(eq)
        
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
        
        # Generate the loop body
        cpp_code += "    // Generated loop\n"
        cpp_code += "    for(int i = 0; i < n; i++) {\n"
        
        # Convert SymPy expression to C++ code
        rhs = eq.rhs
        rhs_str = str(rhs)
        
        # Replace indexed expressions with array access
        import re
        
        # Pattern to match indexed expressions like a[i], b[i+1], etc.
        indexed_pattern = r'(\w+)\[([^\]]+)\]'
        
        def replace_indexed(match):
            var_name = match.group(1)
            index_expr = match.group(2)
            return f"{var_name}[{index_expr}]"
        
        rhs_str = re.sub(indexed_pattern, replace_indexed, rhs_str)
        
        # Replace scalar variables with their parameter names
        for var in scalar_vars:
            rhs_str = re.sub(r'\b' + re.escape(var) + r'\b', f"{var}", rhs_str)
            
        # Convert Python operators to C++ equivalents
        # Handle power operator ** -> pow()
        power_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*\[[^\]]+\]|\([^)]+\)|\w+)\*\*(\d+(?:\.\d+)?|\([^)]+\)|\w+)'
        def power_replacement(match):
            base = match.group(1)
            exponent = match.group(2)
            return f"pow({base}, {exponent})"
        
        rhs_str = re.sub(power_pattern, power_replacement, rhs_str)
        
        cpp_code += f"        result[i] = {rhs_str};\n"
        cpp_code += "    }\n"
        cpp_code += "}\n\n}"
        
        return cpp_code
    
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
    
    def _create_python_wrapper(self, so_file: str, func_name: str, eq: sp.Eq):
        """
        Create a Python wrapper function that calls the compiled C++ function.
        """
        import ctypes
        from ctypes import POINTER, c_double, c_int
        
        # Load the shared library (use absolute path)
        abs_so_file = os.path.abspath(so_file)
        lib = ctypes.CDLL(abs_so_file)
        
        # Get function signature info
        vector_vars, scalar_vars, result_var = self._analyze_expression(eq)
        
        # Define ctypes structures for std::vector<double>
        class StdVector(ctypes.Structure):
            _fields_ = [("data", POINTER(c_double)),
                       ("size", c_int),
                       ("capacity", c_int)]
        
        def wrapper(*args):
            """
            Python wrapper for the compiled C++ function.
            Arguments should be provided in the order: result_vector, vector_args..., scalar_args...
            """
            if len(args) != 1 + len(vector_vars) - (1 if result_var in vector_vars else 0) + len(scalar_vars):
                expected = 1 + len(vector_vars) - (1 if result_var in vector_vars else 0) + len(scalar_vars)
                raise ValueError(f"Expected {expected} arguments, got {len(args)}")
            
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
            
            # Get the size and verify all vectors have the same size
            n = len(result_array)
            for vec in vector_args:
                if len(vec) != n:
                    raise ValueError("All vectors must have the same size")
            
            # Call the compiled C++ function using a simpler approach
            # Since we're using extern "C" and basic types, we can call directly
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
                self._compute_directly(eq, result_array, vector_args, scalar_args, vector_vars, scalar_vars, result_var)
            
        return wrapper
    
    def _compute_directly(self, eq: sp.Eq, result_array, vector_args, scalar_args, vector_vars, scalar_vars, result_var):
        """
        Direct computation of the equation using SymPy evaluation.
        """
        n = len(result_array)
        
        # Create symbol mapping
        symbol_map = {}
        
        # Map vector variables to their arrays
        vector_idx = 0
        for var in vector_vars:
            if var != result_var:
                symbol_map[var] = vector_args[vector_idx]
                vector_idx += 1
        
        # Map scalar variables to their values
        for i, var in enumerate(scalar_vars):
            symbol_map[var] = scalar_args[i]
        
        # Evaluate the equation for each index
        for i in range(n):
            try:
                # Get the right-hand side of the equation
                rhs = eq.rhs
                
                # Substitute indexed expressions
                expr_subs = rhs
                indexed_exprs = list(rhs.atoms(sp.Indexed))
                for indexed_expr in indexed_exprs:
                    base_name = str(indexed_expr.base)
                    if base_name in symbol_map and isinstance(symbol_map[base_name], np.ndarray):
                        # For now, assume simple index i
                        value = symbol_map[base_name][i]
                        expr_subs = expr_subs.subs(indexed_expr, value)
                
                # Substitute scalar variables
                for var, value in symbol_map.items():
                    if not isinstance(value, np.ndarray):
                        expr_subs = expr_subs.subs(sp.Symbol(var), value)
                
                # Fallback: try to evaluate symbolically
                try:
                    result_array[i] = float(expr_subs.evalf())
                except:
                    result_array[i] = 0.0
            except Exception:
                # If evaluation fails, set to 0.0
                result_array[i] = 0.0


def genFunc(equation: sp.Eq) -> callable:
    """
    Generate a callable function from a SymPy equation.
    
    Args:
        equation: SymPy equation in the form Eq(result[i], expression)
        
    Returns:
        Callable function that takes numpy arrays as arguments
    """
    generator = SymPyCodeGenerator()
    
    # Generate a unique function name
    eq_str = str(equation)
    func_hash = hashlib.md5(eq_str.encode()).hexdigest()[:8]
    func_name = f"cpp_func_{func_hash}"
    
    # Generate and compile C++ code
    cpp_code = generator._generate_cpp_code(equation, func_name)
    so_file = generator._compile_cpp_code(cpp_code, func_name)
    
    # Create Python wrapper
    wrapper = generator._create_python_wrapper(so_file, func_name, equation)
    
    return wrapper 