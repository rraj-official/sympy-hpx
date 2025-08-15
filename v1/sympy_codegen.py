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
            elif os.path.isdir(file_path): # Also remove build directories
                import shutil
                shutil.rmtree(file_path)
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
    
    def _generate_cpp_code(self, eq: sp.Eq, func_name: str) -> Tuple[str, str]:
        """
        Generate C++ code and CMakeLists.txt for the given SymPy equation.
        """
        vector_vars, scalar_vars, result_var = self._analyze_expression(eq)

        # Build parameter list for C++ functions
        params = [f"double* {result_var}"]
        for var in sorted(vector_vars):
            if var != result_var:
                params.append(f"const double* {var}")
        params.append("int n")
        for var in sorted(scalar_vars):
            params.append(f"const double {var}")
        param_str = ", ".join(params)

        # Build argument list for kernel call
        arg_list = [result_var]
        for var in sorted(vector_vars):
            if var != result_var:
                arg_list.append(var)
        arg_list.append("n")
        arg_list.extend(sorted(scalar_vars))
        arg_list_str = ", ".join(arg_list)

        # Convert SymPy expression to C++
        rhs = self._convert_expression_to_cpp(eq.rhs, vector_vars, scalar_vars, result_var)

        cpp_code = f"""
#include <hpx/init.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <cmath>

int hpx_kernel({param_str})
{{
    hpx::experimental::for_loop(hpx::execution::par, 0, n, [=](std::size_t i) {{
        {result_var}[i] = {rhs};
    }});
    return hpx::finalize();
}}

extern "C" void {func_name}({param_str})
{{
    int argc = 0;
    char *argv[] = {{ nullptr }};
    hpx::start(nullptr, argc, argv);
    hpx::run_as_hpx_thread([&]() {{
        return hpx_kernel({arg_list_str});
    }});
    hpx::post([](){{ hpx::finalize(); }});
    hpx::stop();
}}
"""

        cmake_code = f"""
cmake_minimum_required(VERSION 3.18)
project(sympy_hpx_kernel LANGUAGES CXX)

find_package(HPX REQUIRED)
add_library({func_name} SHARED kernel.cpp)
target_link_libraries({func_name} PRIVATE HPX::hpx)
"""
        return cpp_code, cmake_code

    def _convert_expression_to_cpp(self, expr, vector_vars, scalar_vars, result_var):
        """Converts a SymPy expression to a C++ string."""
        import re
        expr_str = str(expr)
        
        indexed_pattern = r'(\w+)\[([^\]]+)\]'
        def replace_indexed(match):
            var_name = match.group(1)
            index_expr = match.group(2)
            if var_name in vector_vars or var_name == result_var:
                 return f"{var_name}[{index_expr}]"
            return match.group(0)
        expr_str = re.sub(indexed_pattern, replace_indexed, expr_str)

        for var in scalar_vars:
            expr_str = re.sub(r'\b' + re.escape(var) + r'\b', var, expr_str)
            
        power_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*\[[^\]]+\]|\([^)]+\)|\w+)\*\*(\d+(?:\.\d+)?|\([^)]+\)|\w+)'
        def power_replacement(match):
            base, exponent = match.groups()
            return f"pow({base}, {exponent})"
        return re.sub(power_pattern, power_replacement, expr_str)

    def _write_file_if_changed(self, filepath: str, content: str) -> bool:
        """
        Write content to file only if it has changed.
        Returns True if file was written, False if content was unchanged.
        """
        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    existing_content = f.read()
                if existing_content == content:
                    return False  # Content unchanged, no need to write
            except (IOError, OSError):
                pass  # If we can't read, we'll write anyway
        
        with open(filepath, "w") as f:
            f.write(content)
        return True

    def _compile_cpp_code(self, cpp_code_pair: Tuple[str, str], func_name: str) -> str:
        """
        Compile the C++ code into a shared library using CMake.
        Only recompiles if source files have actually changed.
        """
        cpp_code, cmake_code = cpp_code_pair
        
        # Create build directory inside tmp folder
        tmp_dir = "tmp"
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        
        build_dir = os.path.join(tmp_dir, f"build_{func_name}")
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)
        
        # Only write files if content has changed
        cpp_path = os.path.join(build_dir, "kernel.cpp")
        cmake_path = os.path.join(build_dir, "CMakeLists.txt")
        
        cpp_changed = self._write_file_if_changed(cpp_path, cpp_code)
        cmake_changed = self._write_file_if_changed(cmake_path, cmake_code)
        
        # Check if shared library already exists and is newer than source files
        so_file = os.path.join(build_dir, f"lib{func_name}.so")
        if not os.path.exists(so_file):
            so_file = os.path.join(build_dir, f"{func_name}.so")  # Fallback name
        
        # Skip compilation if nothing changed and shared library exists
        if not cpp_changed and not cmake_changed and os.path.exists(so_file):
            print(f"HPX kernel for {func_name} is up to date, skipping compilation...")
            _generated_files.add(build_dir)
            return so_file
            
        print(f"Building HPX kernel in {build_dir}...")
        
        cmake_cmd = ["cmake", f"-DHPX_DIR={os.environ.get('HOME')}/hpx-install-system/lib/cmake/HPX", "."]
        
        try:
            subprocess.run(cmake_cmd, cwd=build_dir, check=True, capture_output=True, text=True)
            subprocess.run(["make", "-j"], cwd=build_dir, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"CMake or Make failed for {func_name}:")
            print(f"  CMake command: {' '.join(e.cmd)}")
            print(f"  CMake stderr: {e.stderr}")
            raise

        so_file = os.path.join(build_dir, f"lib{func_name}.so")
        if not os.path.exists(so_file):
             so_file = os.path.join(build_dir, f"{func_name}.so") # Fallback name

        _generated_files.add(build_dir)
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
        
        def wrapper(*args, **kwargs):
            """
            Python wrapper for the compiled C++ function.
            Can be called with positional or keyword arguments.
            Expected keywords: result_var, vector_vars (excluding result), scalar_vars
            """
            if kwargs:
                # Handle keyword arguments
                result_array = np.asarray(kwargs[result_var], dtype=np.float64)
                
                # Vector arguments (excluding result if it's also in vector_vars)
                vector_args = []
                for var in sorted(vector_vars):
                    if var != result_var:
                        vec_array = np.asarray(kwargs[var], dtype=np.float64)
                        vector_args.append(vec_array)
                
                # Scalar arguments
                scalar_args = []
                for var in sorted(scalar_vars):
                    scalar_args.append(float(kwargs[var]))
                    
            else:
                # Handle positional arguments (existing logic)
                if len(args) != 1 + len(vector_vars) - (1 if result_var in vector_vars else 0) + len(scalar_vars):
                    expected = 1 + len(vector_vars) - (1 if result_var in vector_vars else 0) + len(scalar_vars)
                    raise ValueError(f"Expected {{expected}} arguments, got {{len(args)}}")
                
                arg_idx = 0
                
                # Result vector (first argument)
                result_array = np.asarray(args[arg_idx], dtype=np.float64)
                arg_idx += 1
                
                # Vector arguments (excluding result if it's also in vector_vars)
                vector_args = []
                for var in sorted(vector_vars):
                    if var != result_var:
                        vec_array = np.asarray(args[arg_idx], dtype=np.float64)
                        vector_args.append(vec_array)
                        arg_idx += 1
                
                # Scalar arguments
                scalar_args = []
                for var in sorted(scalar_vars):
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
                print(f"Failed to call C++ function: {{e}}")
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
    cpp_code_pair = generator._generate_cpp_code(equation, func_name)
    so_file = generator._compile_cpp_code(cpp_code_pair, func_name)
    
    # Create Python wrapper
    wrapper = generator._create_python_wrapper(so_file, func_name, equation)
    
    return wrapper 