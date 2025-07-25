"""
SymPy-based code generation for HPX Python API v3 - Multiple Equations
Generates C++ functions from multiple SymPy equations processed together for efficiency.
"""

import os
import subprocess
import tempfile
import hashlib
from typing import Dict, List, Tuple, Any, Set, Union
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


class MultiEquationCodeGenerator:
    """
    Enhanced code generator that handles multiple equations processed together.
    """
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
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
        
        # Process each equation
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
            for base in indexed_bases:
                all_vector_vars.add(str(base))
            
            # Analyze stencil patterns in this equation
            indexed_exprs = list(eq.atoms(sp.Indexed))
            for indexed in indexed_exprs:
                for idx_expr in indexed.indices:
                    offset = self._parse_index_offset(idx_expr)
                    stencil_info.add_offset(offset)
            
            # Get index variables from this equation
            index_vars = set()
            for indexed in indexed_exprs:
                for idx in indexed.indices:
                    if isinstance(idx, (sp.Symbol, sp.Idx)):
                        index_vars.add(str(idx))
                    elif hasattr(idx, 'free_symbols'):
                        for sym in idx.free_symbols:
                            if isinstance(sym, (sp.Symbol, sp.Idx)):
                                index_vars.add(str(sym))
            
            # Find scalar variables in this equation
            all_symbols = list(eq.atoms(sp.Symbol))
            for symbol in all_symbols:
                symbol_name = str(symbol)
                
                # Skip if this symbol name corresponds to an IndexedBase
                if symbol_name in all_vector_vars:
                    continue
                    
                # Skip if this is an index variable
                if symbol_name in index_vars:
                    continue
                
                # Add to scalar variables
                all_scalar_vars.add(symbol_name)
        
        # Separate input vectors from result vectors
        input_vector_vars = []
        for var in sorted(all_vector_vars):
            if var not in result_vars:
                input_vector_vars.append(var)
        
        # Convert to sorted lists
        vector_vars = sorted(input_vector_vars)
        scalar_vars = sorted(list(all_scalar_vars))
        result_vars = sorted(list(set(result_vars)))  # Remove duplicates and sort
        
        return vector_vars, scalar_vars, result_vars, stencil_info
    
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
    
    def _generate_cpp_code(self, equations: List[sp.Eq], func_name: str) -> str:
        """
        Generate C++ code for multiple SymPy equations.
        """
        vector_vars, scalar_vars, result_vars, stencil_info = self._analyze_equations(equations)
        
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
            cpp_code += f"std::vector<double>& v{var}"
        
        # Add input vector parameters (alphabetically ordered)
        for var in vector_vars:
            cpp_code += f",\n               const std::vector<double>& v{var}"
        
        # Add scalar parameters (alphabetically ordered) 
        for var in scalar_vars:
            cpp_code += f",\n               const double& s{var}"
            
        cpp_code += ")\n{\n"
        
        # Add size assertions
        all_vectors = result_vars + vector_vars
        if all_vectors:
            first_vector = all_vectors[0]
            cpp_code += f"    const int n = v{first_vector}.size();\n"
            
            # Add assertions for all vectors
            for var in all_vectors[1:]:
                cpp_code += f"    assert(n == v{var}.size());\n"
        
        # Generate stencil bounds
        if stencil_info.offsets and (stencil_info.min_offset < 0 or stencil_info.max_offset > 0):
            min_bound, max_bound = stencil_info.get_loop_bounds("n")
            cpp_code += f"\n    const int min_index = {min_bound}; // from stencil pattern\n"
            cpp_code += f"    const int max_index = {max_bound}; // from stencil pattern\n"
            loop_start = "min_index"
            loop_end = "max_index"
        else:
            loop_start = "0"
            loop_end = "n"
        
        # Generate the loop body
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
    
    def _convert_expression_to_cpp(self, expr, vector_vars: List[str], scalar_vars: List[str], result_vars: List[str]) -> str:
        """
        Convert a SymPy expression to C++ code, handling stencil patterns and multiple vectors.
        """
        expr_str = str(expr)
        
        # Handle stencil patterns - replace indexed expressions with offsets
        # Pattern: variable[i+offset] or variable[i-offset] or variable[i]
        indexed_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\[([^\]]+)\]'
        
        def replace_indexed(match):
            var_name = match.group(1)
            index_expr = match.group(2)
            
            # Check if this is a vector variable (input or result)
            if var_name in vector_vars or var_name in result_vars:
                # Parse the index expression
                if index_expr == 'i':
                    return f"v{var_name}[i]"
                else:
                    # Handle expressions like i+1, i-2
                    # Simple parsing for common patterns
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
                    
                    # Fallback: use the expression as-is
                    return f"v{var_name}[{index_expr}]"
            
            return match.group(0)  # Return unchanged if not a vector variable
        
        expr_str = re.sub(indexed_pattern, replace_indexed, expr_str)
        
        # Replace scalar variables with their parameter names
        for var in scalar_vars:
            expr_str = re.sub(r'\b' + re.escape(var) + r'\b', f"s{var}", expr_str)
            
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
    
    def _create_python_wrapper(self, so_file: str, func_name: str, equations: List[sp.Eq]):
        """
        Create a Python wrapper function that calls the compiled C++ function.
        """
        import ctypes
        
        # Load the shared library
        lib = ctypes.CDLL(so_file)
        
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
            
            # Direct computation for v3
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
                    
                except Exception as e:
                    # Fallback for complex expressions
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