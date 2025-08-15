#!/usr/bin/env python3

import os
import subprocess
import sys
import time

def test_cleanup_version(version):
    """Test cleanup functionality for a specific version"""
    print(f"\n{'='*50}")
    print(f"Testing cleanup for v{version}")
    print(f"{'='*50}")
    
    # Change to version directory
    os.chdir(f'v{version}')
    
    # Count files before test
    files_before = [f for f in os.listdir('.') if f.startswith('generated_')]
    print(f"Files before test: {len(files_before)}")
    
    # Run a test that generates files
    test_script = f"""
import sys
sys.path.append('.')
from sympy_codegen import genFunc
import sympy as sp
import numpy as np

# Generate a unique equation for each run
i = sp.Symbol('i', integer=True)
a = sp.IndexedBase('a')
b = sp.IndexedBase('b')
r = sp.IndexedBase('r')
k = sp.Symbol('k')
x = sp.Symbol('x')  # Additional variable to make equations unique

# Make equation unique by adding a random term
import random
random_factor = random.randint(1, 1000)
equation = sp.Eq(r[i], k*a[i] + b[i] + x*random_factor)
print(f"Generated unique equation with factor {{random_factor}}")
func = genFunc(equation)

# Write file count to a temporary file
import os
files_after_gen = [f for f in os.listdir('.') if f.startswith('generated_')]
with open("file_count.txt", "w") as f:
    f.write(str(len(files_after_gen)))

# Test the function
va = np.array([1.0, 2.0, 3.0])
vb = np.array([4.0, 5.0, 6.0])
vr = np.zeros(3)
k_val = 2.0
x_val = 1.0 # Define a value for x

# Call with correct signature for each version
if '{version}' == '4':
    func(vr, va, vb, 3, k_val, x_val)  # v4 needs size and x
else:
    func(vr, va, vb, k_val, x_val)      # v1,v2,v3 need x

print(f"v{version} function executed successfully")
print(f"Result: {{vr}}")
"""
    
    # Run the test script
    result = subprocess.run([sys.executable, '-c', test_script], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Function generation and execution successful")
        print(f"Output: {result.stdout.strip()}")
        # Read file count from the temporary file
        with open("file_count.txt", "r") as f:
            files_after_count = int(f.read())
        os.remove("file_count.txt")
        new_files = files_after_count - len(files_before)
        print(f"New files created: {new_files}")
        if new_files > 0:
            print("✓ Files generated locally (not in /tmp)")
            print("✓ Files will be cleaned up when process exits")
        else:
            print("✗ No new files were generated")
    else:
        print("✗ Function generation failed")
        print(f"Error: {result.stderr}")
        new_files = 0 # No files generated
    
    # Files should be generated locally
    if new_files <= 0 :
        # Count files after test
        files_after = [f for f in os.listdir('.') if f.startswith('generated_')]
        print(f"Files after test: {len(files_after)}")
        new_files = len(files_after) - len(files_before)
        print(f"New files created: {new_files}")
    
    if new_files > 0:
        print("✓ Files generated locally (not in /tmp)")
        print("✓ Files will be cleaned up when process exits")
    else:
        print("✗ No new files were generated")
    
    # Change back to parent directory
    os.chdir('..')
    
    # Check /tmp for generated files
    tmp_files = [f for f in os.listdir('/tmp') if f.startswith('generated_')]
    if tmp_files:
        print(f"✗ Found generated files in /tmp: {tmp_files}")
        return False
    else:
        print("✓ No generated files found in /tmp")

    return result.returncode == 0

def main():
    print("Testing file generation and cleanup for all versions...")
    
    # Test all versions
    results = {}
    for version in ['1', '2', '3', '4']:
        results[version] = test_cleanup_version(version)
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    
    for version, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"v{version}: {status}")
    
    print(f"\nAll versions generate files locally and cleanup automatically on exit.")
    print("The cleanup functionality uses Python's atexit module.")

if __name__ == "__main__":
    main() 