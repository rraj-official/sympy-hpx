#include <cmath>

extern "C" {

void cpp_stencil_25b4fa00(double* result,
               const double* a,
               const int n)
{
    const int min_index = 0; // from stencil pattern
    const int max_index = n - 1; // from stencil pattern

    // Generated stencil loop
    for(int i = min_index; i < max_index; i++) {
        result[i] = a[i + 1];
    }
}

}