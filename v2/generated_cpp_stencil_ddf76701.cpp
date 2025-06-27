#include <cmath>

extern "C" {

void cpp_stencil_ddf76701(double* result,
               const double* a,
               const int n)
{
    const int min_index = 1; // from stencil pattern
    const int max_index = n - 1; // from stencil pattern

    // Generated stencil loop
    for(int i = min_index; i < max_index; i++) {
        result[i] = a[i + 1] + a[i - 1] + a[i];
    }
}

}