#include <cmath>

extern "C" {

void cpp_multi_699c054c(double* result_r1,
               double* result_r2,
               const double* a,
               const double* b,
               const int n)
{
    const int min_index = 1; // from stencil pattern
    const int max_index = n - 2; // from stencil pattern

    // Generated multi-equation loop
    for(int i = min_index; i < max_index; i++) {
        result_r1[i] = a[i + 1] + a[i - 1] + a[i];
        result_r2[i] = b[i + 2] + result_r1[i];
    }
}

}