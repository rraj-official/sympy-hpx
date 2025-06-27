#include <cmath>

extern "C" {

void cpp_multi_8c329fb4(double* result_r,
               double* result_r2,
               const double* a,
               const double* b,
               const double* c,
               const double d,
               const int n)
{
    const int min_index = 2; // from stencil pattern
    const int max_index = n - 1; // from stencil pattern

    // Generated multi-equation loop
    for(int i = min_index; i < max_index; i++) {
        result_r[i] = d*a[i] + b[i + 1]*c[i - 2];
        result_r2[i] = pow(a[i], 2) + result_r[i];
    }
}

}