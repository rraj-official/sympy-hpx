#include <cmath>

extern "C" {

void cpp_multidim_886f2f35(double* result_r,
               const double* a,
               const double* b,
               const int rows,
               const int cols,
               const double k)
{
    // Multi-dimensional stencil bounds
    const int min_i = 0;
    const int max_i = rows;
    const int min_j = 0;
    const int max_j = cols;

    // Generated multi-dimensional loop
    for(int i = min_i; i < max_i; i++) {
        for(int j = min_j; j < max_j; j++) {
            result_r[i * cols + j] = k*a[(i) * cols + (j)] + b[(i) * cols + (j)];
        }
    }
}

}