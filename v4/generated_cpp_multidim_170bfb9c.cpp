#include <cmath>

extern "C" {

void cpp_multidim_170bfb9c(double* result_T_new,
               const double* T,
               const int rows,
               const int cols,
               const double alpha,
               const double dt,
               const double dx)
{
    // Multi-dimensional stencil bounds
    const int min_i = 1;
    const int max_i = rows - 1;
    const int min_j = 1;
    const int max_j = cols - 1;

    // Generated multi-dimensional loop
    for(int i = min_i; i < max_i; i++) {
        for(int j = min_j; j < max_j; j++) {
            result_T_new[i * cols + j] = alpha*dt*(T[(i + 1) * cols + (j)] + T[(i - 1) * cols + (j)] + T[(i) * cols + (j + 1)] + T[(i) * cols + (j - 1)] - 4*T[(i) * cols + (j)])/pow(dx, 2) + T[(i) * cols + (j)];
        }
    }
}

}