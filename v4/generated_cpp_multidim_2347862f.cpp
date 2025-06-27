#include <cmath>

extern "C" {

void cpp_multidim_2347862f(double* result_laplacian,
               const double* u,
               const int rows,
               const int cols)
{
    // Multi-dimensional stencil bounds
    const int min_i = 1;
    const int max_i = rows - 1;
    const int min_j = 1;
    const int max_j = cols - 1;

    // Generated multi-dimensional loop
    for(int i = min_i; i < max_i; i++) {
        for(int j = min_j; j < max_j; j++) {
            result_laplacian[i * cols + j] = u[(i + 1) * cols + (j)] + u[(i - 1) * cols + (j)] + u[(i) * cols + (j + 1)] + u[(i) * cols + (j - 1)] - 4*u[(i) * cols + (j)];
        }
    }
}

}