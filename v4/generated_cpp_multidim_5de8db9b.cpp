#include <cmath>

extern "C" {

void cpp_multidim_5de8db9b(double* result_r,
               const double* a,
               const int rows,
               const int cols,
               const int depth,
               const double c)
{
    // Multi-dimensional stencil bounds
    const int min_i = 0;
    const int max_i = rows;
    const int min_j = 0;
    const int max_j = cols;
    const int min_k = 0;
    const int max_k = depth;

    // Generated multi-dimensional loop
    for(int i = min_i; i < max_i; i++) {
        for(int j = min_j; j < max_j; j++) {
            for(int k = min_k; k < max_k; k++) {
                result_r[i * cols * depth + j * depth + k] = c*a[(i) * cols * depth + (j) * depth + (k)];
            }
        }
    }
}

}