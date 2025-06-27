#include <cmath>

extern "C" {

void cpp_multidim_a1dabb61(double* result_result,
               const double* coeff_1d,
               const double* field_2d,
               const int rows,
               const int cols)
{
    // Multi-dimensional stencil bounds
    const int min_i = 0;
    const int max_i = rows;
    const int min_j = 0;
    const int max_j = cols;

    // Generated multi-dimensional loop
    for(int i = min_i; i < max_i; i++) {
        for(int j = min_j; j < max_j; j++) {
            result_result[i * cols + j] = coeff_1d[i]*field_2d[(i) * cols + (j)];
        }
    }
}

}