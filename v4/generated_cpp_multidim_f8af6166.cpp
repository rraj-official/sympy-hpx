#include <cmath>

extern "C" {

void cpp_multidim_f8af6166(double* result_grad_x,
               double* result_grad_y,
               double* result_grad_mag,
               const double* u,
               const int rows,
               const int cols,
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
            result_grad_x[i * cols + j] = (u[(i + 1) * cols + (j)] - u[(i - 1) * cols + (j)])/(2*dx);
            result_grad_y[i * cols + j] = (u[(i) * cols + (j + 1)] - u[(i) * cols + (j - 1)])/(2*dx);
            result_grad_mag[i * cols + j] = sqrt(pow(result_grad_x[(i) * cols + (j)], 2) + pow(result_grad_y[(i) * cols + (j)], 2));
        }
    }
}

}