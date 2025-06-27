#include <cmath>

extern "C" {

void cpp_multidim_d0125679(double* result_grad_x,
               double* result_grad_y,
               double* result_grad_mag,
               const double* u,
               const int rows,
               const int cols,
               const double dx)
{
    // Multi-dimensional stencil bounds
    const int min_i = 0;
    const int max_i = rows - 6;
    const int min_j = 0;
    const int max_j = cols - 8;

    // Generated multi-dimensional loop
    for(int i = min_i; i < max_i; i++) {
        for(int j = min_j; j < max_j; j++) {
            result_grad_x[i * cols + j] = (-u[(4) * cols + (7)] + u[(6) * cols + (7)])/(2*dx);
            result_grad_y[i * cols + j] = (-u[(5) * cols + (6)] + u[(5) * cols + (8)])/(2*dx);
            result_grad_mag[i * cols + j] = sqrt(pow(result_grad_x[(5) * cols + (7)], 2) + pow(result_grad_y[(5) * cols + (7)], 2));
        }
    }
}

}