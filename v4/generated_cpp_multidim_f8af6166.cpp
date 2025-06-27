#include <vector>
#include <cassert>
#include <cmath>

extern "C" {

void cpp_multidim_f8af6166(std::vector<double>& vgrad_x,
               std::vector<double>& vgrad_y,
               std::vector<double>& vgrad_mag,
               const std::vector<double>& vu,
               const int& rows,
               const int& cols,
               const double& sdx)
{
    // Multi-dimensional array handling
    const int total_size = rows * cols;
    assert(vgrad_x.size() == total_size);
    assert(vgrad_y.size() == total_size);
    assert(vgrad_mag.size() == total_size);
    assert(vu.size() == total_size);

    // Multi-dimensional stencil bounds
    const int min_i = 1;
    const int max_i = rows - 1;
    const int min_j = 1;
    const int max_j = cols - 1;

    // Generated multi-dimensional loop
    for(int i = min_i; i < max_i; i++) {
        for(int j = min_j; j < max_j; j++) {
            vgrad_x[i * cols + j] = (vu[(i + 1) * cols + (j)] - vu[(i - 1) * cols + (j)])/(2*sdx);
            vgrad_y[i * cols + j] = (vu[(i) * cols + (j + 1)] - vu[(i) * cols + (j - 1)])/(2*sdx);
            vgrad_mag[i * cols + j] = sqrt(pow(vgrad_x[(i) * cols + (j)], 2) + pow(vgrad_y[(i) * cols + (j)], 2));
        }
    }
}

}