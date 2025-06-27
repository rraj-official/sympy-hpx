#include <vector>
#include <cassert>
#include <cmath>

extern "C" {

void cpp_multidim_2347862f(std::vector<double>& vlaplacian,
               const std::vector<double>& vu,
               const int& rows,
               const int& cols)
{
    // Multi-dimensional array handling
    const int total_size = rows * cols;
    assert(vlaplacian.size() == total_size);
    assert(vu.size() == total_size);

    // Multi-dimensional stencil bounds
    const int min_i = 1;
    const int max_i = rows - 1;
    const int min_j = 1;
    const int max_j = cols - 1;

    // Generated multi-dimensional loop
    for(int i = min_i; i < max_i; i++) {
        for(int j = min_j; j < max_j; j++) {
            vlaplacian[i * cols + j] = vu[(i + 1) * cols + (j)] + vu[(i - 1) * cols + (j)] + vu[(i) * cols + (j + 1)] + vu[(i) * cols + (j - 1)] - 4*vu[(i) * cols + (j)];
        }
    }
}

}