#include <vector>
#include <cassert>
#include <cmath>

extern "C" {

void cpp_multidim_5de8db9b(std::vector<double>& vr,
               const std::vector<double>& va,
               const int& rows,
               const int& cols,
               const int& depth,
               const double& sc)
{
    // Multi-dimensional array handling
    const int total_size = rows * cols * depth;
    assert(vr.size() == total_size);
    assert(va.size() == total_size);

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
                vr[i * cols * depth + j * depth + k] = sc*va[(i) * cols * depth + (j) * depth + (k)];
            }
        }
    }
}

}