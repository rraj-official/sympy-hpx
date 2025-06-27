#include <vector>
#include <cassert>
#include <cmath>

extern "C" {

void cpp_multidim_886f2f35(std::vector<double>& vr,
               const std::vector<double>& va,
               const std::vector<double>& vb,
               const int& rows,
               const int& cols,
               const double& sk)
{
    // Multi-dimensional array handling
    const int total_size = rows * cols;
    assert(vr.size() == total_size);
    assert(va.size() == total_size);
    assert(vb.size() == total_size);

    // Multi-dimensional stencil bounds
    const int min_i = 0;
    const int max_i = rows;
    const int min_j = 0;
    const int max_j = cols;

    // Generated multi-dimensional loop
    for(int i = min_i; i < max_i; i++) {
        for(int j = min_j; j < max_j; j++) {
            vr[i * cols + j] = sk*va[(i) * cols + (j)] + vb[(i) * cols + (j)];
        }
    }
}

}