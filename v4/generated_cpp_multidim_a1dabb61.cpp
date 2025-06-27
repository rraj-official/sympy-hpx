#include <vector>
#include <cassert>
#include <cmath>

extern "C" {

void cpp_multidim_a1dabb61(std::vector<double>& vresult,
               const std::vector<double>& vcoeff_1d,
               const std::vector<double>& vfield_2d,
               const int& rows,
               const int& cols)
{
    // Multi-dimensional array handling
    const int total_size = rows * cols;
    assert(vresult.size() == total_size);
    assert(vcoeff_1d.size() == rows);
    assert(vfield_2d.size() == total_size);

    // Multi-dimensional stencil bounds
    const int min_i = 0;
    const int max_i = rows;
    const int min_j = 0;
    const int max_j = cols;

    // Generated multi-dimensional loop
    for(int i = min_i; i < max_i; i++) {
        for(int j = min_j; j < max_j; j++) {
            vresult[i * cols + j] = vcoeff_1d[i]*vfield_2d[(i) * cols + (j)];
        }
    }
}

}