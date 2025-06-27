#include <vector>
#include <cassert>
#include <cmath>

extern "C" {

void cpp_stencil_a9c48b15(std::vector<double>& vr,
               const std::vector<double>& va,
               const std::vector<double>& vb,
               const std::vector<double>& vc,
               const double& sd)
{
    const int n = va.size();
    assert(n == vb.size());
    assert(n == vc.size());
    assert(n == vr.size());

    const int min_index = 2; // from stencil pattern
    const int max_index = n - 1; // from stencil pattern

    // Generated stencil loop
    for(int i = min_index; i < max_index; i++) {
        vr[i] = sd*va[i] + vb[i + 1]*vc[i - 2];
    }
}

}