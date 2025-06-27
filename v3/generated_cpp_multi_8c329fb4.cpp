#include <vector>
#include <cassert>
#include <cmath>

extern "C" {

void cpp_multi_8c329fb4(std::vector<double>& vr,
               std::vector<double>& vr2,
               const std::vector<double>& va,
               const std::vector<double>& vb,
               const std::vector<double>& vc,
               const double& sd)
{
    const int n = vr.size();
    assert(n == vr2.size());
    assert(n == va.size());
    assert(n == vb.size());
    assert(n == vc.size());

    const int min_index = 2; // from stencil pattern
    const int max_index = n - 1; // from stencil pattern

    // Generated multi-equation loop
    for(int i = min_index; i < max_index; i++) {
        vr[i] = sd*va[i] + vb[i + 1]*vc[i - 2];
        vr2[i] = pow(va[i], 2) + vr[i];
    }
}

}