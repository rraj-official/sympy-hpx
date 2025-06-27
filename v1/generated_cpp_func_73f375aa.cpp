#include <vector>
#include <cassert>
#include <cmath>

extern "C" {

void cpp_func_73f375aa(std::vector<double>& vr,
               const std::vector<double>& va,
               const std::vector<double>& vb,
               const std::vector<double>& vc,
               const double& sd)
{
    const int n = va.size();
    assert(n == vb.size());
    assert(n == vc.size());
    assert(n == vr.size());

    // Generated loop
    for(int i = 0; i < n; i++) {
        vr[i] = sd*va[i] + vb[i]*vc[i];
    }
}

}