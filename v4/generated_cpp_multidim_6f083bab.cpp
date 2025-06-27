#include <vector>
#include <cassert>
#include <cmath>

extern "C" {

void cpp_multidim_6f083bab(std::vector<double>& vr,
               const std::vector<double>& va,
               const std::vector<double>& vb,
               const double& sk)
{
    const int n = vr.size();
    assert(n == va.size());
    assert(n == vb.size());

    // Generated multi-equation loop
    for(int i = 0; i < n; i++) {
        vr[i] = sk*va[i] + vb[i];
    }
}

}