#include <cmath>

extern "C" {

void cpp_multidim_6f083bab(double* result_r,
               const double* a,
               const double* b,
               const int n,
               const double k)
{

    // Generated multi-equation loop
    for(int i = 0; i < n; i++) {
        result_r[i] = k*a[i] + b[i];
    }
}

}