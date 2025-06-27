#include <cmath>

extern "C" {

void cpp_multi_6f083bab(double* result_r,
               const double* a,
               const double* b,
               const double k,
               const int n)
{

    // Generated multi-equation loop
    for(int i = 0; i < n; i++) {
        result_r[i] = k*a[i] + b[i];
    }
}

}