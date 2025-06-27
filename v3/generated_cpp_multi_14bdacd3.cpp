#include <cmath>

extern "C" {

void cpp_multi_14bdacd3(double* result_r1,
               double* result_r2,
               double* result_r3,
               const double* a,
               const double* b,
               const double k,
               const int n)
{

    // Generated multi-equation loop
    for(int i = 0; i < n; i++) {
        result_r1[i] = k*a[i];
        result_r2[i] = b[i] + result_r1[i];
        result_r3[i] = result_r1[i]*result_r2[i];
    }
}

}