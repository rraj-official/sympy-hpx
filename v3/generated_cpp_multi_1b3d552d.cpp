#include <cmath>

extern "C" {

void cpp_multi_1b3d552d(double* result_r1,
               double* result_r2,
               const double* a,
               const double* b,
               const double x,
               const double y,
               const int n)
{

    // Generated multi-equation loop
    for(int i = 0; i < n; i++) {
        result_r1[i] = x*a[i] + y*b[i];
        result_r2[i] = a[i]*result_r1[i];
    }
}

}