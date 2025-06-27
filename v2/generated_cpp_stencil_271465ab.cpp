#include <cmath>

extern "C" {

void cpp_stencil_271465ab(double* result,
               const double* a,
               const double* b,
               const double c,
               const int n)
{

    // Generated stencil loop
    for(int i = 0; i < n; i++) {
        result[i] = c*a[i] + b[i];
    }
}

}