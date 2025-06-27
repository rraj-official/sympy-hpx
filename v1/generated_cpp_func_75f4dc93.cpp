#include <cmath>

extern "C" {

void cpp_func_75f4dc93(double* result,
               const double* a,
               const int n)
{
    // Generated loop
    for(int i = 0; i < n; i++) {
        result[i] = 2*a[i];
    }
}

}