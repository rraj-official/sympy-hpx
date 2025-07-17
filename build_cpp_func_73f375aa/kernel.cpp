
#include <hpx/init.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <cmath>

int hpx_kernel(double* r, const double* a, const double* b, const double* c, int n, const double d)
{
    hpx::experimental::for_loop(hpx::execution::par, 0, n, [=](std::size_t i) {
        r[i] = d*a[i] + b[i]*c[i];
    });
    return hpx::finalize();
}

extern "C" void cpp_func_73f375aa(double* r, const double* a, const double* b, const double* c, int n, const double d)
{
    hpx::start(0, nullptr);
    hpx::async([&]() {
        return hpx_kernel(r, a, b, c, n, d);
    }).get();
    hpx::stop();
}
