#include <hpx/init.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#include <hpx/execution.hpp>
#include <vector>
#include <iostream>
#include <numeric>

int hpx_main()
{
    const std::size_t n = 10;
    std::vector<double> a(n), r(n);
    for (std::size_t i = 0; i < n; ++i) a[i] = static_cast<double>(i);

    // r[i] = 2*a[i] in parallel
    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    hpx::for_each(hpx::execution::par,
                  indices.begin(), indices.end(),
                  [&r, &a](std::size_t idx) {
                      r[idx] = 2.0 * a[idx];
                  });

    for (double v : r) std::cout << v << ' ';
    std::cout << '\n';
    
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    return hpx::init(argc, argv);
} 