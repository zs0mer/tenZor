/*#include "Tenzor.hpp"
#include <iostream>

int main() {
    std::vector<std::vector<int>> h = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    TZ::Tensor<int> t(h);
    int z = t[2][0].get();
    std::cout << t[2] << std::endl;
}*/

#include "Tenzor.hpp"
#include <iostream>
#include <random>

static std::mt19937 rng(345678);

int main() {
	const int n = 1;
	std::vector<int> v(n, 0);
	std::vector<void*> space(n);


	for (int i = 0; i < n; i++) {
		v[i] = (rng() % 1024) + 1;
	}

	TZ::mem::salloc& aalloc = TZ::mem::salloc::instance();
	{
		TZ::_Timer timer("salloc");
		for (int i = 0; i < n; i++)
			space[i] = aalloc.allocate(v[i]);

		// for (int i = 0; i < n; i++)
		// aalloc.deallocate(space[i], v[i]);
	}
}