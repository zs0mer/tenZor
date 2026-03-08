#include "Tenzor.hpp"
#include <iostream>
#include <random>

static std::mt19937 rng(345678);

int main() {
	const int n = 100000;
	const int m = 18;
	std::vector<int> v(n, 0);


	for (int i = 0; i < n; i++) {
		v[i] = (rng() % 1024 * 20) + 1;
	}


	TZ::mem::salloc& aalloc = TZ::mem::salloc::instance();
	{
		TZ::_Timer timer("salloc");
		std::vector<void*> space(n);

		for (int k = 0; k < m; k++) {
			for (int i = 0; i < n; i++)
				space[i] = aalloc.allocate(v[i]);

			for (int i = 0; i < n; i++)
				aalloc.deallocate(space[i], v[i]);
		}
	}
}