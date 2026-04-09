#include "Tenzor.hpp"
#include <iostream>
#include <random>

static std::mt19937 rng(345678);

int main() {
	const int n = 10000;
	const int m = 1;
	std::vector<int> v(n, 0);


	for (int i = 0; i < n; i++) {
		v[i] = (rng() % (1024 * 8)) + 1;
	}

	{
		TZ::internal::Timer timer("malloc");
		std::vector<void*> space(n);

		for (int k = 0; k < m; k++) {
			for (int i = 0; i < n; i++)
				space[i] = malloc(v[i]);

			for (int i = 0; i < n; i++)
				free(space[i]);
		}
	}

	TZ::mem::Salloc& aalloc = TZ::mem::Salloc::instance();
	{
		TZ::internal::Timer timer("Salloc");
		std::vector<void*> space(n);

		for (int k = 0; k < m; k++) {
			for (int i = 0; i < n; i++)
				space[i] = aalloc.allocate(v[i]);

			for (int i = 0; i < n; i++)
				aalloc.deallocate(space[i], v[i]);
		}
	}
}
