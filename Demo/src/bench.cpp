#include "Tenzor.hpp"
#include <iostream>
#include <random>

static std::mt19937 rng(345678);

int main() {
	const int n = 1000;
	const int m = 1000;
	std::vector<std::vector<int>> v(n, std::vector<int>(m));

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			v[i][j] = rng() % 10;
		}
	}

	TZ::Tensor<int> t(v);
	std::array<uint64_t, 2> l;

	{
		TZ::_Timer timer("vector");

		long long g = 0;
		for (int i = 0; i < n; i++) {
			l[1] = i;

			for (int j = 0; j < m; j++) {
				l[0] = j;

				g += v[i][j];
			}
		}

		std::cout << "sum: " << g << std::endl;
	}

	{
		TZ::_Timer timer("tensor");
		long long g = 0;
		for (uint64_t i = 0; i < n; i++) {
			l[1] = i;
			for (uint64_t j = 0; j < m; j++) {
				l[0] = j;
				// g += *(t.data() + i * m + j);
				//  g += t[i][j].get();
				g += t.at(l.begin());
			}
		}

		std::cout << "sum: " << g << std::endl;
	}
}