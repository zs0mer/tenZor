#include "Tenzor_math.hpp"
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

	TZ::Matrix<int> t = TZ::Tensor<int>::fromSTDVec(v).tensor_();
	std::vector<uint64_t> l(2);

	{
		TZ::internal::Timer timer("vector");

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
		TZ::internal::Timer timer("tensor");
		long long g = 0;
		for (uint64_t i = 0; i < n; i++) {
			l[1] = i;
			for (uint64_t j = 0; j < m; j++) {
				l[0] = j;
				// g += *(t.data() + i * m + j);
				// g += t[i][j].get();
				// g += *(t.data() + i * m + j);
				g += t.at(i, j);
			}
		}

		std::cout << "sum: " << g << std::endl;
	}
}