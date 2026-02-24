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

	{
		TZ::_Timer timer("vector");

		long long g = 0;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				g += v[i][j];
			}
		}

		std::cout << "sum: " << g;
	}

	{
		TZ::_Timer timer("tensor");

		long long g = 0;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				g += t[i][j].get();
			}
		}

		std::cout << "sum: " << g;
	}
}