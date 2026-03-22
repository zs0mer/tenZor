#include "Tenzor.hpp"
#include <iostream>
#include <random>

int main() {
	std::vector<std::vector<std::vector<int>>> a = {{{1, 2, 3}, {1, 2, 3}, {1, 2, 3}},
	                                                {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}},
	                                                {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}}};
	TZ::Tensor<int> t = TZ::Tensor<int>::fromSTDVec(a);
	TZ::Matrix<int> k = t.clone()[1];
	t.apply([](int& a) { a = 0; });
	TZ::Scalar g(3);

	std::cout << TZ::math::matmul(k, k);
}