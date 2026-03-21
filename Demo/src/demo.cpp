#include "Tenzor.hpp"
#include <iostream>
#include <random>

int main() {
	std::vector<std::vector<int>> a = {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}};
	TZ::Tensor<int> t = TZ::Tensor<int>::fromSTDVec(a);
	TZ::Matrix<int> m(t);
	std::cout << m;
}