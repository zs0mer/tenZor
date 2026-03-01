#include "Tenzor.hpp"
#include <iostream>

int main() {
	std::vector<std::vector<int>> h = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
	TZ::Tensor<int> t(h);
	int z = t[2][0].get();
	std::cout << t[2] << std::endl;
}