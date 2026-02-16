#include "Tenzor.hpp"
#include <iostream>

int main() {
	std::vector<std::vector<int>> h = {{1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}};
	TZ::Tensor<int> t;
	std::cout << t.scalar() << std::endl;
}