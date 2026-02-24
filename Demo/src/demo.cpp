#include "Tenzor.hpp"
#include <iostream>

int main() {
	std::vector<std::vector<std::vector<int>>> h = {};
	TZ::Tensor<int> t(h);
	std::cout << t << std::endl;
}