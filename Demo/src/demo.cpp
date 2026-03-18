#include "Tenzor.hpp"
#include <iostream>
#include <random>

int main() {
	std::vector<int> a = {1, 2, 3};
	/*TZ::Tensor<int> t(a);
	TZ::Vector<int> vec(t);
	std::cout << vec[0];*/
	TZ::Tensor<int> t = TZ::Tensor<int>::fromNested(a);
	std::cout << t;
}