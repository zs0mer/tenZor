#include "Tenzor.hpp"

int main() {
	std::vector<std::vector<float>> a = {{1, 2}, {2, 3}};
	TZ::Matrix<float> k = TZ::Tensor<float>::fromSTDVec(a);
	k.col(0) = TZ::Tensor<float>::fromSTDVec(std::vector<float>{4, 4});
	std::cout << k;
}