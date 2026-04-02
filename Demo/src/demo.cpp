#include "Tenzor_math.hpp"

int main() {
	std::vector<std::vector<float>> a = {{1, 2}, {2, 3}};
	TZ::Matrix<float> k = TZ::internal::TensorIMPL<float>::fromSTDVec(a);
	k.col(0) = TZ::internal::TensorIMPL<float>::fromSTDVec(std::vector<float>{4, 4});
	std::cout << k;
}
