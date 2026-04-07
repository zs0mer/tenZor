#include "Tenzor_math.hpp"
#include "math_classes.hpp"

int main() {
	std::vector<std::vector<float>> a = {{1, 2}, {2, 3}};
	TZ::Matrix<float> k = TZ::Tensor<float>::fromSTDVec(a);
	k.row(0) = TZ::Tensor<float>::fromSTDVec(std::vector<float>{4, 4});
	TZ::Tensor<float> t = k.toTensor();
	std::cout << TZ::Scalar<float>(t);
}
