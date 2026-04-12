#include "Tenzor.hpp"

void calc();

using namespace TZ;

int main() {
	std::vector<std::vector<float>> a = {{1, 2}, {2, 3}};
	Matrix<float> k = Tensor<float>::fromSTDVec(a);
	Matrix<float> m = k.copyTo(GPU);

	// k.row(0) = Tensor<float>::fromSTDVec(std::vector<float>{4, 4});
	// Tensor<float> t = k.toTensor();
	// m += m;

	// k = m.copyTo(CPU);
	// std::cout << k;
	//   calc();
}
