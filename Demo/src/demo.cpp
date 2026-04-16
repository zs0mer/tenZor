#include "Tenzor.hpp"

using namespace TZ;

int main() {
	Matrix<float> k =
	    TZ::Tensor<float>::fromSTDVec(std::vector<std::vector<float>>{{1, 2}, {3, 4}});
	k = k.copyTo(GPU);
	Matrix<float> l = k.clone();

	// k.row(0) = Tensor<float>::fromSTDVec(std::vector<float>{4, 4});
	// Tensor<float> t = k.toTensor();
	std::cout << "starting\n";

	Matrix<float> z = matmul(k, l);
	z = z.copyTo(CPU);
	std::cout << z;

	// k = m.copyTo(CPU);
	//   std::cout << k;
	//      calc();
}
