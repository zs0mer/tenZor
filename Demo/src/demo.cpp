#include "Tenzor.hpp"

using namespace TZ;

int main() {
	Vector<float> kC = TZ::Tensor<float>::fromSTDVec(std::vector<float>{1, 2, 3, 4});
	Vector<float> lC = TZ::Tensor<float>::fromSTDVec(std::vector<float>{2, 1, 1, 5});

	Vector<float> kG = kC;
	Vector<float> lG = lC;

	// k.row(0) = Tensor<float>::fromSTDVec(std::vector<float>{4, 4});
	// Tensor<float> t = k.toTensor();
	std::cout << "starting\n";

	float z = dot(kG, lG);

	std::cout << z;

	// k = m.copyTo(CPU);
	//   std::cout << k;
	//      calc();
}
