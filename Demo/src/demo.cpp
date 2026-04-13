#include "Tenzor.hpp"

using namespace TZ;

int main() {
	Vector<float> k(1000);
	Vector<float> m = k.clone();

	// k.row(0) = Tensor<float>::fromSTDVec(std::vector<float>{4, 4});
	// Tensor<float> t = k.toTensor();
	std::cout << "starting";

	float z = dot(m, k);

	std::cout << z;

	// k = m.copyTo(CPU);
	//   std::cout << k;
	//      calc();
}
