#include "Tenzor.hpp"

void calc();

using namespace TZ;

int main() {
	Matrix<float> k(100000, 10000);
	Matrix<float> m = k.copyTo(GPU);

	// k.row(0) = Tensor<float>::fromSTDVec(std::vector<float>{4, 4});
	// Tensor<float> t = k.toTensor();
	std::cout << "starting";

	{
		TZ::internal::Timer z("CPU");
		k *= 2;
		k *= 2;
		k *= 2;
		k *= 2;
		k *= 2;
	}

	{
		TZ::internal::Timer z("GPU");
		m *= 2;
		m *= 2;
		m *= 2;
		m *= 2;
		m *= 2;
	}

	// k = m.copyTo(CPU);
	//   std::cout << k;
	//      calc();
}
