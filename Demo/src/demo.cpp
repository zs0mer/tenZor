/*#include "Tenzor.hpp"
#include "math_classes.hpp"
#include "operators.hpp"

using namespace TZ;

int main() {
    Vector<float> k = TZ::Tensor<float>::fromSTDVec(std::vector<float>{1, 2, 3, 4});
    k = k.copyTo(GPU);
    Vector<float> z = k;
    // k = k.copyTo(GPU);
    // Matrix<float> l = k.clone();

    // k.row(0) = Tensor<float>::fromSTDVec(std::vector<float>{4, 4});
    // Tensor<float> t = k.toTensor();
    std::cout << "starting\n";

    Scalar p = TZ::dot(k, z);
    // z = z.copyTo(CPU);
    std::cout << p.copyTo(CPU);

    // k = m.copyTo(CPU);
    //   std::cout << k;
    //      calc();
}
*/
#include <iostream>

#include "tensor.hpp"
#include "math_classes.hpp"
#include "operators.hpp"

using namespace TZ;

int main() {
	std::cout << "--- cuBLAS Matrix Multiplication Demo ---\n\n";

	// 1. Create two matrices on the CPU so we can easily fill them with data
	Vector<float> A({1, 2, 3, 4, 5}, CPU);

	std::cout << A.broadcast(6, 5).transpose() << "\n";

	return 0;
}
