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

	Vector<int> v = {1, 2, 3};
	Matrix<int> m(v);
	std::cout << m << m.rows();
	return 0;
}
