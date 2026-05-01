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
	/*
	    Matrix<float> M = {{1.f, 2.f}, {3.f, 4.f}};
	    Matrix<float> N = {{5.f, 6.f}, {7.f, 8.f}};
	    Matrix<float> Mt = M.transpose();

	    Matrix<float> gMt = Mt.copyTo(GPU);
	    Matrix<float> gN = N.copyTo(GPU);
	    Matrix<float> gc = matmul(gMt, gN);
	    Matrix<float> back = gc.copyTo(CPU);

	*/
	Matrix<float> M = {{1.f, 2.f}, {3.f, 4.f}};
	Matrix<float> C = {{0, 0}, {0, 0}};
	M = M.transpose();
	TZ::impl::TensorIMPL<float>::apply(M.tensor_(), C.tensor_(), TZ::impl::Copy<float>{});

	std::cout << M.tensor_().strides()[0] << " " << M.tensor_().strides()[1] << "\n";
	std::cout << C.tensor_().strides()[0] << " " << C.tensor_().strides()[1] << "\n";
}
