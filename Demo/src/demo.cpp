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
	Matrix<float> A_cpu(2, 3, CPU);
	Matrix<float> B_cpu(3, 2, CPU);

	// Fill Matrix A:
	// [1, 2, 3]
	// [4, 5, 6]
	A_cpu.tensor_().at({0, 0}) = 1.0f;
	A_cpu.tensor_().at({0, 1}) = 2.0f;
	A_cpu.tensor_().at({0, 2}) = 3.0f;
	A_cpu.tensor_().at({1, 0}) = 4.0f;
	A_cpu.tensor_().at({1, 1}) = 5.0f;
	A_cpu.tensor_().at({1, 2}) = 6.0f;

	// Fill Matrix B:
	// [7,  8 ]
	// [9,  10]
	// [11, 12]
	B_cpu.tensor_().at({0, 0}) = 7.0f;
	B_cpu.tensor_().at({0, 1}) = 8.0f;
	B_cpu.tensor_().at({1, 0}) = 9.0f;
	B_cpu.tensor_().at({1, 1}) = 10.0f;
	B_cpu.tensor_().at({2, 0}) = 11.0f;
	B_cpu.tensor_().at({2, 1}) = 12.0f;

	std::cout << "Matrix A (CPU):\n" << A_cpu.tensor_() << "\n\n";
	std::cout << "Matrix B (CPU):\n" << B_cpu.tensor_() << "\n\n";

	// 2. Move the matrices to the GPU
	// (Assuming Matrix wrapper allows accessing the underlying tensor to copy)
	Matrix<float> A_gpu(A_cpu.tensor_().copyTo(GPU));
	Matrix<float> B_gpu(B_cpu.tensor_().copyTo(GPU));

	// 3. Perform Matrix Multiplication on the GPU
	// This will route to your GPU matmul (and cuBLAS if you implemented the wrapper)
	std::cout << "Calculating A * B on GPU...\n\n";
	Matrix<float> C_gpu = matmul(A_gpu, B_gpu);

	// 4. Copy the result back to the CPU to print it
	Matrix<float> C_cpu(C_gpu.tensor_().copyTo(CPU));

	// Expected Result:
	// [58,  64 ]
	// [139, 154]
	std::cout << "Result Matrix C (CPU):\n" << C_cpu.tensor_() << "\n";

	return 0;
}
