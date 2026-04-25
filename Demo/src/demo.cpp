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
	Matrix<float> A_cpu(500, 500, CPU);
	Matrix<float> B_cpu(500, 500, CPU);

	std::cout << "Matrix A (CPU):\n" << "A_cpu" << "\n\n";
	std::cout << "Matrix B (CPU):\n" << "B_cpu" << "\n\n";

	// 2. Move the matrices to the GPU
	// (Assuming Matrix wrapper allows accessing the underlying tensor to copy)
	Matrix<float> A_gpu(A_cpu.copyTo(GPU));
	Matrix<float> B_gpu(B_cpu.copyTo(GPU));

	// 3. Perform Matrix Multiplication on the GPU
	// This will route to your GPU matmul (and cuBLAS if you implemented the wrapper)
	std::cout << "Calculating A * B on GPU...\n\n";
	Matrix<float> C_gpu;
	{
		internal::Timer timer("GPU Matmul");
		C_gpu = matmul(A_cpu, B_cpu);
	}

	// 4. Copy the result back to the CPU to print it
	Matrix<float> C_cpu(C_gpu.copyTo(CPU));

	// Expected Result:
	// [58,  64 ]
	// [139, 154]
	std::cout << "Result Matrix C (CPU):\n" << "C_cpu" << "\n";

	return 0;
}
