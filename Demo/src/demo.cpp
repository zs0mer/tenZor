#include "math_classes.hpp"
#include <iostream>
#include <Tenzor.hpp>

using namespace TZ;

int main() {
	Matrix<float> m({{2, 3}, {4, 5}});
	Vector<float> v1({7, 2});
	Vector<float> v2 = Vector<float>({9, 1}).copyTo(GPU);

	std::cout << matmul(m.copyTo(GPU), matmul(Matrix(v1.copyTo(GPU)), v2.transpose())).copyTo(CPU);

	//[
	//[180, 20],
	//[342, 38]
	//]

	return 0;
}
