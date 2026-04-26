#include <Tenzor.hpp>
#include "math_classes.hpp"
#define DOCTEST_CONFIG_COLORS
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <vector>

using namespace TZ;

TEST_CASE("math_classes_scalar") {
	Scalar<int> s1(42, CPU);
	CHECK(s1.dim() == 0);
	CHECK(s1.get() == 42);

	s1 = 100;
	CHECK(s1.get() == 100);

	int val = s1; // Implicit conversion
	CHECK(val == 100);

	Scalar<int> s2(0, CPU);
	s2 = s1;
	CHECK(s2.get() == 100);
}

TEST_CASE("math_classes_vector") {
	Vector<float> v(3, CPU);
	CHECK(v.dim() == 1);
	CHECK(v.size() == 3);

	v.at(0) = 1.0f;
	v.at(1) = 2.0f;
	v.at(2) = 3.0f;

	CHECK(v.at(0) == doctest::Approx(1.0f));
	CHECK(v.at(1) == doctest::Approx(2.0f));
	CHECK(v.at(2) == doctest::Approx(3.0f));

	Scalar<float> s = v[1];
	CHECK(s.get() == doctest::Approx(2.0f));
}

TEST_CASE("math_classes_matrix_basic") {
	Matrix<int> m(2, 3, CPU);
	CHECK(m.dim() == 2);
	CHECK(m.rows() == 2);
	CHECK(m.cols() == 3);
	CHECK(m.size() == 6);

	int counter = 0;
	for (uint64_t i = 0; i < m.rows(); ++i) {
		for (uint64_t j = 0; j < m.cols(); ++j) {
			m.at(i, j) = ++counter;
		}
	}

	CHECK(m.at(0, 0) == 1);
	CHECK(m.at(0, 2) == 3);
	CHECK(m.at(1, 0) == 4);
	CHECK(m.at(1, 2) == 6);
}

TEST_CASE("math_classes_matrix_row_col") {
	Matrix<int> m(3, 3, CPU);
	for (uint64_t i = 0; i < 3; ++i) {
		for (uint64_t j = 0; j < 3; ++j) {
			m.at(i, j) = (i * 3) + j;
			// 0 1 2
			// 3 4 5
			// 6 7 8
		}
	}

	Vector<int> r1 = m.row(1);
	CHECK(r1.size() == 3);
	CHECK(r1.at(0) == 3);
	CHECK(r1.at(1) == 4);
	CHECK(r1.at(2) == 5);

	Vector<int> c2 = m.col(2);
	CHECK(c2.size() == 3);
	CHECK(c2.at(0) == 2);
	CHECK(c2.at(1) == 5);
	CHECK(c2.at(2) == 8);

	Vector<int> r_op = m[2]; // operator[] returns a row
	CHECK(r_op.at(0) == 6);
	CHECK(r_op.at(2) == 8);
}

TEST_CASE("math_classes_matrix_swap") {
	Matrix<int> m(2, 2, CPU);
	m.at(0, 0) = 1;
	m.at(0, 1) = 2;
	m.at(1, 0) = 3;
	m.at(1, 1) = 4;

	m.swapRow(0, 1);
	CHECK(m.at(0, 0) == 3);
	CHECK(m.at(0, 1) == 4);
	CHECK(m.at(1, 0) == 1);
	CHECK(m.at(1, 1) == 2);

	m.swapCol(0, 1);
	CHECK(m.at(0, 0) == 4);
	CHECK(m.at(0, 1) == 3);
	CHECK(m.at(1, 0) == 2);
	CHECK(m.at(1, 1) == 1);
}

TEST_CASE("math_classes_operations") {
	Vector<int> v1(3, CPU);
	v1.setAll(5);

	CHECK(v1.at(0) == 5);
	CHECK(v1.at(1) == 5);
	CHECK(v1.at(2) == 5);

	Scalar<int> sum = v1.sum();
	CHECK(sum.get() == 15);

	Vector<int> v2(3, CPU);
	v2.setAll(2);

	Vector<int> v3 = v1 + v2;
	CHECK(v3.at(0) == 7);

	v3 -= v2;
	CHECK(v3.at(0) == 5);

	Scalar<int> s(10);
	Vector<int> v4 = v1 + s;
	CHECK(v4.at(0) == 15);

	v4 *= Scalar<int>(2);
	CHECK(v4.at(0) == 30);
}

TEST_CASE("math_classes_tensor_from_std_vec") {
	std::vector<std::vector<int>> vec2d = {{1, 2, 3}, {4, 5, 6}};

	Tensor<int> t = Tensor<int>::fromSTDVec(vec2d);

	CHECK(t.dim() == 2);
	CHECK(t.shape()[0] == 2);
	CHECK(t.shape()[1] == 3);

	// Convert to matrix to check elements easily
	Matrix<int> m(t.tensor_());
	CHECK(m.at(0, 0) == 1);
	CHECK(m.at(0, 2) == 3);
	CHECK(m.at(1, 0) == 4);
	CHECK(m.at(1, 2) == 6);
}

TEST_CASE("math_classes_exceptions") {
	Vector<int> v(3, CPU);
	CHECK_THROWS_AS(v.at(3), std::runtime_error);

	Matrix<int> m(2, 2, CPU);
	CHECK_THROWS_AS(m.at(2, 0), std::runtime_error);
	CHECK_THROWS_AS(m.at(0, 2), std::runtime_error);

	CHECK_THROWS_AS(m.row(2), std::runtime_error);
	CHECK_THROWS_AS(m.col(2), std::runtime_error);

	CHECK_THROWS_AS(m.swapRow(0, 2), std::runtime_error);
	CHECK_THROWS_AS(m.swapCol(2, 0), std::runtime_error);

	TZ::internal::TensorIMPL<int> t_1d({3}, CPU);
	CHECK_THROWS_AS(Matrix<int>(t_1d), std::runtime_error);

	TZ::internal::TensorIMPL<int> t_2d({2, 2}, CPU);
	CHECK_THROWS_AS(Vector<int>(t_2d), std::runtime_error);
}
