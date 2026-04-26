#include <Tenzor.hpp>
#include "operators.hpp"
#include "math_classes.hpp"
#define DOCTEST_CONFIG_COLORS
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <vector>

using namespace TZ;
using namespace TZ::internal;

TEST_CASE("operators_apply_unary_dense") {
	TensorIMPL<int> t({4}, CPU);
	for (int i = 0; i < 4; ++i) {
		t.rawData()[i] = i;
	}

	TensorIMPL<int>::apply(t, Negate<int>{});

	CHECK(t.rawData()[0] == 0);
	CHECK(t.rawData()[1] == -1);
	CHECK(t.rawData()[2] == -2);
	CHECK(t.rawData()[3] == -3);
}

TEST_CASE("operators_apply_binary_dense") {
	TensorIMPL<int> a({3}, CPU);
	TensorIMPL<int> b({3}, CPU);

	for (int i = 0; i < 3; ++i) {
		a.rawData()[i] = i;
		b.rawData()[i] = 10;
	}

	TensorIMPL<int>::apply(a, b, Copy<int>{});

	CHECK(b.rawData()[0] == 0);
	CHECK(b.rawData()[1] == 1);
	CHECK(b.rawData()[2] == 2);
}

TEST_CASE("operators_apply_ternary_dense") {
	TensorIMPL<int> a({2, 2}, CPU);
	TensorIMPL<int> b({2, 2}, CPU);
	TensorIMPL<int> c({2, 2}, CPU);

	for (int i = 0; i < 4; ++i) {
		a.rawData()[i] = i;
		b.rawData()[i] = i * 2;
		c.rawData()[i] = 0;
	}

	TensorIMPL<int>::apply(a, b, c, Add<int>{});

	CHECK(c.rawData()[0] == 0);
	CHECK(c.rawData()[1] == 3);
	CHECK(c.rawData()[2] == 6);
	CHECK(c.rawData()[3] == 9);
}

TEST_CASE("operators_math_dot") {
	Vector<int> v1(3, CPU);
	Vector<int> v2(3, CPU);

	for (int i = 0; i < 3; ++i) {
		v1.at(i) = i + 1; // 1, 2, 3
		v2.at(i) = i + 2; // 2, 3, 4
	}

	// 1*2 + 2*3 + 3*4 = 2 + 6 + 12 = 20
	Scalar<int> res = dot(v1, v2);
	CHECK(res.get() == 20);
}

TEST_CASE("operators_math_matmul") {
	Matrix<int> m1(2, 3, CPU);
	Matrix<int> m2(3, 2, CPU);

	// m1:
	// 1 2 3
	// 4 5 6
	m1.at(0, 0) = 1;
	m1.at(0, 1) = 2;
	m1.at(0, 2) = 3;
	m1.at(1, 0) = 4;
	m1.at(1, 1) = 5;
	m1.at(1, 2) = 6;

	// m2:
	// 7 8
	// 9 10
	// 11 12
	m2.at(0, 0) = 7;
	m2.at(0, 1) = 8;
	m2.at(1, 0) = 9;
	m2.at(1, 1) = 10;
	m2.at(2, 0) = 11;
	m2.at(2, 1) = 12;

	Matrix<int> res = matmul(m1, m2);

	CHECK(res.rows() == 2);
	CHECK(res.cols() == 2);

	// 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
	CHECK(res.at(0, 0) == 58);
	// 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
	CHECK(res.at(0, 1) == 64);
	// 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
	CHECK(res.at(1, 0) == 139);
	// 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
	CHECK(res.at(1, 1) == 154);
}

TEST_CASE("operators_math_transpose") {
	Matrix<int> m(2, 3, CPU);
	m.at(0, 0) = 1;
	m.at(0, 1) = 2;
	m.at(0, 2) = 3;
	m.at(1, 0) = 4;
	m.at(1, 1) = 5;
	m.at(1, 2) = 6;

	Matrix<int> mt = transpose(m);

	CHECK(mt.rows() == 3);
	CHECK(mt.cols() == 2);
	CHECK(mt.tensor_().dense() ==
	      false); // Strides are swapped, no longer dense in the standard C-order

	// Should view the same data, just swapped strides
	CHECK(mt.at(0, 0) == 1);
	CHECK(mt.at(1, 0) == 2);
	CHECK(mt.at(2, 0) == 3);
	CHECK(mt.at(0, 1) == 4);
	CHECK(mt.at(1, 1) == 5);
	CHECK(mt.at(2, 1) == 6);
}

TEST_CASE("operators_math_det") {
	// Determinant requires floating point to avoid truncation in Gaussian elimination
	Matrix<double> m(3, 3, CPU);

	m.at(0, 0) = 6;
	m.at(0, 1) = 1;
	m.at(0, 2) = 1;
	m.at(1, 0) = 4;
	m.at(1, 1) = -2;
	m.at(1, 2) = 5;
	m.at(2, 0) = 2;
	m.at(2, 1) = 8;
	m.at(2, 2) = 7;

	Scalar<double> d = det(m);

	// Det calculation:
	// 6*(-14 - 40) - 1*(28 - 10) + 1*(32 - (-4))
	// 6*(-54) - 18 + 36
	// -324 - 18 + 36 = -306
	CHECK(d.get() == doctest::Approx(-306.0));
}

TEST_CASE("operators_apply_unary_non_dense") {
	Matrix<int> m(2, 3, CPU);
	for (int i = 0; i < 6; ++i) {
		m.tensor_().rawData()[i] = i + 1;
	}

	Matrix<int> mt = transpose(m);
	CHECK(mt.tensor_().dense() == false);

	auto t_mt = mt.tensor_();
	TZ::internal::TensorIMPL<int>::apply(t_mt, Negate<int>{});

	// mt is a view over m, so m should be negated
	CHECK(m.at(0, 0) == -1);
	CHECK(m.at(0, 1) == -2);
	CHECK(m.at(0, 2) == -3);
	CHECK(m.at(1, 0) == -4);
	CHECK(m.at(1, 1) == -5);
	CHECK(m.at(1, 2) == -6);
}

TEST_CASE("operators_apply_binary_non_dense") {
	Matrix<int> m1(2, 2, CPU);
	m1.at(0, 0) = 1;
	m1.at(0, 1) = 2;
	m1.at(1, 0) = 3;
	m1.at(1, 1) = 4;

	Matrix<int> m2(2, 2, CPU);
	m2.setAll(0);

	Matrix<int> m1t = transpose(m1);
	auto t_m1t = m1t.tensor_();
	auto t_m2 = m2.tensor_();
	TZ::internal::TensorIMPL<int>::apply(t_m1t, t_m2, Copy<int>{});

	CHECK(m2.at(0, 0) == 1);
	CHECK(m2.at(0, 1) == 3);
	CHECK(m2.at(1, 0) == 2);
	CHECK(m2.at(1, 1) == 4);
}

TEST_CASE("operators_apply_ternary_non_dense") {
	Matrix<int> a(2, 2, CPU);
	a.at(0, 0) = 1;
	a.at(0, 1) = 2;
	a.at(1, 0) = 3;
	a.at(1, 1) = 4;

	Matrix<int> b(2, 2, CPU);
	b.setAll(10);

	Matrix<int> c(2, 2, CPU);
	c.setAll(0);

	Matrix<int> at = transpose(a);
	// at is: [1, 3]
	//        [2, 4]

	auto t_at = at.tensor_();
	auto t_b = b.tensor_();
	auto t_c = c.tensor_();
	TZ::internal::TensorIMPL<int>::apply(t_at, t_b, t_c, Add<int>{});

	CHECK(c.at(0, 0) == 11); // 1 + 10
	CHECK(c.at(0, 1) == 13); // 3 + 10
	CHECK(c.at(1, 0) == 12); // 2 + 10
	CHECK(c.at(1, 1) == 14); // 4 + 10
}

TEST_CASE("operators_math_exceptions") {
	Vector<int> v1(3, CPU);
	Vector<int> v2(4, CPU);

	// Mismatched sizes for dot product
	CHECK_THROWS_AS(dot(v1, v2), std::runtime_error);

	Matrix<int> m1(2, 3, CPU);
	Matrix<int> m2(4, 2, CPU);

	// Inner dimensions do not match for matmul
	CHECK_THROWS_AS(matmul(m1, m2), std::runtime_error);

	Matrix<double> m3(2, 3, CPU);

	// Determinant requires a square matrix
	CHECK_THROWS_AS(det(m3), std::runtime_error);
}
