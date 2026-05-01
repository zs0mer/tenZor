#include "math_classes.hpp"
#define DOCTEST_CONFIG_COLORS
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <vector>

#include <Tenzor.hpp>

// ! functions using apply are not tested!

using namespace TZ;
using namespace TZ::impl;

// # ============================================================

TEST_CASE("tensor_fromSTDVec1") {
	std::vector<int> v = {5, 10, 15, 20};
	Tensor<int> t = Tensor<int>::fromSTDVec(v);
	CHECK(t.dim() == 1);
	CHECK(t.shape()[0] == 4);

	Vector<int> vec(t);
	CHECK(vec[0].get() == 5);
	CHECK(vec.at(3) == 20);
}

TEST_CASE("tensor_fromSTDVec2") {
	std::vector<std::vector<int>> v = {{1, 2, 3}, {4, 5, 6}};
	Tensor<int> t = Tensor<int>::fromSTDVec(v);
	CHECK(t.dim() == 2);
	CHECK(t.shape()[0] == 2);
	CHECK(t.shape()[1] == 3);

	Matrix<int> m(t);
	CHECK(m[0][0].get() == 1);
	CHECK(m.at(1, 2) == 6);
}

TEST_CASE("tensor_fromSTDVec3") {
	std::vector<std::vector<std::vector<int>>> v = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
	Tensor<int> t = Tensor<int>::fromSTDVec(v);
	CHECK(t.dim() == 3);
	CHECK(t.shape()[0] == 2);
	CHECK(t.shape()[1] == 2);
	CHECK(t.shape()[2] == 2);
	CHECK(t.tensor_()[0][0][0].get() == 1);
	CHECK(t.tensor_().at({1, 1, 1}) == 8);
}

TEST_CASE("tensor_slice") {
	Tensor<int> t({3, 4}, CPU);
	for (int i = 0; i < 12; i++)
		t.tensor_().rawData()[i] = i;


	Tensor<int> row = t[1];
	CHECK(row.dim() == 1);
	CHECK(row.shape()[0] == 4);
	CHECK(row.tensor_().offset() == 4);
}

TEST_CASE("tensor_wrapper_copyTo") {
	Vector<int> v = {10, 20, 30};
	Vector<int> c = v.copyTo(CPU);
	CHECK(c.at(0) == 10);
	c.at(0) = 999;
	CHECK(v.at(0) == 10);
}

TEST_CASE("tensor_wrapper_negation") {
	Vector<int> v = {1, -2, 3, -4};
	Vector<int> n = -v;
	CHECK(n.at(0) == -1);
	CHECK(n.at(1) == 2);
	CHECK(n.at(2) == -3);
	CHECK(n.at(3) == 4);

	CHECK(v.at(0) == 1);
}

// # ============================================================

TEST_CASE("scalar_constructor") {
	Scalar<int> s(42, CPU);
	CHECK(s.get() == 42);
	CHECK(s.dim() == 0);
	CHECK(s.empty() == false);
}

TEST_CASE("scalar_constructor2") {
	Scalar<float> s(1.5f, CPU);
	CHECK(s.tensor_().scalar());
	CHECK(s.tensor_().size() == 1);
	CHECK(!s.tensor_().indexable());
	CHECK(s.device() == CPU);
}

TEST_CASE("scalar_constructor_error") {
	TensorIMPL<int> t({3}, CPU);
	CHECK_THROWS_AS(Scalar<int>(TensorIMPL<int>(t)), std::runtime_error);
}

TEST_CASE("scalar_assignment1") {
	Scalar<int> s(0, CPU);
	s = 77;
	CHECK(s.get() == 77);
}

TEST_CASE("scalar_assignment2") {
	Scalar<int> a(10, CPU);
	Scalar<int> b(0, CPU);
	b = a;
	CHECK(b.get() == 10);
	a = 99;
	CHECK(b.get() == 10);
}

TEST_CASE("scalar_conversion_to_T") {
	Scalar<int> s(55, CPU);
	int v = s;
	CHECK(v == 55);
}

TEST_CASE("scalar_broadcast1") {
	Scalar<int> s(9, CPU);
	Vector<int> v = s.broadcast(5);

	CHECK(v.dim() == 1);
	CHECK(v.size() == 5);
	CHECK(v.tensor_().strides()[0] == 0);

	for (uint64_t i = 0; i < 5; i++)
		CHECK(v.at(i) == 9);
}

TEST_CASE("scalar_broadcast2") {
	Scalar<int> s(3, CPU);
	Matrix<int> m = s.broadcast(4, 5);

	CHECK(m.rows() == 4);
	CHECK(m.cols() == 5);
	CHECK(m.tensor_().strides()[0] == 0);
	CHECK(m.tensor_().strides()[1] == 0);

	for (uint64_t i = 0; i < 4; i++)
		for (uint64_t j = 0; j < 5; j++)
			CHECK(m.at(i, j) == 3);
}

// # ============================================================

TEST_CASE("vector_constructor1") {
	Vector<float> v(5, CPU);
	CHECK(v.dim() == 1);
	CHECK(v.size() == 5);
	CHECK(v.shape()[0] == 5);
	CHECK(v.tensor_().strides()[0] == 1);
	CHECK(v.tensor_().dense());
}

TEST_CASE("vector_constructor2") {
	Vector<int> v = {10, 20, 30, 40};
	CHECK(v.size() == 4);
	CHECK(v.at(0) == 10);
	CHECK(v.at(1) == 20);
	CHECK(v.at(2) == 30);
	CHECK(v.at(3) == 40);
}

TEST_CASE("vector_constructor3") {
	Matrix<int> m(3, 1, CPU);
	m.at(0, 0) = 10;
	m.at(1, 0) = 20;
	m.at(2, 0) = 30;

	Vector<int> v(m);
	CHECK(v.size() == 3);
	CHECK(v.at(0) == 10);
	CHECK(v.at(1) == 20);
	CHECK(v.at(2) == 30);
}

TEST_CASE("vector_constructor_error") {
	TensorIMPL<int> t({2, 2}, CPU);
	CHECK_THROWS_AS(Vector<int>(TensorIMPL<int>(t)), std::runtime_error);
}

TEST_CASE("vector_at") {
	Vector<double> v(3, CPU);
	v.at(0) = 1.1;
	v.at(1) = 2.2;
	v.at(2) = 3.3;
	CHECK(v.at(0) == doctest::Approx(1.1));
	CHECK(v.at(1) == doctest::Approx(2.2));
	CHECK(v.at(2) == doctest::Approx(3.3));
}

TEST_CASE("vector_at_error") {
	Vector<int> v(3, CPU);
	CHECK_THROWS_AS(v.at(3), std::runtime_error);
}

TEST_CASE("vector_slice") {
	Vector<int> v = {5, 10, 15};
	Scalar<int> s = v[1];
	CHECK(s.get() == 10);

	s.get() = 99;
	CHECK(v.at(1) == 99);
}

TEST_CASE("vector_arithmetic_with_scalar") {
	Vector<int> v = {1, 2, 3};
	Scalar<int> s(10, CPU);

	Vector<int> added = v + s;
	CHECK(added.at(0) == 11);
	CHECK(added.at(2) == 13);

	Vector<int> subbed = v - s;
	CHECK(subbed.at(0) == -9);

	Vector<int> mulled = v * s;
	CHECK(mulled.at(0) == 10);
	CHECK(mulled.at(1) == 20);
	CHECK(mulled.at(2) == 30);

	v += s;
	CHECK(v.at(0) == 11);
	v -= s;
	CHECK(v.at(0) == 1);
	v *= s;
	CHECK(v.at(0) == 10);
}

TEST_CASE("vector_transpose") {
	Vector<int> v = {1, 2, 3, 4};
	Matrix<int> mt = v.transpose();

	CHECK(mt.rows() == 1);
	CHECK(mt.cols() == 4);

	for (int i = 0; i < 4; i++)
		CHECK(mt.at(0, i) == i + 1);

	mt.at(0, 2) = 99;
	CHECK(v.at(2) == 99);
}

TEST_CASE("vector_broadcast") {
	Vector<int> v = {1, 2, 3};
	Matrix<int> m = v.broadcast(4, 3);

	CHECK(m.rows() == 4);
	CHECK(m.cols() == 3);
	CHECK(m.tensor_().strides()[0] == 0);
	CHECK(m.tensor_().strides()[1] == 1);

	for (uint64_t r = 0; r < 4; r++)
		for (uint64_t c = 0; c < 3; c++)
			CHECK(m.at(r, c) == (int)(c + 1));
}

TEST_CASE("vector_clone") {
	Vector<int> v = {1, 2, 3};
	Vector<int> c = v.clone();

	CHECK(c.at(0) == 1);
	c.at(0) = 999;
	CHECK(v.at(0) == 1);
	CHECK(c.tensor_().rawData() != v.tensor_().rawData());
}

// # ============================================================

TEST_CASE("matrix_constructor1") {
	Matrix<int> m(3, 4, CPU);
	CHECK(m.dim() == 2);
	CHECK(m.rows() == 3);
	CHECK(m.cols() == 4);
	CHECK(m.size() == 12);
	CHECK(m.tensor_().strides()[0] == 4);
	CHECK(m.tensor_().strides()[1] == 1);
	CHECK(m.tensor_().dense());
}

TEST_CASE("matrix_constructor2") {
	Matrix<int> m = {{1, 2, 3}, {4, 5, 6}};
	CHECK(m.rows() == 2);
	CHECK(m.cols() == 3);
	CHECK(m.at(0, 0) == 1);
	CHECK(m.at(0, 2) == 3);
	CHECK(m.at(1, 0) == 4);
	CHECK(m.at(1, 2) == 6);
}

TEST_CASE("matrix_constructor3") {
	Vector<int> v = {1, 2, 3};
	Matrix<int> m(v);
	std::cout << m;
	CHECK(m.cols() == 1);
	CHECK(m.rows() == 3);
	CHECK(m.at(0, 0) == 1);
	CHECK(m.at(2, 0) == 3);
}

TEST_CASE("matrix_constructor_error") {
	TensorIMPL<int> t({3}, CPU);
	CHECK_THROWS_AS(Matrix<int>(TensorIMPL<int>(t)), std::runtime_error);
}

TEST_CASE("matrix_at") {
	Matrix<double> m(2, 3, CPU);
	m.at(0, 0) = 1.0;
	m.at(0, 2) = 3.0;
	m.at(1, 1) = 5.5;
	CHECK(m.at(0, 0) == doctest::Approx(1.0));
	CHECK(m.at(0, 2) == doctest::Approx(3.0));
	CHECK(m.at(1, 1) == doctest::Approx(5.5));
}

TEST_CASE("matrix_row") {
	Matrix<int> m = {{1, 2, 3}, {4, 5, 6}};
	Vector<int> r = m.row(0);
	CHECK(r.size() == 3);
	CHECK(r.at(0) == 1);
	CHECK(r.at(2) == 3);

	r.at(1) = 99;
	CHECK(m.at(0, 1) == 99);
}

TEST_CASE("matrix_col") {
	Matrix<int> m = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
	Vector<int> c = m.col(2);
	CHECK(c.size() == 3);
	CHECK(c.at(0) == 3);
	CHECK(c.at(1) == 6);
	CHECK(c.at(2) == 9);
	CHECK(c.tensor_().strides()[0] != 1);

	c.at(0) = 77;
	CHECK(m.at(0, 2) == 77);
}

TEST_CASE("matrix_slice") {
	Matrix<int> m = {{10, 20}, {30, 40}};
	Vector<int> row0 = m[0];
	CHECK(row0.at(0) == 10);
	CHECK(row0.at(1) == 20);

	row0.at(0) = 55;
	CHECK(m.at(0, 0) == 55);
}

TEST_CASE("matrix_swapRow") {
	Matrix<int> m = {{1, 2}, {3, 4}, {5, 6}};
	m.swapRow(0, 2);
	CHECK(m.at(0, 0) == 5);
	CHECK(m.at(0, 1) == 6);
	CHECK(m.at(2, 0) == 1);
	CHECK(m.at(2, 1) == 2);
	CHECK(m.at(1, 0) == 3);
}

TEST_CASE("matrix_swapCol") {
	Matrix<int> m = {{1, 2, 3}, {4, 5, 6}};
	m.swapCol(0, 2);
	CHECK(m.at(0, 0) == 3);
	CHECK(m.at(0, 2) == 1);
	CHECK(m.at(1, 0) == 6);
	CHECK(m.at(1, 2) == 4);
	CHECK(m.at(0, 1) == 2);
}

TEST_CASE("matrix_swap_same_index") {
	Matrix<int> m = {{1, 2}, {3, 4}};
	m.swapRow(1, 1);
	CHECK(m.at(1, 0) == 3);
	m.swapCol(0, 0);
	CHECK(m.at(0, 0) == 1);
}

TEST_CASE("matrix_transpose1") {
	Matrix<int> m = {{1, 2, 3}, {4, 5, 6}};
	Matrix<int> mt = m.transpose();

	CHECK(mt.rows() == 3);
	CHECK(mt.cols() == 2);
	CHECK(!mt.tensor_().dense());

	CHECK(mt.at(0, 0) == 1);
	CHECK(mt.at(0, 1) == 4);
	CHECK(mt.at(1, 0) == 2);
	CHECK(mt.at(1, 1) == 5);
	CHECK(mt.at(2, 0) == 3);
	CHECK(mt.at(2, 1) == 6);

	mt.at(0, 1) = 99;
	CHECK(m.at(1, 0) == 99);
}

TEST_CASE("matrix_transpose2") {
	Matrix<int> m = {{1, 2, 3}, {4, 5, 6}};
	Matrix<int> mtt = m.transpose().transpose();

	CHECK(mtt.rows() == m.rows());
	CHECK(mtt.cols() == m.cols());
	for (uint64_t i = 0; i < 2; i++)
		for (uint64_t j = 0; j < 3; j++)
			CHECK(mtt.at(i, j) == m.at(i, j));
}

TEST_CASE("matrix_clone") {
	Matrix<int> m = {{1, 2}, {3, 4}};
	Matrix<int> c = m.clone();

	CHECK(c.at(0, 0) == 1);
	c.at(0, 0) = 999;
	CHECK(m.at(0, 0) == 1);
	CHECK(c.tensor_().rawData() != m.tensor_().rawData());
}

TEST_CASE("matrix_errors") {
	Matrix<int> m(2, 2, CPU);
	CHECK_THROWS_AS(m.at(2, 0), std::runtime_error);
	CHECK_THROWS_AS(m.at(0, 2), std::runtime_error);
	CHECK_THROWS_AS(m.row(2), std::runtime_error);
	CHECK_THROWS_AS(m.col(2), std::runtime_error);
	CHECK_THROWS_AS(m.swapRow(0, 2), std::runtime_error);
	CHECK_THROWS_AS(m.swapCol(2, 0), std::runtime_error);
}
