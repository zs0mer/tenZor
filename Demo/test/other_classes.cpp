#include <Tenzor.hpp>
#include "math_classes.hpp"
#include "tensor.hpp"
#define DOCTEST_CONFIG_COLORS
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <vector>

using namespace TZ;
using namespace TZ::impl;

//& ============================================================
//& Scalar
//& ============================================================

TEST_CASE("scalar_construction_and_get") {
	Scalar<int> s(42, CPU);
	CHECK(s.get() == 42);
	CHECK(s.dim() == 0);
	CHECK(s.empty() == false);
}

TEST_CASE("scalar_tensor_metadata") {
	Scalar<float> s(1.5f, CPU);
	CHECK(s.tensor_().scalar());
	CHECK(s.tensor_().size() == 1);
	CHECK(!s.tensor_().indexable());
	CHECK(s.device() == CPU);
}

TEST_CASE("scalar_set") {
	Scalar<double> s(0.0, CPU);
	s.set(3.14159);
	CHECK(s.get() == doctest::Approx(3.14159));
}

TEST_CASE("scalar_assignment_from_T") {
	Scalar<int> s(0, CPU);
	s = 77;
	CHECK(s.get() == 77);
}

TEST_CASE("scalar_assignment_from_scalar") {
	Scalar<int> a(10, CPU);
	Scalar<int> b(0, CPU);
	b = a;
	CHECK(b.get() == 10);
	a = 99;
	CHECK(b.get() == 10); // b is independent
}

TEST_CASE("scalar_implicit_conversion_to_T") {
	Scalar<int> s(55, CPU);
	int v = s;
	CHECK(v == 55);
}

TEST_CASE("scalar_set_all_and_sum") {
	Scalar<int> s(0, CPU);
	s.setAll(7);
	CHECK(s.get() == 7);
	Scalar<int> result = s.sum();
	CHECK(result.get() == 7); // sum of a scalar is itself
}

TEST_CASE("scalar_arithmetic_with_scalar") {
	Scalar<int> a(10, CPU), b(3, CPU);

	Scalar<int> add = a + b;
	CHECK(add.get() == 13);
	Scalar<int> sub = a - b;
	CHECK(sub.get() == 7);
	Scalar<int> neg = -a;
	CHECK(neg.get() == -10);

	a += b;
	CHECK(a.get() == 13);
	a -= b;
	CHECK(a.get() == 10);
	a *= b;
	CHECK(a.get() == 30);
}

TEST_CASE("scalar_arithmetic_with_T_scalar") {
	Scalar<float> v(4.f, CPU);
	Scalar<float> k(2.f, CPU);

	Scalar<float> result = v + k;
	CHECK(result.get() == doctest::Approx(6.f));

	result = v * Scalar<float>(3.f);
	CHECK(result.get() == doctest::Approx(12.f));
}

TEST_CASE("scalar_broadcast_to_vector") {
	Scalar<int> s(9, CPU);
	Vector<int> v = s.broadcast(5);

	CHECK(v.dim() == 1);
	CHECK(v.size() == 5);
	CHECK(v.tensor_().strides()[0] == 0); // zero stride = broadcast

	// Every element must read back as the scalar's value
	for (uint64_t i = 0; i < 5; i++)
		CHECK(v.at(i) == 9);
}

TEST_CASE("scalar_broadcast_to_matrix") {
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

TEST_CASE("scalar_clone_independence") {
	Scalar<int> s(10, CPU);
	Scalar<int> c = s.clone();
	CHECK(c.get() == 10);
	c = 99;
	CHECK(s.get() == 10); // original unchanged
}

TEST_CASE("scalar_copy_to_cpu_is_clone") {
	Scalar<double> s(2.71828, CPU);
	Scalar<double> copy = s.copyTo(CPU);
	CHECK(copy.get() == doctest::Approx(2.71828));
	copy = 0.0;
	CHECK(s.get() == doctest::Approx(2.71828));
}

//& ============================================================
//& Vector
//& ============================================================

TEST_CASE("vector_construction_by_size") {
	Vector<float> v(5, CPU);
	CHECK(v.dim() == 1);
	CHECK(v.size() == 5);
	CHECK(v.shape()[0] == 5);
	CHECK(v.tensor_().strides()[0] == 1);
	CHECK(v.tensor_().dense());
}

TEST_CASE("vector_initializer_list") {
	Vector<int> v = {10, 20, 30, 40};
	CHECK(v.size() == 4);
	CHECK(v.at(0) == 10);
	CHECK(v.at(1) == 20);
	CHECK(v.at(2) == 30);
	CHECK(v.at(3) == 40);
}

TEST_CASE("vector_at_read_write") {
	Vector<double> v(3, CPU);
	v.at(0) = 1.1;
	v.at(1) = 2.2;
	v.at(2) = 3.3;
	CHECK(v.at(0) == doctest::Approx(1.1));
	CHECK(v.at(1) == doctest::Approx(2.2));
	CHECK(v.at(2) == doctest::Approx(3.3));
}

TEST_CASE("vector_subscript_returns_scalar_view") {
	Vector<int> v = {5, 10, 15};
	Scalar<int> s = v[1];
	CHECK(s.get() == 10);

	// It is a view — mutating through the scalar mutates the vector
	s.get() = 99;
	CHECK(v.at(1) == 99);
}

TEST_CASE("vector_set_all_and_sum") {
	Vector<int> v(4, CPU);
	v.setAll(5);
	for (int i = 0; i < 4; i++)
		CHECK(v.at(i) == 5);

	Scalar<int> s = v.sum();
	CHECK(s.get() == 20);
}

TEST_CASE("vector_arithmetic_with_vector") {
	Vector<int> a = {1, 2, 3};
	Vector<int> b = {10, 20, 30};

	Vector<int> sum = a + b;
	CHECK(sum.at(0) == 11);
	CHECK(sum.at(1) == 22);
	CHECK(sum.at(2) == 33);

	Vector<int> diff = b - a;
	CHECK(diff.at(0) == 9);
	CHECK(diff.at(2) == 27);

	Vector<int> neg = -a;
	CHECK(neg.at(0) == -1);
	CHECK(neg.at(2) == -3);

	a += b;
	CHECK(a.at(0) == 11);

	a -= b;
	CHECK(a.at(0) == 1);
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

TEST_CASE("vector_transpose_to_row_matrix") {
	Vector<int> v = {1, 2, 3, 4};
	Matrix<int> mt = v.transpose();

	CHECK(mt.rows() == 1);
	CHECK(mt.cols() == 4);

	// The strides let it navigate the same underlying data
	for (int i = 0; i < 4; i++)
		CHECK(mt.at(0, i) == i + 1);

	// It is a view — modification propagates back
	mt.at(0, 2) = 99;
	CHECK(v.at(2) == 99);
}

TEST_CASE("vector_broadcast_to_matrix") {
	// v = [1,2,3]; broadcast to 4x3 -> each row is [1,2,3]
	Vector<int> v = {1, 2, 3};
	Matrix<int> m = v.broadcast(4, 3);

	CHECK(m.rows() == 4);
	CHECK(m.cols() == 3);
	CHECK(m.tensor_().strides()[0] == 0); // row stride is broadcast
	CHECK(m.tensor_().strides()[1] == 1);

	for (uint64_t r = 0; r < 4; r++)
		for (uint64_t c = 0; c < 3; c++)
			CHECK(m.at(r, c) == (int)(c + 1));
}

TEST_CASE("vector_clone_is_independent") {
	Vector<int> v = {1, 2, 3};
	Vector<int> c = v.clone();

	CHECK(c.at(0) == 1);
	c.at(0) = 999;
	CHECK(v.at(0) == 1); // original unchanged
	CHECK(c.tensor_().rawData() != v.tensor_().rawData());
}

TEST_CASE("vector_from_column_matrix") {
	// A column vector is a Mx1 matrix; Vector can be constructed from it
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

TEST_CASE("vector_at_oob_throws") {
	Vector<int> v(3, CPU);
	CHECK_THROWS_AS(v.at(3), std::runtime_error);
}

TEST_CASE("vector_wrong_dim_tensor_throws") {
	TensorIMPL<int> t({2, 2}, CPU);
	CHECK_THROWS_AS(Vector<int>(TensorIMPL<int>(t)), std::runtime_error);
}

//& ============================================================
//& Matrix
//& ============================================================

TEST_CASE("matrix_construction_rows_cols") {
	Matrix<int> m(3, 4, CPU);
	CHECK(m.dim() == 2);
	CHECK(m.rows() == 3);
	CHECK(m.cols() == 4);
	CHECK(m.size() == 12);
	CHECK(m.tensor_().strides()[0] == 4);
	CHECK(m.tensor_().strides()[1] == 1);
	CHECK(m.tensor_().dense());
}

TEST_CASE("matrix_initializer_list") {
	Matrix<int> m = {{1, 2, 3}, {4, 5, 6}};
	CHECK(m.rows() == 2);
	CHECK(m.cols() == 3);
	CHECK(m.at(0, 0) == 1);
	CHECK(m.at(0, 2) == 3);
	CHECK(m.at(1, 0) == 4);
	CHECK(m.at(1, 2) == 6);
}

TEST_CASE("matrix_at_read_write") {
	Matrix<double> m(2, 3, CPU);
	m.at(0, 0) = 1.0;
	m.at(0, 2) = 3.0;
	m.at(1, 1) = 5.5;
	CHECK(m.at(0, 0) == doctest::Approx(1.0));
	CHECK(m.at(0, 2) == doctest::Approx(3.0));
	CHECK(m.at(1, 1) == doctest::Approx(5.5));
}

TEST_CASE("matrix_row_is_a_view") {
	Matrix<int> m = {{1, 2, 3}, {4, 5, 6}};
	Vector<int> r = m.row(0);
	CHECK(r.size() == 3);
	CHECK(r.at(0) == 1);
	CHECK(r.at(2) == 3);

	// Modify through row view — must propagate to matrix
	r.at(1) = 99;
	CHECK(m.at(0, 1) == 99);
}

TEST_CASE("matrix_col_is_a_view") {
	Matrix<int> m = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
	Vector<int> c = m.col(2);
	CHECK(c.size() == 3);
	CHECK(c.at(0) == 3);
	CHECK(c.at(1) == 6);
	CHECK(c.at(2) == 9);
	CHECK(c.tensor_().strides()[0] != 1); // non-unit stride

	// Modify through column view — must propagate
	c.at(0) = 77;
	CHECK(m.at(0, 2) == 77);
}

TEST_CASE("matrix_operator_subscript_is_row_view") {
	Matrix<int> m = {{10, 20}, {30, 40}};
	Vector<int> row0 = m[0];
	CHECK(row0.at(0) == 10);
	CHECK(row0.at(1) == 20);

	// It is a view
	row0.at(0) = 55;
	CHECK(m.at(0, 0) == 55);
}

TEST_CASE("matrix_swap_row") {
	Matrix<int> m = {{1, 2}, {3, 4}, {5, 6}};
	m.swapRow(0, 2);
	CHECK(m.at(0, 0) == 5);
	CHECK(m.at(0, 1) == 6);
	CHECK(m.at(2, 0) == 1);
	CHECK(m.at(2, 1) == 2);
	CHECK(m.at(1, 0) == 3); // unchanged
}

TEST_CASE("matrix_swap_col") {
	Matrix<int> m = {{1, 2, 3}, {4, 5, 6}};
	m.swapCol(0, 2);
	CHECK(m.at(0, 0) == 3);
	CHECK(m.at(0, 2) == 1);
	CHECK(m.at(1, 0) == 6);
	CHECK(m.at(1, 2) == 4);
	CHECK(m.at(0, 1) == 2); // unchanged
}

TEST_CASE("matrix_swap_same_index_noop") {
	Matrix<int> m = {{1, 2}, {3, 4}};
	m.swapRow(1, 1); // no-op
	CHECK(m.at(1, 0) == 3);
	m.swapCol(0, 0); // no-op
	CHECK(m.at(0, 0) == 1);
}

TEST_CASE("matrix_transpose_is_view_not_dense") {
	Matrix<int> m = {{1, 2, 3}, {4, 5, 6}};
	Matrix<int> mt = m.transpose();

	CHECK(mt.rows() == 3);
	CHECK(mt.cols() == 2);
	CHECK(!mt.tensor_().dense());

	// Check all logical values
	CHECK(mt.at(0, 0) == 1);
	CHECK(mt.at(0, 1) == 4);
	CHECK(mt.at(1, 0) == 2);
	CHECK(mt.at(1, 1) == 5);
	CHECK(mt.at(2, 0) == 3);
	CHECK(mt.at(2, 1) == 6);

	// It is a view — modification propagates
	mt.at(0, 1) = 99;
	CHECK(m.at(1, 0) == 99);
}

TEST_CASE("matrix_double_transpose_is_original") {
	Matrix<int> m = {{1, 2, 3}, {4, 5, 6}};
	Matrix<int> mtt = m.transpose().transpose();

	CHECK(mtt.rows() == m.rows());
	CHECK(mtt.cols() == m.cols());
	for (uint64_t i = 0; i < 2; i++)
		for (uint64_t j = 0; j < 3; j++)
			CHECK(mtt.at(i, j) == m.at(i, j));
}

TEST_CASE("matrix_set_all_and_sum") {
	Matrix<int> m(3, 3, CPU);
	m.setAll(4);
	for (uint64_t i = 0; i < 3; i++)
		for (uint64_t j = 0; j < 3; j++)
			CHECK(m.at(i, j) == 4);

	Scalar<int> s = m.sum();
	CHECK(s.get() == 36); // 9 elements * 4
}

TEST_CASE("matrix_arithmetic") {
	Matrix<int> a = {{1, 2}, {3, 4}};
	Matrix<int> b = {{5, 6}, {7, 8}};

	Matrix<int> sum = a + b;
	CHECK(sum.at(0, 0) == 6);
	CHECK(sum.at(0, 1) == 8);
	CHECK(sum.at(1, 0) == 10);
	CHECK(sum.at(1, 1) == 12);

	Matrix<int> diff = b - a;
	CHECK(diff.at(0, 0) == 4);
	CHECK(diff.at(1, 1) == 4);

	Matrix<int> neg = -a;
	CHECK(neg.at(0, 0) == -1);
	CHECK(neg.at(1, 1) == -4);

	Matrix<int> scaled = a * Scalar<int>(3);
	CHECK(scaled.at(0, 0) == 3);
	CHECK(scaled.at(1, 1) == 12);

	a += b;
	CHECK(a.at(0, 0) == 6);
	a -= b;
	CHECK(a.at(0, 0) == 1);
	a *= Scalar<int>(2);
	CHECK(a.at(0, 0) == 2);
}

TEST_CASE("matrix_clone_is_independent") {
	Matrix<int> m = {{1, 2}, {3, 4}};
	Matrix<int> c = m.clone();

	CHECK(c.at(0, 0) == 1);
	c.at(0, 0) = 999;
	CHECK(m.at(0, 0) == 1); // original unchanged
	CHECK(c.tensor_().rawData() != m.tensor_().rawData());
}

TEST_CASE("matrix_from_vector_is_row_matrix") {
	Vector<int> v = {1, 2, 3};
	Matrix<int> m(v);
	std::cout << m;
	CHECK(m.cols() == 1);
	CHECK(m.rows() == 3);
	CHECK(m.at(0, 0) == 1);
	CHECK(m.at(2, 0) == 3);
}

TEST_CASE("matrix_oob_exceptions") {
	Matrix<int> m(2, 2, CPU);
	CHECK_THROWS_AS(m.at(2, 0), std::runtime_error);
	CHECK_THROWS_AS(m.at(0, 2), std::runtime_error);
	CHECK_THROWS_AS(m.row(2), std::runtime_error);
	CHECK_THROWS_AS(m.col(2), std::runtime_error);
	CHECK_THROWS_AS(m.swapRow(0, 2), std::runtime_error);
	CHECK_THROWS_AS(m.swapCol(2, 0), std::runtime_error);
}

TEST_CASE("matrix_wrong_dim_tensor_throws") {
	TensorIMPL<int> t({3}, CPU);
	CHECK_THROWS_AS(Matrix<int>(TensorIMPL<int>(t)), std::runtime_error);
}

//& ============================================================
//& Tensor<T> and TensorWrapper
//& ============================================================

TEST_CASE("tensor_class_from_std_vec_1d") {
	std::vector<int> v = {5, 10, 15, 20};
	Tensor<int> t = Tensor<int>::fromSTDVec(v);
	CHECK(t.dim() == 1);
	CHECK(t.shape()[0] == 4);
	// Access via operator[] which returns a Tensor
	CHECK(t[0].tensor_().get() == 5);
	CHECK(t[3].tensor_().get() == 20);
}

TEST_CASE("tensor_class_from_std_vec_2d") {
	std::vector<std::vector<int>> v = {{1, 2, 3}, {4, 5, 6}};
	Tensor<int> t = Tensor<int>::fromSTDVec(v);
	CHECK(t.dim() == 2);
	CHECK(t.shape()[0] == 2);
	CHECK(t.shape()[1] == 3);

	Matrix<int> m(t.tensor_());
	CHECK(m.at(0, 0) == 1);
	CHECK(m.at(1, 2) == 6);
}

TEST_CASE("tensor_class_from_std_vec_3d") {
	std::vector<std::vector<std::vector<int>>> v = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
	Tensor<int> t = Tensor<int>::fromSTDVec(v);
	CHECK(t.dim() == 3);
	CHECK(t.shape()[0] == 2);
	CHECK(t.shape()[1] == 2);
	CHECK(t.shape()[2] == 2);
	CHECK(t.tensor_().at({0, 0, 0}) == 1);
	CHECK(t.tensor_().at({1, 1, 1}) == 8);
}

TEST_CASE("tensor_class_operator_subscript_returns_tensor") {
	Tensor<int> t({3, 4}, CPU);
	for (int i = 0; i < 12; i++)
		t.tensor_().rawData()[i] = i;

	// operator[] on Tensor<T> returns a Tensor<T>
	Tensor<int> row = t[1];
	CHECK(row.dim() == 1);
	CHECK(row.shape()[0] == 4);
	CHECK(row.tensor_().offset() == 4);
}

TEST_CASE("tensor_wrapper_to_tensor") {
	Vector<int> v = {1, 2, 3};
	Tensor<int> t = v.toTensor();
	CHECK(t.dim() == 1);
	CHECK(t.shape()[0] == 3);
	CHECK(t.tensor_().rawData() == v.tensor_().rawData());
}

TEST_CASE("tensor_wrapper_copy_to_cpu_is_independent") {
	Vector<int> v = {10, 20, 30};
	Vector<int> c = v.copyTo(CPU);
	CHECK(c.at(0) == 10);
	c.at(0) = 999;
	CHECK(v.at(0) == 10);
}

TEST_CASE("tensor_wrapper_negate_operator") {
	Vector<int> v = {1, -2, 3, -4};
	Vector<int> n = -v;
	CHECK(n.at(0) == -1);
	CHECK(n.at(1) == 2);
	CHECK(n.at(2) == -3);
	CHECK(n.at(3) == 4);
	// Original unchanged
	CHECK(v.at(0) == 1);
}

TEST_CASE("scalar_wrong_dim_tensor_throws") {
	TensorIMPL<int> t({3}, CPU);
	CHECK_THROWS_AS(Scalar<int>(TensorIMPL<int>(t)), std::runtime_error);
}
