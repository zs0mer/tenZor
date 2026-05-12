#define DOCTEST_CONFIG_COLORS
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "TenZor.hpp"

using namespace tz;
using namespace tz::impl;

// # ============================================================

TEST_CASE("ops_apply1_negate") {
	TensorIMPL<int> t({4}, CPU);
	for (int i = 0; i < 4; i++)
		t.rawData()[i] = i;

	TensorIMPL<int>::apply(t, Negate<int>{});

	CHECK(t.rawData()[0] == 0);
	CHECK(t.rawData()[1] == -1);
	CHECK(t.rawData()[2] == -2);
	CHECK(t.rawData()[3] == -3);
}

TEST_CASE("ops_apply1_set") {
	TensorIMPL<float> t({3, 3}, CPU);
	TensorIMPL<float>::apply(t, Set<float>(7.f));
	for (int i = 0; i < 9; i++)
		CHECK(t.rawData()[i] == doctest::Approx(7.f));
}

TEST_CASE("ops_apply1_empty") {
	TensorIMPL<int> t({0, 5}, CPU);
	CHECK_NOTHROW(TensorIMPL<int>::apply(t, Negate<int>{}));
}

TEST_CASE("ops_apply1_scalar") {
	TensorIMPL<float> s(0, nullptr, CPU);
	s.get() = 5.f;
	TensorIMPL<float>::apply(s, Negate<float>{});
	CHECK(s.get() == doctest::Approx(-5.f));
}

TEST_CASE("ops_apply1_nondense") {
	TensorIMPL<int> t({2, 3}, CPU);
	for (int i = 0; i < 6; i++)
		t.rawData()[i] = i + 1;

	uint64_t sh[] = {3, 2}, st[] = {1, 3};
	TensorIMPL<int> tr(2, sh, st, 0, t.buffer());
	CHECK(!tr.dense());

	TensorIMPL<int>::apply(tr, Negate<int>{});

	for (int i = 0; i < 6; i++)
		CHECK(t.rawData()[i] == -(i + 1));
}

TEST_CASE("ops_apply1_sums1") {
	TensorIMPL<int> t({5}, CPU);
	for (int i = 1; i <= 5; i++)
		t.rawData()[i - 1] = i;

	TensorIMPL<int> acc(0, nullptr, CPU);
	acc.get() = 0;

	TensorIMPL<int>::apply(t, Sum<int>(acc.data()));

	CHECK(acc.get() == 15);
}

TEST_CASE("ops_apply1_sum2") {
	TensorIMPL<int> t({3, 3}, CPU);
	for (int i = 0; i < 9; i++)
		t.rawData()[i] = i + 1;

	TensorIMPL<int> acc(0, nullptr, CPU);
	acc.get() = 0;
	TensorIMPL<int>::apply(t, Sum<int>(acc.data()));
	CHECK(acc.get() == 45);
}

// # ============================================================

TEST_CASE("ops_apply2_copy") {
	TensorIMPL<int> a({4}, CPU);
	TensorIMPL<int> b({4}, CPU);
	for (int i = 0; i < 4; i++) {
		a.rawData()[i] = i * 3;
		b.rawData()[i] = 0;
	}

	TensorIMPL<int>::apply(a, b, Copy<int>{});

	for (int i = 0; i < 4; i++)
		CHECK(b.rawData()[i] == i * 3);
}

TEST_CASE("ops_apply2_add") {
	TensorIMPL<int> a({5}, CPU);
	TensorIMPL<int> b({5}, CPU);
	for (int i = 0; i < 5; i++)
		a.rawData()[i] = i;

	TensorIMPL<int>::apply(a, b, AddScalar<int>(100));

	for (int i = 0; i < 5; i++)
		CHECK(b.rawData()[i] == i + 100);
}

TEST_CASE("ops_apply2_subtract") {
	TensorIMPL<int> a({3}, CPU);
	TensorIMPL<int> b({3}, CPU);
	for (int i = 0; i < 3; i++)
		a.rawData()[i] = i + 10;

	TensorIMPL<int>::apply(a, b, SubtractScalar<int>(5));

	CHECK(b.rawData()[0] == 5);
	CHECK(b.rawData()[1] == 6);
	CHECK(b.rawData()[2] == 7);
}

TEST_CASE("ops_apply2_multiply") {
	TensorIMPL<float> a({4}, CPU);
	TensorIMPL<float> b({4}, CPU);
	for (int i = 0; i < 4; i++)
		a.rawData()[i] = static_cast<float>(i + 1);

	TensorIMPL<float>::apply(a, b, MultiplyScalar<float>(3.f));

	CHECK(b.rawData()[0] == doctest::Approx(3.f));
	CHECK(b.rawData()[1] == doctest::Approx(6.f));
	CHECK(b.rawData()[2] == doctest::Approx(9.f));
	CHECK(b.rawData()[3] == doctest::Approx(12.f));
}

TEST_CASE("ops_apply2_nondense1") {
	TensorIMPL<int> src({2, 3}, CPU);
	for (int i = 0; i < 6; i++)
		src.rawData()[i] = i;

	uint64_t sh[] = {3, 2}, st[] = {1, 3};
	TensorIMPL<int> tsrc(2, sh, st, 0, src.buffer());

	TensorIMPL<int> dst({3, 2}, CPU);
	TensorIMPL<int>::apply(tsrc, dst, Copy<int>{});

	CHECK(dst.at({0, 0}) == 0);
	CHECK(dst.at({0, 1}) == 3);
	CHECK(dst.at({1, 0}) == 1);
	CHECK(dst.at({1, 1}) == 4);
	CHECK(dst.at({2, 0}) == 2);
	CHECK(dst.at({2, 1}) == 5);
}

TEST_CASE("ops_apply2_nondense2") {
	TensorIMPL<int> src({2, 3}, CPU);
	for (int i = 0; i < 6; i++)
		src.rawData()[i] = i + 1;

	mem::Buffer dbuf(6 * sizeof(int));
	int* dd = static_cast<int*>(dbuf->data());
	std::fill(dd, dd + 6, 0);

	uint64_t sh[] = {2, 3}, st[] = {1, 2};
	TensorIMPL<int> dst(2, sh, st, 0, dbuf);

	TensorIMPL<int>::apply(src, dst, Copy<int>{});

	for (uint64_t r = 0; r < 2; r++)
		for (uint64_t c = 0; c < 3; c++)
			CHECK(dst.at({r, c}) == src.at({r, c}));
}

TEST_CASE("ops_apply2_broadcasted") {
	TensorIMPL<int> vec({3}, CPU);
	vec.rawData()[0] = 1;
	vec.rawData()[1] = 2;
	vec.rawData()[2] = 3;

	uint64_t bsh[] = {4, 3}, bst[] = {0, 1};
	TensorIMPL<int> bcast(2, bsh, bst, 0, vec.buffer());
	CHECK(bcast.broadcasted());

	TensorIMPL<int> dst({4, 3}, CPU);
	TensorIMPL<int>::apply(bcast, dst, Copy<int>{});

	for (uint64_t r = 0; r < 4; r++)
		for (uint64_t c = 0; c < 3; c++)
			CHECK(dst.at({r, c}) == (int)(c + 1));
}

TEST_CASE("ops_apply2_error") {
	TensorIMPL<int> a({3}, CPU);
	TensorIMPL<int> b({4}, CPU);
	CHECK_THROWS_AS(TensorIMPL<int>::apply(a, b, Copy<int>{}), std::runtime_error);
}

// # ============================================================

TEST_CASE("ops_apply3_add") {
	TensorIMPL<int> a({2, 2}, CPU);
	TensorIMPL<int> b({2, 2}, CPU);
	TensorIMPL<int> c({2, 2}, CPU);

	for (int i = 0; i < 4; i++) {
		a.rawData()[i] = i;
		b.rawData()[i] = i * 2;
	}

	TensorIMPL<int>::apply(a, b, c, Add<int>{});

	CHECK(c.rawData()[0] == 0);
	CHECK(c.rawData()[1] == 3);
	CHECK(c.rawData()[2] == 6);
	CHECK(c.rawData()[3] == 9);
}

TEST_CASE("ops_apply3_subtract") {
	TensorIMPL<int> a({3}, CPU);
	TensorIMPL<int> b({3}, CPU);
	TensorIMPL<int> c({3}, CPU);

	a.rawData()[0] = 10;
	a.rawData()[1] = 20;
	a.rawData()[2] = 30;
	b.rawData()[0] = 1;
	b.rawData()[1] = 2;
	b.rawData()[2] = 3;

	TensorIMPL<int>::apply(a, b, c, Subtract<int>{});

	CHECK(c.rawData()[0] == 9);
	CHECK(c.rawData()[1] == 18);
	CHECK(c.rawData()[2] == 27);
}

TEST_CASE("ops_apply3_nondense1") {
	TensorIMPL<int> base({2, 2}, CPU);
	base.rawData()[0] = 1;
	base.rawData()[1] = 2;
	base.rawData()[2] = 3;
	base.rawData()[3] = 4;

	uint64_t sh[] = {2, 2}, st[] = {1, 2};
	TensorIMPL<int> a_tr(2, sh, st, 0, base.buffer());

	TensorIMPL<int> b({2, 2}, CPU);

	b.at({0, 0}) = 10;
	b.at({0, 1}) = 10;
	b.at({1, 0}) = 10;
	b.at({1, 1}) = 10;

	TensorIMPL<int> c({2, 2}, CPU);

	TensorIMPL<int>::apply(a_tr, b, c, Add<int>{});

	CHECK(c.at({0, 0}) == 11);
	CHECK(c.at({0, 1}) == 13);
	CHECK(c.at({1, 0}) == 12);
	CHECK(c.at({1, 1}) == 14);
}

TEST_CASE("ops_apply3_errors") {
	TensorIMPL<int> a({3}, CPU);
	TensorIMPL<int> b({3}, CPU);
	TensorIMPL<int> c({4}, CPU);
	CHECK_THROWS_AS(TensorIMPL<int>::apply(a, b, c, Add<int>{}), std::runtime_error);
}

// # ============================================================

TEST_CASE("ops_dot1") {
	Vector<int> a = {1, 2, 3};
	Vector<int> b = {4, 5, 6};
	Scalar<int> r = dot(a, b);

	CHECK(r.get() == 32);
}

TEST_CASE("ops_dot2") {
	Vector<float> a = {1.f, 0.f, -1.f};
	Vector<float> b = {2.f, 3.f, 4.f};
	Scalar<float> r = dot(a, b);

	CHECK(r.get() == doctest::Approx(-2.f));
}

TEST_CASE("ops_dot_zero") {
	Vector<float> a(4, CPU), b(4, CPU);
	a.setAll(0.f);
	b.setAll(0.f);
	Scalar<float> r = dot(a, b);
	CHECK(r.get() == doctest::Approx(0.f));
}

TEST_CASE("ops_dot_nondense") {
	Matrix<int> m = {{1, 2, 3}, {4, 5, 6}};
	Vector<int> col0 = m.col(0);
	Vector<int> col1 = m.col(1);
	CHECK(!col0.tensor_().dense());

	Scalar<int> r = dot(col0, col1);
	CHECK(r.get() == 22);
}

TEST_CASE("ops_dot_error") {
	Vector<int> a(3, CPU), b(4, CPU);
	CHECK_THROWS_AS(dot(a, b), std::runtime_error);
}

// # ============================================================

TEST_CASE("ops_matmul1") {
	Matrix<int> a = {{1, 2, 3}, {4, 5, 6}};
	Matrix<int> b = {{7, 8}, {9, 10}, {11, 12}};

	Matrix<int> c = matmul(a, b);
	CHECK(c.rows() == 2);
	CHECK(c.cols() == 2);

	CHECK(c.at(0, 0) == 58);
	CHECK(c.at(0, 1) == 64);

	CHECK(c.at(1, 0) == 139);
	CHECK(c.at(1, 1) == 154);
}

TEST_CASE("ops_matmul2") {
	Matrix<int> a(1, 1, CPU);
	a.at(0, 0) = 7;
	Matrix<int> b(1, 1, CPU);
	b.at(0, 0) = 6;
	Matrix<int> c = matmul(a, b);
	CHECK(c.rows() == 1);
	CHECK(c.cols() == 1);
	CHECK(c.at(0, 0) == 42);
}

TEST_CASE("ops_matmul3") {
	Matrix<int> a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
	Matrix<int> id(3, 3, CPU);
	id.setAll(0);
	id.at(0, 0) = 1;
	id.at(1, 1) = 1;
	id.at(2, 2) = 1;

	Matrix<int> c = matmul(a, id);
	for (uint64_t i = 0; i < 3; i++)
		for (uint64_t j = 0; j < 3; j++)
			CHECK(c.at(i, j) == a.at(i, j));
}

TEST_CASE("ops_matmul_nondense") {
	Matrix<int> M = {{1, 2}, {3, 4}};
	Matrix<int> N = {{5, 6}, {7, 8}};
	Matrix<int> Mt = M.transpose();
	CHECK(!Mt.tensor_().dense());

	Matrix<int> c = matmul(Mt, N);

	CHECK(c.at(0, 0) == 26);
	CHECK(c.at(0, 1) == 30);
	CHECK(c.at(1, 0) == 38);
	CHECK(c.at(1, 1) == 44);
}

TEST_CASE("ops_matmul_error") {
	Matrix<int> a(2, 3, CPU);
	Matrix<int> b(4, 2, CPU);
	CHECK_THROWS_AS(matmul(a, b), std::runtime_error);
}

// # ============================================================

TEST_CASE("ops_det1") {
	Matrix<double> m(1, 1, CPU);
	m.at(0, 0) = 7.0;
	Scalar<double> d = det(m);
	CHECK(d.get() == doctest::Approx(7.0));
}

TEST_CASE("ops_det2") {
	Matrix<double> m = {{3.0, 8.0}, {4.0, 6.0}};
	Scalar<double> d = det(m);
	CHECK(d.get() == doctest::Approx(-14.0));
}

TEST_CASE("ops_det3") {
	Matrix<double> m = {{6.0, 1.0, 1.0}, {4.0, -2.0, 5.0}, {2.0, 8.0, 7.0}};
	Scalar<double> d = det(m);
	CHECK(d.get() == doctest::Approx(-306.0));
}

TEST_CASE("ops_det4") {
	Matrix<double> m(3, 3, CPU);
	m.setAll(0.0);
	m.at(0, 0) = 1.0;
	m.at(1, 1) = 1.0;
	m.at(2, 2) = 1.0;
	Scalar<double> d = det(m);
	CHECK(d.get() == doctest::Approx(1.0));
}

TEST_CASE("ops_det5") {
	Matrix<double> m = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {1.0, 2.0, 3.0}};
	Scalar<double> d = det(m);
	CHECK(d.get() == doctest::Approx(0.0).epsilon(1e-9));
}

TEST_CASE("ops_det6") {
	Matrix<double> a = {{1.0, 2.0}, {3.0, 4.0}};
	double d_orig = det(a).get();

	Matrix<double> b = {{3.0, 4.0}, {1.0, 2.0}};
	double d_swap = det(b).get();

	CHECK(d_swap == doctest::Approx(-d_orig));
}

TEST_CASE("ops_det_error") {
	Matrix<double> m(2, 3, CPU);
	CHECK_THROWS_AS(det(m), std::runtime_error);
}
