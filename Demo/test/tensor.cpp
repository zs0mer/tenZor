#include "tensor.hpp"
#include <Tenzor.hpp>
#include <cstdint>
#include "math_classes.hpp"
#include "tensor_impl.hpp"
#define DOCTEST_CONFIG_COLORS
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <sstream>
#include <vector>

using namespace TZ;
using namespace TZ::impl;

// # ============================================================

TEST_CASE("tensorIMPL_constructor_default") {
	TensorIMPL<float> t;
	CHECK(t.device() == CPU);
	CHECK(t.dim() == 0);
	CHECK(t.size() == 0);
	CHECK(t.empty());
	CHECK(!t.scalar());
	CHECK(t.dense());
	CHECK(!t.broadcasted());
	CHECK(t.data() == nullptr);
	CHECK(t.rawData() == nullptr);
}

// # ------------------

TEST_CASE("tensorIMPL_data_constructor1") {
	std::vector<uint64_t> shape = {7};
	TensorIMPL<float> t(1, shape.data(), CPU);
	CHECK(t.dim() == 1);
	CHECK(t.size() == 7);
	CHECK(t.strides()[0] == 1);
	CHECK(t.dense());
	CHECK(t.shape()[0] == 7);
}

TEST_CASE("tensorIMPL_data_constructor2") {
	std::vector<uint64_t> shape = {5, 3};
	TensorIMPL<int> t(2, shape.data(), CPU);
	CHECK(t.strides()[0] == 3);
	CHECK(t.strides()[1] == 1);
	CHECK(t.size() == 15);
	CHECK(t.dense());
}

TEST_CASE("tensorIMPL_data_constructor3") {
	std::vector<uint64_t> shape = {2, 3, 4, 5};
	TensorIMPL<float> t(4, shape.data(), CPU);
	CHECK(t.strides()[0] == 60);
	CHECK(t.strides()[1] == 20);
	CHECK(t.strides()[2] == 5);
	CHECK(t.strides()[3] == 1);
	CHECK(t.size() == 120);
}

// # ------------------

TEST_CASE("tensorIMPL_normal_constructor1") {
	TensorIMPL<float> t({7}, CPU);
	CHECK(t.dim() == 1);
	CHECK(t.size() == 7);
	CHECK(t.strides()[0] == 1);
	CHECK(t.dense());
	CHECK(t.shape()[0] == 7);
}

TEST_CASE("tensorIMPL_normal_constructor2") {
	TensorIMPL<int> t({5, 3}, CPU);
	CHECK(t.strides()[0] == 3);
	CHECK(t.strides()[1] == 1);
	CHECK(t.size() == 15);
	CHECK(t.dense());
}

TEST_CASE("tensorIMPL_normal_constructor3") {
	TensorIMPL<float> t({2, 3, 4, 5}, CPU);
	CHECK(t.strides()[0] == 60);
	CHECK(t.strides()[1] == 20);
	CHECK(t.strides()[2] == 5);
	CHECK(t.strides()[3] == 1);
	CHECK(t.size() == 120);
}

// # ------------------

TEST_CASE("tensorIMPL_low_constructor1") {
	uint64_t shape[] = {2, 3};
	uint64_t strides[] = {1, 2};
	mem::Buffer buf(6 * sizeof(int));
	int* d = static_cast<int*>(buf->data());
	for (int i = 0; i < 6; i++)
		d[i] = i;

	TensorIMPL<int> t(2, shape, strides, 0, buf);
	CHECK(!t.dense());
	CHECK(!t.broadcasted());
	CHECK(t.size() == 6);
	CHECK(t.at({0, 0}) == 0);
	CHECK(t.at({1, 0}) == 1);
	CHECK(t.at({0, 1}) == 2);
	CHECK(t.at({1, 1}) == 3);
	CHECK(t.at({0, 2}) == 4);
	CHECK(t.at({1, 2}) == 5);
}

TEST_CASE("tensorIMPL_low_constructor2") {
	TensorIMPL<int> parent({4, 4}, CPU);
	for (int i = 0; i < 16; i++)
		parent.rawData()[i] = i;

	uint64_t shape[] = {4};
	uint64_t strides[] = {1};

	TensorIMPL<int> view(1, shape, strides, 8, parent.buffer());

	CHECK(view.offset() == 8);
	CHECK(view.data() == parent.rawData() + 8);
	CHECK(view.at({0}) == 8);
	CHECK(view.at({3}) == 11);
}

TEST_CASE("tensorIMPL_low_constructor3") {
	std::vector<uint64_t> shape = {2, 3, 4, 5};
	TensorIMPL<float> f(4, shape.data(), CPU);

	TensorIMPL<float> t(f.dim(), f.shape(), f.strides(), f.offset(), f.buffer());
	CHECK(t.strides()[0] == 60);
	CHECK(t.strides()[1] == 20);
	CHECK(t.strides()[2] == 5);
	CHECK(t.strides()[3] == 1);
	CHECK(t.size() == 120);
}

// # ============================================================

TEST_CASE("tensorIMPL_data_qeries1") {
	TensorIMPL<int> t({4, 4}, CPU);
	CHECK(t.data() == t.rawData());
	CHECK(t.offset() == 0);
}

TEST_CASE("tensorIMPL_data_qeries2") {
	TensorIMPL<int> t({3, 4}, CPU);
	auto row1 = t[1];
	CHECK(row1.offset() == 4);
	CHECK(row1.rawData() == t.rawData());
	CHECK(row1.data() == t.rawData() + 4);
}

// # ============================================================

TEST_CASE("tensorIMPL_at1") {
	TensorIMPL<int> t({3, 3}, CPU);
	for (int i = 0; i < 9; i++)
		t.rawData()[i] = i;

	CHECK(t.at({0, 0}) == 0);
	CHECK(t.at({0, 1}) == 1);
	CHECK(t.at({0, 2}) == 2);
	CHECK(t.at({1, 0}) == 3);
	CHECK(t.at({2, 2}) == 8);
}

TEST_CASE("tensorIMPL_at2") {
	TensorIMPL<int> t({2, 4}, CPU);
	for (int i = 0; i < 8; i++)
		t.rawData()[i] = i * 10;

	uint64_t idx0[] = {0, 3};
	uint64_t idx1[] = {1, 2};
	CHECK(t.at(idx0) == 30);
	CHECK(t.at(idx1) == 60);
}

TEST_CASE("tensorIMPL_at3") {
	TensorIMPL<int> t({2, 3, 4}, CPU);
	for (int i = 0; i < 24; i++)
		t.rawData()[i] = i;

	CHECK(t.at({0, 0, 0}) == 0);
	CHECK(t.at({1, 0, 0}) == 12);
	CHECK(t.at({1, 2, 3}) == 23);
	CHECK(t.at({0, 2, 1}) == 9);
}

TEST_CASE("tensorIMPL_at_error1") {
	TensorIMPL<int> t({2, 3}, CPU);
	CHECK_THROWS_AS(t.at({0}), std::runtime_error);
	CHECK_THROWS_AS(t.at({0, 0, 0}), std::runtime_error);
}

TEST_CASE("tensorIMPL_at_error2") {
	TensorIMPL<int> t({2, 2}, CPU);
	CHECK_THROWS_AS(t.at({2, 0}), std::runtime_error);
	CHECK_THROWS_AS(t.at({0, 2}), std::runtime_error);

	uint64_t bad[] = {5, 0};
	CHECK_THROWS_AS(t.at(bad), std::runtime_error);
}

// # ============================================================

TEST_CASE("tensorIMPL_slice1") {
	TensorIMPL<float> t({3, 4}, CPU);
	for (int i = 0; i < 12; i++)
		t.rawData()[i] = static_cast<float>(i);

	auto row0 = t[0];
	CHECK(row0.dim() == 1);
	CHECK(row0.size() == 4);
	CHECK(row0.shape()[0] == 4);
	CHECK(row0.strides()[0] == 1);
	CHECK(row0.offset() == 0);

	auto row2 = t[2];
	CHECK(row2.offset() == 8);
	CHECK(row2[0].get() == doctest::Approx(8.f));
	CHECK(row2[3].get() == doctest::Approx(11.f));
}

TEST_CASE("tensorIMPL_slice2") {
	TensorIMPL<int> t({2, 3, 4}, CPU);
	for (int i = 0; i < 24; i++)
		t.rawData()[i] = i;

	auto s = t[1][2][3];
	CHECK(s.dim() == 0);
	CHECK(s.scalar());
	CHECK(s.get() == 23);
	CHECK(s.offset() == 23);
}

TEST_CASE("tensorIMPL_slice_error") {
	TensorIMPL<int> t({3, 3}, CPU);
	CHECK_THROWS_AS(t[3], std::runtime_error);
	CHECK_THROWS_AS(t[0][3], std::runtime_error);
}

// # ============================================================

TEST_CASE("tensorIMPL_scalar_construction1") {
	TensorIMPL<double> t(0, nullptr, CPU);
	CHECK(t.dim() == 0);
	CHECK(t.size() == 1);
	CHECK(t.scalar());
	CHECK(!t.empty());
	CHECK(!t.indexable());
	CHECK(t.dense());

	t.get() = 3.14159;
	CHECK(t.get() == doctest::Approx(3.14159));
}

TEST_CASE("tensorIMPL_scalar_error") {
	TensorIMPL<int> t({3}, CPU);
	CHECK_THROWS_AS(t.get(), std::runtime_error);

	TensorIMPL<int> t2({2, 2}, CPU);
	CHECK_THROWS_AS(t2.get(), std::runtime_error);
}

TEST_CASE("tensorIMPL_scalar_slice") {
	TensorIMPL<int> t({4, 4}, CPU);
	for (int i = 0; i < 16; i++)
		t.rawData()[i] = i;
	auto s = t[3][3];
	CHECK(s.scalar());
	CHECK(s.get() == 15);
}

// # ============================================================

TEST_CASE("tensorIMPL_empty") {
	TensorIMPL<float> t({0, 5, 5}, CPU);
	CHECK(t.empty());
	CHECK(t.size() == 0);
	CHECK(!t.indexable());
	CHECK(!t.scalar());
}

// # ============================================================

TEST_CASE("tensorIMPL_dense1") {
	TensorIMPL<int> t({4, 3}, CPU);
	CHECK(t.dense());
	CHECK(!t.broadcasted());
}

TEST_CASE("tensorIMPL_dense2") {
	uint64_t shape[] = {3, 4};
	uint64_t strides[] = {1, 3};
	mem::Buffer buf(3 * 4 * sizeof(int));
	TensorIMPL<int> t(2, shape, strides, 0, buf);

	CHECK(!t.dense());
	CHECK(!t.broadcasted());
}

// # ============================================================

TEST_CASE("tensorIMPL_broadcasted1") {
	TensorIMPL<float> src({4}, CPU);
	for (int i = 0; i < 4; i++)
		src.rawData()[i] = static_cast<float>(i);

	uint64_t shape[] = {3, 4};
	uint64_t strides[] = {0, 1};
	TensorIMPL<float> b(2, shape, strides, 0, src.buffer());

	CHECK(b.broadcasted());
	CHECK(!b.dense());
	CHECK(b.size() == 12);


	for (uint64_t r = 0; r < 3; r++)
		for (uint64_t c = 0; c < 4; c++)
			CHECK(b.at({r, c}) == doctest::Approx(static_cast<float>(c)));
}

TEST_CASE("tensorIMPL_broadcasted2") {
	TensorIMPL<int> s(0, nullptr, CPU);
	s.get() = 7;

	auto b = s.broadcast({3, 4});
	CHECK(b.dim() == 2);
	CHECK(b.shape()[0] == 3);
	CHECK(b.shape()[1] == 4);
	CHECK(b.strides()[0] == 0);
	CHECK(b.strides()[1] == 0);
	CHECK(b.broadcasted());

	for (uint64_t i = 0; i < 3; i++)
		for (uint64_t j = 0; j < 4; j++)
			CHECK(b.at({i, j}) == 7);
}

TEST_CASE("tensorIMPL_broadcast3") {
	TensorIMPL<int> v({4}, CPU);
	for (int i = 0; i < 4; i++)
		v.rawData()[i] = i + 1;

	auto b = v.broadcast({3, 4});
	CHECK(b.dim() == 2);
	CHECK(b.strides()[0] == 0);
	CHECK(b.strides()[1] == 1);
	CHECK(b.broadcasted());

	for (uint64_t r = 0; r < 3; r++)
		for (uint64_t c = 0; c < 4; c++)
			CHECK(b.at({r, c}) == static_cast<int>(c + 1));
}

TEST_CASE("tensorIMPL_broadcast4") {
	uint64_t shape[] = {3, 1};
	uint64_t strides[] = {1, 1};
	mem::Buffer buf(3 * sizeof(int));
	int* d = static_cast<int*>(buf->data());
	d[0] = 10;
	d[1] = 20;
	d[2] = 30;

	TensorIMPL<int> col(2, shape, strides, 0, buf);
	auto b = col.broadcast({3, 4});

	CHECK(b.strides()[0] == 1);
	CHECK(b.strides()[1] == 0);
	CHECK(b.broadcasted());

	for (uint64_t r = 0; r < 3; r++)
		for (uint64_t c = 0; c < 4; c++)
			CHECK(b.at({r, c}) == (int)((r + 1) * 10));
}

TEST_CASE("tensorIMPL_broadcast_error") {
	TensorIMPL<int> t({2, 3}, CPU);

	CHECK_THROWS_AS(t.broadcast({3}), std::runtime_error);
	CHECK_THROWS_AS(t.broadcast({4, 5}), std::runtime_error);
}

// # ============================================================

TEST_CASE("tensorIMPL_clone1") {
	TensorIMPL<int> t({2, 5}, CPU);
	for (int i = 0; i < 10; i++)
		t.rawData()[i] = i * 10;

	TensorIMPL<int> c = t.clone();
	CHECK(c.size() == t.size());
	CHECK(c.dim() == t.dim());
	CHECK(c.rawData() != t.rawData());
	CHECK(c.dense());

	for (int i = 0; i < 10; i++)
		CHECK(c.rawData()[i] == i * 10);

	c.rawData()[0] = 999;
	CHECK(t.rawData()[0] == 0);
}

TEST_CASE("tensorIMPL_clone2") {
	TensorIMPL<int> t({3, 3}, CPU);
	for (int i = 0; i < 9; i++)
		t.rawData()[i] = i;

	uint64_t sh[] = {3, 3};
	uint64_t st[] = {1, 3};
	TensorIMPL<int> tr(2, sh, st, 0, t.buffer());
	CHECK(!tr.dense());

	TensorIMPL<int> k = tr.clone();
	CHECK(k.dense());
	CHECK(k.rawData() != t.rawData());

	for (uint64_t r = 0; r < 3; r++)
		for (uint64_t c = 0; c < 3; c++)
			CHECK(k.at({r, c}) == t.at({c, r}));
}

// # ============================================================

TEST_CASE("tensorIMPL_move_semantics1") {
	TensorIMPL<int> t1({3, 3}, CPU);

	TensorIMPL<int> t2(t1);
	CHECK(t2.rawData() == t1.rawData());

	t1.rawData()[0] = 99;
	CHECK(t2.rawData()[0] == 99);
}

// spacial case: with TZ_NORMAL_EQUAL=1
TEST_CASE("tensorIMPL_move_semantics2") {
	TensorIMPL<int> t1({2, 2}, CPU);
	t1.rawData()[0] = 42;
	int* t1_raw = t1.rawData();

	TensorIMPL<int> t2({2, 2}, CPU);
	int* t2_raw = t2.rawData();

	t2 = t1;

	CHECK(t2.rawData()[0] == 42);
	CHECK(t2.rawData() == t2_raw);
	CHECK(t2.rawData() != t1.rawData());

	t2.rawData()[0] = 100;
	CHECK(t1.rawData()[0] == 42);
}

TEST_CASE("tensorIMPL_move_semantics3") {
	TensorIMPL<int> t1({2, 3}, CPU);
	t1.rawData()[0] = 77;

	TensorIMPL<int> t2({4, 5}, CPU);
	t2 = t1;

	CHECK(t2.rawData() == t1.rawData());
	CHECK(t2.shape()[0] == 2);
	CHECK(t2.shape()[1] == 3);
	CHECK(t2.size() == 4 * 5);

	t1.rawData()[0] = 99;
	CHECK(t2.rawData()[0] == 99);
}

TEST_CASE("tensorIMPL_move_semantics4") {
	TensorIMPL<int> t1({2, 2}, CPU);
	t1.rawData()[0] = 55;

	TensorIMPL<int> t2({2, 2}, CPU);
	int* t2_raw = t2.rawData();
	t2 = std::move(t1);

	CHECK(t2.rawData()[0] == 55);
	CHECK(t2.rawData() == t2_raw);
}

// # ============================================================

TEST_CASE("tensorIMPL_getCuda1") {
	TensorIMPL<float> t({2, 3, 4}, CPU);
	auto ct = t.getCudaTensor();

	CHECK(ct.dim == 3);
	CHECK(ct.offset == 0);
	CHECK(ct.size == 24);
	CHECK(ct.dense == true);
	CHECK(ct.data == t.rawData());
	CHECK(ct.shape[0] == 2);
	CHECK(ct.shape[1] == 3);
	CHECK(ct.shape[2] == 4);
	CHECK(ct.strides[0] == 12);
	CHECK(ct.strides[1] == 4);
	CHECK(ct.strides[2] == 1);
}

TEST_CASE("tensorIMPL_getCuda2") {
	TensorIMPL<int> t({4, 4}, CPU);
	auto row2 = t[2];
	auto ct = row2.getCudaTensor();

	CHECK(ct.offset == 8);
	CHECK(ct.dense == true);
	CHECK(ct.dim == 1);
	CHECK(ct.data == t.rawData());
}

// # ============================================================

TEST_CASE("tensorIMPL_printing_empty") {
	TensorIMPL<int> t({0}, CPU);
	std::ostringstream oss;
	oss << t;
	CHECK(oss.str() == "[]");
}

TEST_CASE("tensorIMPL_printing_scalar") {
	TensorIMPL<int> t(0, nullptr, CPU);
	t.get() = 42;
	std::ostringstream oss;
	oss << t;
	CHECK(oss.str() == "42");
}

TEST_CASE("tensorIMPL_printing_vector") {
	TensorIMPL<int> t({3}, CPU);
	t.at({0}) = 1;
	t.at({1}) = 2;
	t.at({2}) = 3;
	std::ostringstream oss;
	oss << t;
	CHECK(oss.str() == "\n[1, 2, 3]");
}

TEST_CASE("tensorIMPL_printing_matrix") {
	TensorIMPL<int> t({2, 2}, CPU);
	t.at({0, 0}) = 1;
	t.at({0, 1}) = 2;
	t.at({1, 0}) = 3;
	t.at({1, 1}) = 4;
	std::ostringstream oss;
	oss << t;
	CHECK(oss.str() == "[\n[1, 2],\n[3, 4]\n]");
}
