#include <Tenzor.hpp>
#include "tensor_impl.hpp"
#define DOCTEST_CONFIG_COLORS
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <vector>
#include <sstream>

using namespace TZ;
using namespace TZ::internal;

TEST_CASE("tensor_impl_default_constructor") {
	TensorIMPL<float> t;
	CHECK(t.dim() == 0);
	CHECK(t.size() == 0);
	CHECK(t.empty() == true);
	CHECK(t.scalar() == false);
	CHECK(t.data() == nullptr);
	CHECK(t.rawData() == nullptr);
}

TEST_CASE("tensor_impl_init_list_constructor") {
	TensorIMPL<int> t({2, 3, 4}, CPU);

	CHECK(t.dim() == 3);
	CHECK(t.size() == 24);
	CHECK(t.empty() == false);
	CHECK(t.scalar() == false);
	CHECK(t.dense() == true);
	CHECK(t.offset() == 0);
	CHECK(t.device() == CPU);

	const uint64_t* shape = t.shape();
	CHECK(shape[0] == 2);
	CHECK(shape[1] == 3);
	CHECK(shape[2] == 4);

	const uint64_t* strides = t.strides();
	CHECK(strides[0] == 12);
	CHECK(strides[1] == 4);
	CHECK(strides[2] == 1);

	CHECK(t.data() != nullptr);
	CHECK(t.rawData() == t.data());
}

TEST_CASE("tensor_impl_indexing_at") {
	TensorIMPL<int> t({2, 2}, CPU);

	t.at({0, 0}) = 1;
	t.at({0, 1}) = 2;
	t.at({1, 0}) = 3;
	t.at({1, 1}) = 4;

	CHECK(t.at({0, 0}) == 1);
	CHECK(t.at({0, 1}) == 2);
	CHECK(t.at({1, 0}) == 3);
	CHECK(t.at({1, 1}) == 4);

	uint64_t idx1[] = {0, 1};
	uint64_t idx2[] = {1, 0};
	CHECK(t.at(idx1) == 2);
	CHECK(t.at(idx2) == 3);
}

TEST_CASE("tensor_impl_operator_brackets") {
	TensorIMPL<float> t({3, 4}, CPU);

	for (int i = 0; i < 12; ++i) {
		t.rawData()[i] = static_cast<float>(i);
	}

	auto row0 = t[0];
	CHECK(row0.dim() == 1);
	CHECK(row0.size() == 4);
	CHECK(row0.shape()[0] == 4);
	CHECK(row0.strides()[0] == 1);
	CHECK(row0.offset() == 0);

	auto row1 = t[1];
	CHECK(row1.offset() == 4);
	CHECK(row1[0].get() ==
	      doctest::Approx(4.0f)); // First element of row 1, assuming get works on scalar or offset

	auto val = row1[2];
	CHECK(val.dim() == 0);
	CHECK(val.size() == 1);
	CHECK(val.scalar() == true);
	CHECK(val.get() == doctest::Approx(6.0f));
}

TEST_CASE("tensor_impl_clone") {
	TensorIMPL<int> t1({2, 5}, CPU);
	for (int i = 0; i < 10; ++i) {
		t1.rawData()[i] = i * 10;
	}

	TensorIMPL<int> t2 = t1.clone();
	CHECK(t2.dim() == t1.dim());
	CHECK(t2.size() == t1.size());
	CHECK(t2.data() != t1.data());

	for (int i = 0; i < 10; ++i) {
		CHECK(t2.rawData()[i] == i * 10);
	}

	t2.rawData()[0] = 999;
	CHECK(t1.rawData()[0] == 0);
}

TEST_CASE("tensor_impl_scalar") {
	uint64_t shape[] = {1}; // Dummy shape, actually dim=0
	TensorIMPL<double> t(0, shape, CPU);

	CHECK(t.dim() == 0);
	CHECK(t.size() == 1);
	CHECK(t.scalar() == true);

	t.get() = 3.1415;
	CHECK(t.get() == doctest::Approx(3.1415));
}

TEST_CASE("tensor_impl_empty") {
	TensorIMPL<float> t({0, 5, 5}, CPU);
	CHECK(t.empty() == true);
	CHECK(t.size() == 0);
	CHECK(t.indexable() == false);
}

TEST_CASE("tensor_impl_copy_assignment") {
	TensorIMPL<int> t1({2, 2}, CPU);
	t1.at({0, 0}) = 42;

	TensorIMPL<int> t2({2, 2}, CPU);
	t2 = t1;
	// Depends on TZ_NORMAL_EQUAL, if it copies data or moves reference.
	// The implementation states it copies data if shape/device match.

	CHECK(t2.at({0, 0}) == 42);
	t2.at({0, 0}) = 100;
	CHECK(t1.at({0, 0}) == 42); // Assumes deep copy via apply(Copy)
}

TEST_CASE("tensor_impl_boundary_exceptions") {
	TensorIMPL<int> t({2, 2}, CPU);

	// Out of bounds for at()
	CHECK_THROWS_AS(t.at({2, 0}), std::runtime_error);
	CHECK_THROWS_AS(t.at({0, 2}), std::runtime_error);

	uint64_t bad_idx[] = {3, 0};
	CHECK_THROWS_AS(t.at(bad_idx), std::runtime_error);

	// Out of bounds for operator[]
	CHECK_THROWS_AS(t[2], std::runtime_error);
}

TEST_CASE("tensor_impl_stream_formatting") {
	std::ostringstream oss;

	// Empty tensor
	TensorIMPL<int> t_empty({0}, CPU);
	oss << t_empty;
	CHECK(oss.str() == "[]");
	oss.str(""); // clear

	// Scalar tensor
	uint64_t shape[] = {1};
	TensorIMPL<int> t_scalar(0, shape, CPU);
	t_scalar.get() = 42;
	oss << t_scalar;
	CHECK(oss.str() == "42");
	oss.str("");

	// 1D tensor
	TensorIMPL<int> t_1d({3}, CPU);
	t_1d.at({0}) = 1;
	t_1d.at({1}) = 2;
	t_1d.at({2}) = 3;
	oss << t_1d;
	CHECK(oss.str() == "\n[1, 2, 3]");
	oss.str("");

	// 2D tensor
	TensorIMPL<int> t_2d({2, 2}, CPU);
	t_2d.at({0, 0}) = 1;
	t_2d.at({0, 1}) = 2;
	t_2d.at({1, 0}) = 3;
	t_2d.at({1, 1}) = 4;
	oss << t_2d;
	CHECK(oss.str() == "[\n[1, 2],\n[3, 4]\n]");
}
