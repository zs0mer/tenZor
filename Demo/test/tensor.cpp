#include <Tenzor.hpp>
#include "tensor_impl.hpp"
#define DOCTEST_CONFIG_COLORS
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <sstream>
#include <vector>

using namespace TZ;
using namespace TZ::impl;

//& ============================================================
//& Default construction
//& ============================================================

TEST_CASE("tensor_impl_default_ctor") {
	TensorIMPL<float> t;
	CHECK(t.dim() == 0);
	CHECK(t.size() == 0);
	CHECK(t.empty());
	CHECK(!t.scalar());
	CHECK(t.dense());
	CHECK(!t.broadcasted());
	CHECK(t.data() == nullptr);
	CHECK(t.rawData() == nullptr);
}

//& ============================================================
//& Stride computation
//& ============================================================

TEST_CASE("tensor_impl_strides_1d") {
	TensorIMPL<float> t({7}, CPU);
	CHECK(t.dim() == 1);
	CHECK(t.size() == 7);
	CHECK(t.strides()[0] == 1);
	CHECK(t.dense());
	CHECK(t.shape()[0] == 7);
}

TEST_CASE("tensor_impl_strides_2d") {
	TensorIMPL<int> t({5, 3}, CPU);
	// row-major: strides = {cols, 1}
	CHECK(t.strides()[0] == 3);
	CHECK(t.strides()[1] == 1);
	CHECK(t.size() == 15);
	CHECK(t.dense());
}

TEST_CASE("tensor_impl_strides_3d") {
	TensorIMPL<double> t({2, 3, 4}, CPU);
	// row-major: strides = {3*4, 4, 1}
	CHECK(t.strides()[0] == 12);
	CHECK(t.strides()[1] == 4);
	CHECK(t.strides()[2] == 1);
	CHECK(t.size() == 24);
	CHECK(t.dense());
}

TEST_CASE("tensor_impl_strides_4d") {
	TensorIMPL<float> t({2, 3, 4, 5}, CPU);
	// row-major: strides = {60, 20, 5, 1}
	CHECK(t.strides()[0] == 60);
	CHECK(t.strides()[1] == 20);
	CHECK(t.strides()[2] == 5);
	CHECK(t.strides()[3] == 1);
	CHECK(t.size() == 120);
}

//& ============================================================
//& data() vs rawData() — offset tracking
//& ============================================================

TEST_CASE("tensor_impl_data_equals_rawdata_when_no_offset") {
	TensorIMPL<int> t({4, 4}, CPU);
	CHECK(t.data() == t.rawData());
	CHECK(t.offset() == 0);
}

TEST_CASE("tensor_impl_data_offset_after_slice") {
	// For a 3x4 tensor, t[1] should have offset = 1 * strides[0] = 4
	TensorIMPL<int> t({3, 4}, CPU);
	auto row1 = t[1];
	CHECK(row1.offset() == 4);
	CHECK(row1.rawData() == t.rawData());  // same underlying allocation
	CHECK(row1.data() == t.rawData() + 4); // data() = rawData() + offset
}

//& ============================================================
//& at() indexing
//& ============================================================

TEST_CASE("tensor_impl_at_vector_idx_2d") {
	TensorIMPL<int> t({3, 3}, CPU);
	for (int i = 0; i < 9; ++i)
		t.rawData()[i] = i;

	CHECK(t.at({0, 0}) == 0);
	CHECK(t.at({0, 1}) == 1);
	CHECK(t.at({0, 2}) == 2);
	CHECK(t.at({1, 0}) == 3);
	CHECK(t.at({2, 2}) == 8);
}

TEST_CASE("tensor_impl_at_carray_idx_2d") {
	TensorIMPL<int> t({2, 4}, CPU);
	for (int i = 0; i < 8; ++i)
		t.rawData()[i] = i * 10;

	uint64_t idx0[] = {0, 3};
	uint64_t idx1[] = {1, 2};
	CHECK(t.at(idx0) == 30);
	CHECK(t.at(idx1) == 60);
}

TEST_CASE("tensor_impl_at_3d") {
	TensorIMPL<int> t({2, 3, 4}, CPU);
	for (int i = 0; i < 24; ++i)
		t.rawData()[i] = i;

	// at({r, c, d}) = r*12 + c*4 + d
	CHECK(t.at({0, 0, 0}) == 0);
	CHECK(t.at({1, 0, 0}) == 12);
	CHECK(t.at({1, 2, 3}) == 23);
	CHECK(t.at({0, 2, 1}) == 9);
}

TEST_CASE("tensor_impl_at_wrong_dim_count_throws") {
	TensorIMPL<int> t({2, 3}, CPU);
	CHECK_THROWS_AS(t.at({0}), std::runtime_error);
	CHECK_THROWS_AS(t.at({0, 0, 0}), std::runtime_error);
}

TEST_CASE("tensor_impl_at_out_of_bounds_throws") {
	TensorIMPL<int> t({2, 2}, CPU);
	CHECK_THROWS_AS(t.at({2, 0}), std::runtime_error);
	CHECK_THROWS_AS(t.at({0, 2}), std::runtime_error);

	uint64_t bad[] = {5, 0};
	CHECK_THROWS_AS(t.at(bad), std::runtime_error);
}

//& ============================================================
//& operator[] — slicing and offset propagation
//& ============================================================

TEST_CASE("tensor_impl_operator_brackets_produces_correct_slice") {
	TensorIMPL<float> t({3, 4}, CPU);
	for (int i = 0; i < 12; ++i)
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

TEST_CASE("tensor_impl_chained_indexing_3d") {
	TensorIMPL<int> t({2, 3, 4}, CPU);
	for (int i = 0; i < 24; ++i)
		t.rawData()[i] = i;

	// t[1][2][3] has offset = 1*12 + 2*4 + 3*1 = 23
	auto s = t[1][2][3];
	CHECK(s.dim() == 0);
	CHECK(s.scalar());
	CHECK(s.get() == 23);
	CHECK(s.offset() == 23);
}

TEST_CASE("tensor_impl_operator_brackets_oob_throws") {
	TensorIMPL<int> t({3, 3}, CPU);
	CHECK_THROWS_AS(t[3], std::runtime_error);
	CHECK_THROWS_AS(t[0][3], std::runtime_error);
}

//& ============================================================
//& Scalar tensor
//& ============================================================

TEST_CASE("tensor_impl_scalar_construction_and_get") {
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

TEST_CASE("tensor_impl_get_on_non_scalar_throws") {
	TensorIMPL<int> t({3}, CPU);
	CHECK_THROWS_AS(t.get(), std::runtime_error);

	TensorIMPL<int> t2({2, 2}, CPU);
	CHECK_THROWS_AS(t2.get(), std::runtime_error);
}

TEST_CASE("tensor_impl_scalar_from_bracket_chain") {
	// Indexing all dims produces a scalar
	TensorIMPL<int> t({4, 4}, CPU);
	for (int i = 0; i < 16; ++i)
		t.rawData()[i] = i;
	auto s = t[3][3];
	CHECK(s.scalar());
	CHECK(s.get() == 15);
}

//& ============================================================
//& Empty tensor
//& ============================================================

TEST_CASE("tensor_impl_empty_zero_in_shape") {
	TensorIMPL<float> t({0, 5, 5}, CPU);
	CHECK(t.empty());
	CHECK(t.size() == 0);
	CHECK(!t.indexable());
	CHECK(!t.scalar());
}

TEST_CASE("tensor_impl_size_is_product_of_dims") {
	TensorIMPL<int> t({2, 5, 3}, CPU);
	CHECK(t.size() == 30);
	CHECK(!t.empty());
	CHECK(t.indexable());
}

//& ============================================================
//& dense / broadcasted flags
//& ============================================================

TEST_CASE("tensor_impl_dense_flag_standard_tensor") {
	TensorIMPL<int> t({4, 3}, CPU);
	CHECK(t.dense());
	CHECK(!t.broadcasted());
}

TEST_CASE("tensor_impl_dense_flag_custom_nonstd_strides") {
	// Column-major strides {1, rows} are not standard row-major -> not dense
	uint64_t shape[] = {3, 4};
	uint64_t strides[] = {1, 3}; // column-major
	mem::Buffer buf(3 * 4 * sizeof(int));
	TensorIMPL<int> t(2, shape, strides, 0, buf);
	CHECK(!t.dense());
	CHECK(!t.broadcasted());
}

TEST_CASE("tensor_impl_broadcasted_flag_via_zero_stride") {
	// Build a broadcasted view by passing a zero stride for the first dim
	TensorIMPL<float> src({4}, CPU);
	for (int i = 0; i < 4; ++i)
		src.rawData()[i] = static_cast<float>(i);

	uint64_t shape[] = {3, 4};
	uint64_t strides[] = {0, 1}; // row stride = 0 -> broadcast
	TensorIMPL<float> b(2, shape, strides, 0, src.buffer());

	CHECK(b.broadcasted());
	CHECK(!b.dense());
	CHECK(b.size() == 12);

	// Every row should be the same as the original
	for (uint64_t r = 0; r < 3; ++r)
		for (uint64_t c = 0; c < 4; ++c)
			CHECK(b.at({r, c}) == doctest::Approx(static_cast<float>(c)));
}

//& ============================================================
//& broadcast() method
//& ============================================================

TEST_CASE("tensor_impl_broadcast_scalar_to_2d") {
	TensorIMPL<int> s(0, nullptr, CPU);
	s.get() = 7;

	auto b = s.broadcast({3, 4});
	CHECK(b.dim() == 2);
	CHECK(b.shape()[0] == 3);
	CHECK(b.shape()[1] == 4);
	CHECK(b.strides()[0] == 0);
	CHECK(b.strides()[1] == 0);
	CHECK(b.broadcasted());

	for (uint64_t i = 0; i < 3; ++i)
		for (uint64_t j = 0; j < 4; ++j)
			CHECK(b.at({i, j}) == 7);
}

TEST_CASE("tensor_impl_broadcast_row_vector_to_2d") {
	// Vector of size 4, broadcast to {3, 4}: each row becomes the vector
	TensorIMPL<int> v({4}, CPU);
	for (int i = 0; i < 4; ++i)
		v.rawData()[i] = i + 1; // [1,2,3,4]

	auto b = v.broadcast({3, 4});
	CHECK(b.dim() == 2);
	CHECK(b.strides()[0] == 0); // row stride is broadcast
	CHECK(b.strides()[1] == 1);
	CHECK(b.broadcasted());

	for (uint64_t r = 0; r < 3; ++r)
		for (uint64_t c = 0; c < 4; ++c)
			CHECK(b.at({r, c}) == static_cast<int>(c + 1));
}

TEST_CASE("tensor_impl_broadcast_col_vector_to_2d") {
	// 3x1 matrix, broadcast to {3, 4}: each col becomes the vector
	uint64_t shape[] = {3, 1};
	uint64_t strides[] = {1, 1};
	mem::Buffer buf(3 * sizeof(int));
	int* d = static_cast<int*>(buf->data());
	d[0] = 10;
	d[1] = 20;
	d[2] = 30;

	TensorIMPL<int> col(2, shape, strides, 0, buf);
	auto b = col.broadcast({3, 4});

	CHECK(b.strides()[0] == 1); // row stride kept
	CHECK(b.strides()[1] == 0); // col stride is broadcast
	CHECK(b.broadcasted());

	for (uint64_t r = 0; r < 3; ++r)
		for (uint64_t c = 0; c < 4; ++c)
			CHECK(b.at({r, c}) == (int)((r + 1) * 10));
}

TEST_CASE("tensor_impl_broadcast_incompatible_shapes_throws") {
	TensorIMPL<int> t({2, 3}, CPU);
	// target dim < source dim
	CHECK_THROWS_AS(t.broadcast({3}), std::runtime_error);
	// shape[1] is 3, target shape[1] is 5 -> not broadcastable (neither 1 nor equal)
	CHECK_THROWS_AS(t.broadcast({4, 5}), std::runtime_error);
}

//& ============================================================
//& clone()
//& ============================================================

TEST_CASE("tensor_impl_clone_dense_memcpy_path") {
	TensorIMPL<int> t({2, 5}, CPU);
	for (int i = 0; i < 10; ++i)
		t.rawData()[i] = i * 10;

	TensorIMPL<int> c = t.clone();
	CHECK(c.size() == t.size());
	CHECK(c.dim() == t.dim());
	CHECK(c.rawData() != t.rawData()); // independent allocation
	CHECK(c.dense());

	for (int i = 0; i < 10; ++i)
		CHECK(c.rawData()[i] == i * 10);

	c.rawData()[0] = 999;
	CHECK(t.rawData()[0] == 0); // original untouched
}

TEST_CASE("tensor_impl_clone_nondense_apply_path") {
	// Clone a transposed (non-dense) view; result must be dense and logically equal
	TensorIMPL<int> t({3, 3}, CPU);
	for (int i = 0; i < 9; ++i)
		t.rawData()[i] = i;
	// t as a matrix: [[0,1,2],[3,4,5],[6,7,8]]

	// Build a transposed view: shape={3,3}, strides={1,3}
	uint64_t sh[] = {3, 3}, st[] = {1, 3};
	TensorIMPL<int> tr(2, sh, st, 0, t.buffer());
	CHECK(!tr.dense());

	TensorIMPL<int> k = tr.clone();
	CHECK(k.dense());
	CHECK(k.rawData() != t.rawData());

	// Logical values after transpose: (r,c) -> original(c,r)
	for (uint64_t r = 0; r < 3; ++r)
		for (uint64_t c = 0; c < 3; ++c)
			CHECK(k.at({r, c}) == t.at({c, r}));
}

//& ============================================================
//& Copy / move semantics
//& ============================================================

TEST_CASE("tensor_impl_copy_ctor_shares_buffer") {
	// The defaulted copy constructor shares the Buffer (ref-counted).
	TensorIMPL<int> t1({3, 3}, CPU);
	for (int i = 0; i < 9; ++i)
		t1.rawData()[i] = i;

	TensorIMPL<int> t2(t1);              // copy construct
	CHECK(t2.rawData() == t1.rawData()); // same underlying allocation

	// A write through t1 is visible through t2
	t1.rawData()[0] = 99;
	CHECK(t2.rawData()[0] == 99);
}

TEST_CASE("tensor_impl_copy_assignment_same_shape_deep_copies") {
	// With TZ_NORMAL_EQUAL=1 and matching shape+device, assignment copies data.
	TensorIMPL<int> t1({2, 2}, CPU);
	t1.rawData()[0] = 42;
	int* t1_raw = t1.rawData();

	TensorIMPL<int> t2({2, 2}, CPU);
	int* t2_raw = t2.rawData();

	t2 = t1;

	CHECK(t2.rawData()[0] == 42);        // data was copied
	CHECK(t2.rawData() == t2_raw);       // t2 kept its own buffer
	CHECK(t2.rawData() != t1.rawData()); // independent

	t2.rawData()[0] = 100;
	CHECK(t1.rawData()[0] == 42); // t1 untouched
}

TEST_CASE("tensor_impl_copy_assignment_diff_shape_shares_buffer") {
	// Shapes differ -> the fallback path just shares the Buffer.
	TensorIMPL<int> t1({2, 3}, CPU);
	t1.rawData()[0] = 77;

	TensorIMPL<int> t2({4, 5}, CPU); // different shape
	t2 = t1;

	CHECK(t2.rawData() == t1.rawData()); // now share
	CHECK(t2.shape()[0] == 2);
	CHECK(t2.shape()[1] == 3);
	CHECK(t2.size() == 4 * 5);

	t1.rawData()[0] = 99;
	CHECK(t2.rawData()[0] == 99); // visible through shared buffer
}

TEST_CASE("tensor_impl_move_assignment_same_shape_copies_data") {
	// With TZ_NORMAL_EQUAL=1 and same shape, move-assign also copies data.
	TensorIMPL<int> t1({2, 2}, CPU);
	t1.rawData()[0] = 55;
	int* t2_raw_before;

	TensorIMPL<int> t2({2, 2}, CPU);
	t2_raw_before = t2.rawData();
	t2 = std::move(t1);

	CHECK(t2.rawData()[0] == 55);
	CHECK(t2.rawData() == t2_raw_before); // t2 kept its own buffer
}

//& ============================================================
//& Low-level constructor + buffer/device/allocator accessors
//& ============================================================

TEST_CASE("tensor_impl_low_level_ctor_custom_strides") {
	// Build a 2x3 column-major view over a flat buffer
	uint64_t shape[] = {2, 3};
	uint64_t strides[] = {1, 2}; // column-major: row stride=1, col stride=2
	mem::Buffer buf(6 * sizeof(int));
	int* d = static_cast<int*>(buf->data());
	for (int i = 0; i < 6; ++i)
		d[i] = i;
	// d = [0,1,2,3,4,5]
	// (r,c) = d[r*1 + c*2]
	// (0,0)=0, (1,0)=1, (0,1)=2, (1,1)=3, (0,2)=4, (1,2)=5

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

TEST_CASE("tensor_impl_low_level_ctor_with_offset") {
	// Share a buffer between a parent tensor and a view with non-zero offset
	TensorIMPL<int> parent({4, 4}, CPU);
	for (int i = 0; i < 16; ++i)
		parent.rawData()[i] = i;

	uint64_t shape[] = {4};
	uint64_t strides[] = {1};
	// View of row 2: offset = 2 * 4 = 8
	TensorIMPL<int> view(1, shape, strides, 8, parent.buffer());

	CHECK(view.offset() == 8);
	CHECK(view.data() == parent.rawData() + 8);
	CHECK(view.at({0}) == 8);
	CHECK(view.at({3}) == 11);
}

TEST_CASE("tensor_impl_buffer_device_allocator_accessors") {
	TensorIMPL<float> t({3, 3}, CPU);
	CHECK(t.device() == CPU);
	CHECK(&t.allocator() == &mem::defaultAllocator(CPU));
	CHECK(t.buffer()->size() == 9 * sizeof(float));
}

//& ============================================================
//& getCudaTensor() — metadata only (no GPU needed)
//& ============================================================

TEST_CASE("tensor_impl_get_cuda_tensor_metadata") {
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

TEST_CASE("tensor_impl_get_cuda_tensor_slice_has_offset") {
	TensorIMPL<int> t({4, 4}, CPU);
	auto row2 = t[2];
	auto ct = row2.getCudaTensor();
	CHECK(ct.offset == 8);
	CHECK(ct.dim == 1);
	CHECK(ct.data == t.rawData()); // raw pointer is the same base
}

//& ============================================================
//& Streaming output
//& ============================================================

TEST_CASE("tensor_impl_stream_empty") {
	TensorIMPL<int> t({0}, CPU);
	std::ostringstream oss;
	oss << t;
	CHECK(oss.str() == "[]");
}

TEST_CASE("tensor_impl_stream_scalar") {
	TensorIMPL<int> t(0, nullptr, CPU);
	t.get() = 42;
	std::ostringstream oss;
	oss << t;
	CHECK(oss.str() == "42");
}

TEST_CASE("tensor_impl_stream_1d") {
	TensorIMPL<int> t({3}, CPU);
	t.at({0}) = 1;
	t.at({1}) = 2;
	t.at({2}) = 3;
	std::ostringstream oss;
	oss << t;
	CHECK(oss.str() == "\n[1, 2, 3]");
}

TEST_CASE("tensor_impl_stream_2d") {
	TensorIMPL<int> t({2, 2}, CPU);
	t.at({0, 0}) = 1;
	t.at({0, 1}) = 2;
	t.at({1, 0}) = 3;
	t.at({1, 1}) = 4;
	std::ostringstream oss;
	oss << t;
	CHECK(oss.str() == "[\n[1, 2],\n[3, 4]\n]");
}
