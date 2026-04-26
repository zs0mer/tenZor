#include <Tenzor.hpp>
#include "buffer.hpp"
#define DOCTEST_CONFIG_COLORS
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <vector>

using namespace TZ::mem;

TEST_CASE("buffer_default_constructor") {
	Buffer b;
	CHECK(b->size() == 0);
	CHECK(b->data() == nullptr);
	CHECK(b->device() == TZ::CPU);
}

TEST_CASE("buffer_size_constructor") {
	uint64_t size = 1024;
	Buffer b(size);
	CHECK(b->size() == size);
	CHECK(b->data() != nullptr);
	CHECK(b->device() == TZ::CPU);

	// Test writing to buffer
	uint8_t* ptr = static_cast<uint8_t*>(b->data());
	for (uint64_t i = 0; i < size; ++i) {
		ptr[i] = static_cast<uint8_t>(i % 256);
	}

	for (uint64_t i = 0; i < size; ++i) {
		CHECK(ptr[i] == static_cast<uint8_t>(i % 256));
	}
}

TEST_CASE("buffer_copy_constructor") {
	uint64_t size = 512;
	Buffer b1(size);

	uint8_t* ptr1 = static_cast<uint8_t*>(b1->data());
	ptr1[0] = 42;
	ptr1[size - 1] = 84;

	Buffer b2(b1);
	CHECK(b2->size() == size);
	CHECK(b2->data() == b1->data());

	uint8_t* ptr2 = static_cast<uint8_t*>(b2->data());
	CHECK(ptr2[0] == 42);
	CHECK(ptr2[size - 1] == 84);
}

TEST_CASE("buffer_copy_assignment") {
	uint64_t size1 = 256;
	uint64_t size2 = 512;

	Buffer b1(size1);
	Buffer b2(size2);

	uint8_t* ptr1 = static_cast<uint8_t*>(b1->data());
	ptr1[0] = 10;

	b2 = b1;

	CHECK(b2->size() == size1);
	CHECK(b2->data() == b1->data());

	uint8_t* ptr2 = static_cast<uint8_t*>(b2->data());
	CHECK(ptr2[0] == 10);
}

TEST_CASE("buffer_move_constructor") {
	uint64_t size = 128;
	Buffer b1(size);
	void* orig_data = b1->data();

	Buffer b2(std::move(b1));

	CHECK(b2->size() == size);
	CHECK(b2->data() == orig_data);
	// b1 is now empty/null, but since we shouldn't dereference b1-> without checking,
	// let's just assume move semantics left it in a valid but unspecified state,
	// or in this case, its internal pointer is nullptr.
}

TEST_CASE("buffer_move_assignment") {
	uint64_t size = 64;
	Buffer b1(size);
	void* orig_data = b1->data();

	Buffer b2;
	b2 = std::move(b1);

	CHECK(b2->size() == size);
	CHECK(b2->data() == orig_data);
}

TEST_CASE("buffer_clone") {
	uint64_t size = 1024;
	Buffer b1(size);
	uint8_t* ptr1 = static_cast<uint8_t*>(b1->data());
	for (uint64_t i = 0; i < size; ++i) {
		ptr1[i] = static_cast<uint8_t>((i * 3) % 256);
	}

	Buffer b2 = b1.clone();

	CHECK(b2->size() == size);
	CHECK(b2->data() != b1->data());

	uint8_t* ptr2 = static_cast<uint8_t*>(b2->data());
	for (uint64_t i = 0; i < size; ++i) {
		CHECK(ptr2[i] == ptr1[i]);
	}

	// Modifying b2 shouldn't affect b1
	ptr2[0] = 255;
	CHECK(ptr1[0] != 255);
}

TEST_CASE("buffer_multiple_references") {
	Buffer b1(2048);
	std::vector<Buffer> buffers;

	for (int i = 0; i < 100; ++i) {
		buffers.push_back(b1);
	}

	for (int i = 0; i < 100; ++i) {
		CHECK(buffers[i]->data() == b1->data());
		CHECK(buffers[i]->size() == 2048);
	}

	buffers.clear();
	// b1 should still be valid
	CHECK(b1->size() == 2048);
	CHECK(b1->data() != nullptr);
}
