#define DOCTEST_CONFIG_COLORS
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <vector>

#include <Tenzor.hpp>

using namespace TZ;
using namespace TZ::mem;

// # ============================================================

TEST_CASE("bufferimpl_size_accessor") {
	Buffer b(256);
	CHECK(b->size() == 256);
}

TEST_CASE("bufferimpl_data_non_null") {
	Buffer b(64);
	CHECK(b->data() != nullptr);
}

TEST_CASE("bufferimpl_data_null_on_default") {
	Buffer b;
	CHECK(b->data() == nullptr);
}

TEST_CASE("bufferimpl_const_data_accessor") {
	Buffer b(32);
	uint8_t* pw = static_cast<uint8_t*>(b->data());
	pw[0] = 0xBE;
	pw[31] = 0xEF;

	const Buffer& cb = b;
	const uint8_t* pr = static_cast<const uint8_t*>(cb->data());
	CHECK(pr[0] == 0xBE);
	CHECK(pr[31] == 0xEF);
}

TEST_CASE("bufferimpl_device_cpu_for_default_alloc") {
	Buffer b(128);
	CHECK(b->device() == CPU);
}

TEST_CASE("bufferimpl_allocator_matches_default") {
	Buffer b(128);
	CHECK(b->allocator() == &defaultAllocator(CPU));
}

TEST_CASE("bufferimpl_allocator_matches_given") {
	Buffer b(512, &Malloc::instance());
	CHECK(b->allocator() == &Malloc::instance());
	CHECK(b->device() == CPU);
}

// # ============================================================

TEST_CASE("buffer_constructor_default") {
	Buffer b;
	CHECK(b->size() == 0);
	CHECK(b->data() == nullptr);
	CHECK(b->device() == CPU);
}

TEST_CASE("buffer_constructor_zero") {
	Buffer b(uint64_t(0));
	CHECK(b->size() == 0);
	CHECK(b->data() == nullptr);
}

TEST_CASE("buffer_constructor_nullptr") {
	Buffer b(nullptr);
	CHECK(b->size() == 0);
	CHECK(b->data() == nullptr);
}

TEST_CASE("buffer_constructor") {
	constexpr uint64_t sz = 1024;
	Buffer b(sz);
	CHECK(b->size() == sz);
	CHECK(b->data() != nullptr);

	uint8_t* p = static_cast<uint8_t*>(b->data());
	for (uint64_t i = 0; i < sz; i++)
		p[i] = static_cast<uint8_t>(i % 256);
	for (uint64_t i = 0; i < sz; i++)
		CHECK(p[i] == static_cast<uint8_t>(i % 256));
}


TEST_CASE("buffer_constructor_boundary_sizes") {
	std::vector<uint64_t> sizes = {
	    1, 7, 8, 63, 64, 4095, 4096, 4097, 1024 * 1024 - 1, 1024 * 1024, 1024 * 1024 + 1};
	for (uint64_t sz : sizes) {
		Buffer b(sz);
		CHECK(b->size() == sz);
		CHECK(b->data() != nullptr);
		static_cast<uint8_t*>(b->data())[sz - 1] = 0xFF;
		CHECK(static_cast<uint8_t*>(b->data())[sz - 1] == 0xFF);
	}
}

TEST_CASE("buffer_constructor_copy") {
	Buffer b1(64);
	Buffer b2(b1);

	CHECK(b2->data() == b1->data());
	CHECK(b2->size() == 64);

	static_cast<uint8_t*>(b1->data())[3] = 0xAB;
	CHECK(static_cast<uint8_t*>(b2->data())[3] == 0xAB);

	static_cast<uint8_t*>(b2->data())[3] = 0xCD;
	CHECK(static_cast<uint8_t*>(b1->data())[3] == 0xCD);
}

TEST_CASE("buffer_copy_assignment") {
	Buffer b1(256);
	static_cast<uint8_t*>(b1->data())[10] = 0xCC;

	Buffer b2(128);
	b2 = b1;

	CHECK(b2->data() == b1->data());
	CHECK(b2->size() == 256);
	CHECK(static_cast<uint8_t*>(b2->data())[10] == 0xCC);
}

TEST_CASE("buffer_copy_assignment_self_assign") {
	Buffer b(64);
	static_cast<uint8_t*>(b->data())[0] = 0xCF;
	void* first = b->data();

	b = b;

	CHECK(b->data() == first);
	CHECK(b->size() == 64);
	CHECK(static_cast<uint8_t*>(b->data())[0] == 0xCF);
}

// # ============================================================

TEST_CASE("buffer_move") {
	Buffer b1(128);
	void* first = b1->data();

	Buffer b2(std::move(b1));
	CHECK(b2->size() == 128);
	CHECK(b2->data() == first);
}

TEST_CASE("buffer_move_assignment") {
	Buffer b1(256);
	static_cast<uint8_t*>(b1->data())[0] = 0x11;
	void* b1_data = b1->data();

	Buffer b2(512);
	static_cast<uint8_t*>(b2->data())[0] = 0x22;

	b2 = std::move(b1);

	CHECK(b2->data() == b1_data);
	CHECK(b2->size() == 256);
	CHECK(static_cast<uint8_t*>(b2->data())[0] == 0x11);
}

// # ============================================================

TEST_CASE("buffer_clone") {
	const uint64_t sz = 1024;
	Buffer b1(sz);
	uint8_t* p1 = static_cast<uint8_t*>(b1->data());
	for (uint64_t i = 0; i < sz; i++)
		p1[i] = static_cast<uint8_t>((i * 7 + 3) % 256);

	Buffer b2 = b1.clone();

	CHECK(b2->data() != b1->data());
	CHECK(b2->size() == sz);
	uint8_t* p2 = static_cast<uint8_t*>(b2->data());
	for (uint64_t i = 0; i < sz; i++)
		CHECK(p2[i] == p1[i]);

	p2[0] = ~p2[0];
	CHECK(p1[0] != p2[0]);
}

TEST_CASE("buffer_clone_empty") {
	Buffer b1;
	Buffer b2 = b1.clone();
	CHECK(b2->size() == 0);
	CHECK(b2->data() == nullptr);
}

// # ============================================================

TEST_CASE("buffer_reference_counting1") {
	Buffer b0(2048);
	static_cast<uint8_t*>(b0->data())[0] = 0xDE;
	void* first = b0->data();

	std::vector<Buffer> copies;
	for (int i = 0; i < 200; i++)
		copies.push_back(b0);

	for (auto& c : copies) {
		CHECK(c->data() == first);
		CHECK(c->size() == 2048);
		CHECK(static_cast<uint8_t*>(c->data())[0] == 0xDE);
	}

	copies.clear();
	CHECK(b0->data() == first);
	CHECK(static_cast<uint8_t*>(b0->data())[0] == 0xDE);
}

TEST_CASE("buffer_reference_counting2") {
	Buffer* b1 = new Buffer(512);
	static_cast<uint8_t*>((*b1)->data())[0] = 0xFF;
	void* ptr = (*b1)->data();

	Buffer* b2 = new Buffer(*b1);
	delete b1;

	CHECK((*b2)->data() == ptr);
	CHECK(static_cast<uint8_t*>((*b2)->data())[0] == 0xFF);

	delete b2;
}

TEST_CASE("buffer_reference_counting3") {
	Buffer b0(64);
	static_cast<uint8_t*>(b0->data())[0] = 0xAA;

	Buffer b1(b0);
	Buffer b2(std::move(b1));
	Buffer b3 = b2;
	Buffer b4 = std::move(b3);

	CHECK(b4->data() == b0->data());
	CHECK(b4->size() == 64);
	CHECK(static_cast<uint8_t*>(b4->data())[0] == 0xAA);
}

// # ============================================================

TEST_CASE("buffer_const") {
	Buffer b(32);
	static_cast<uint8_t*>(b->data())[0] = 0x1A;

	const Buffer& cb = b;
	CHECK(cb->size() == 32);
	CHECK(cb->data() != nullptr);
	CHECK(static_cast<const uint8_t*>(cb->data())[0] == 0x1A);
}
