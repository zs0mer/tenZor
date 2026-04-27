#include <Tenzor.hpp>
#include "buffer.hpp"
#define DOCTEST_CONFIG_COLORS
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <cstring>
#include <vector>

using namespace TZ;
using namespace TZ::mem;

//& ============================================================
//& BufferIMPL accessors (tested via Buffer::operator->)
//& ============================================================

TEST_CASE("bufferimpl_size_accessor") {
	Buffer b(256);
	CHECK(b->size() == 256);
}

TEST_CASE("bufferimpl_data_accessor_non_null") {
	Buffer b(64);
	CHECK(b->data() != nullptr);
}

TEST_CASE("bufferimpl_data_accessor_null_on_zero") {
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

TEST_CASE("bufferimpl_device_is_cpu_for_default_alloc") {
	Buffer b(128);
	CHECK(b->device() == CPU);
}

TEST_CASE("bufferimpl_allocator_matches_default") {
	Buffer b(128);
	CHECK(b->allocator() == &defaultAllocator(CPU));
}

TEST_CASE("bufferimpl_allocator_matches_malloc") {
	Buffer b(512, &Malloc::instance());
	CHECK(b->allocator() == &Malloc::instance());
	CHECK(b->device() == CPU);
}

// Tests the BufferIMPL copy constructor indirectly through Buffer::clone().
// The copy constructor must allocate fresh memory and memcpy the bytes.
TEST_CASE("bufferimpl_copy_ctor_deep_copies_data") {
	constexpr uint64_t sz = 128;
	Buffer b1(sz);
	uint8_t* p = static_cast<uint8_t*>(b1->data());
	for (uint64_t i = 0; i < sz; ++i)
		p[i] = static_cast<uint8_t>(i);

	Buffer b2 = b1.clone(); // internally calls new BufferIMPL(*ptr_)

	CHECK(b2->data() != b1->data()); // different allocation
	CHECK(b2->size() == sz);
	uint8_t* p2 = static_cast<uint8_t*>(b2->data());
	for (uint64_t i = 0; i < sz; ++i)
		CHECK(p2[i] == p[i]);
}

//& ============================================================
//& Buffer construction
//& ============================================================

TEST_CASE("buffer_default_constructor") {
	Buffer b;
	CHECK(b->size() == 0);
	CHECK(b->data() == nullptr);
	CHECK(b->device() == CPU);
}

TEST_CASE("buffer_explicit_zero_size") {
	Buffer b(uint64_t(0));
	CHECK(b->size() == 0);
	CHECK(b->data() == nullptr);
}

TEST_CASE("buffer_nullptr_impl_becomes_empty") {
	// Passing nullptr for the BufferIMPL* should NOT crash.
	// It should fall back to constructing a zero-size BufferIMPL.
	Buffer b(nullptr);
	CHECK(b->size() == 0);
	CHECK(b->data() == nullptr);
}

TEST_CASE("buffer_size_constructor_readable_writable") {
	constexpr uint64_t sz = 1024;
	Buffer b(sz);
	CHECK(b->size() == sz);
	CHECK(b->data() != nullptr);

	uint8_t* p = static_cast<uint8_t*>(b->data());
	for (uint64_t i = 0; i < sz; ++i)
		p[i] = static_cast<uint8_t>(i % 256);
	for (uint64_t i = 0; i < sz; ++i)
		CHECK(p[i] == static_cast<uint8_t>(i % 256));
}

// Verify the last byte of allocations that cross the internal allocator
// size-class boundaries (small ≤4KB, medium ≤1MB, large >1MB).
TEST_CASE("buffer_boundary_sizes") {
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

//& ============================================================
//& Buffer copy semantics (shared reference counting)
//& ============================================================

TEST_CASE("buffer_copy_ctor_shares_impl_pointer") {
	Buffer b1(512);
	static_cast<uint8_t*>(b1->data())[0] = 42;

	Buffer b2(b1);
	CHECK(b2->data() == b1->data()); // same raw pointer
	CHECK(b2->size() == 512);
	CHECK(static_cast<uint8_t*>(b2->data())[0] == 42);
}

TEST_CASE("buffer_copy_ctor_write_through") {
	// A write through one handle must be visible through the other.
	Buffer b1(64);
	Buffer b2(b1);

	static_cast<uint8_t*>(b1->data())[3] = 0xAB;
	CHECK(static_cast<uint8_t*>(b2->data())[3] == 0xAB);

	static_cast<uint8_t*>(b2->data())[3] = 0xCD;
	CHECK(static_cast<uint8_t*>(b1->data())[3] == 0xCD);
}

TEST_CASE("buffer_copy_assignment_shares_impl_pointer") {
	Buffer b1(256);
	static_cast<uint8_t*>(b1->data())[10] = 99;

	Buffer b2(128); // different size/alloc
	b2 = b1;

	CHECK(b2->data() == b1->data());
	CHECK(b2->size() == 256);
	CHECK(static_cast<uint8_t*>(b2->data())[10] == 99);
}

TEST_CASE("buffer_copy_assignment_self_assign_safe") {
	Buffer b(64);
	static_cast<uint8_t*>(b->data())[0] = 77;
	void* orig = b->data();

	b = b; // must not crash or corrupt

	CHECK(b->data() == orig);
	CHECK(b->size() == 64);
	CHECK(static_cast<uint8_t*>(b->data())[0] == 77);
}

//& ============================================================
//& Buffer move semantics
//& ============================================================

TEST_CASE("buffer_move_ctor_transfers_ownership") {
	Buffer b1(128);
	void* orig = b1->data();

	Buffer b2(std::move(b1));
	CHECK(b2->size() == 128);
	CHECK(b2->data() == orig);
	// b1's ptr_ is now nullptr — destructor must handle it gracefully (no crash)
}

TEST_CASE("buffer_move_assignment_transfers_ownership") {
	Buffer b1(64);
	void* orig = b1->data();

	Buffer b2;
	b2 = std::move(b1);
	CHECK(b2->size() == 64);
	CHECK(b2->data() == orig);
}

TEST_CASE("buffer_move_assignment_drops_previous_impl") {
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

//& ============================================================
//& Buffer clone (deep copy)
//& ============================================================

TEST_CASE("buffer_clone_independent_memory") {
	constexpr uint64_t sz = 1024;
	Buffer b1(sz);
	uint8_t* p1 = static_cast<uint8_t*>(b1->data());
	for (uint64_t i = 0; i < sz; ++i)
		p1[i] = static_cast<uint8_t>((i * 7 + 3) % 256);

	Buffer b2 = b1.clone();

	CHECK(b2->data() != b1->data()); // different pointer
	CHECK(b2->size() == sz);
	uint8_t* p2 = static_cast<uint8_t*>(b2->data());
	for (uint64_t i = 0; i < sz; ++i)
		CHECK(p2[i] == p1[i]);

	// Mutating b2 must NOT affect b1
	p2[0] = ~p2[0];
	CHECK(p1[0] != p2[0]);
}

TEST_CASE("buffer_clone_of_empty_buffer") {
	Buffer b1;
	Buffer b2 = b1.clone();
	CHECK(b2->size() == 0);
	CHECK(b2->data() == nullptr);
}

//& ============================================================
//& Reference counting / lifetime
//& ============================================================

TEST_CASE("buffer_many_shared_refs_all_valid") {
	Buffer b0(2048);
	static_cast<uint8_t*>(b0->data())[0] = 0xDE;
	void* orig = b0->data();

	std::vector<Buffer> copies;
	for (int i = 0; i < 200; ++i)
		copies.push_back(b0);

	for (auto& c : copies) {
		CHECK(c->data() == orig);
		CHECK(c->size() == 2048);
		CHECK(static_cast<uint8_t*>(c->data())[0] == 0xDE);
	}

	copies.clear(); // destroys 200 refs; b0 must still be valid
	CHECK(b0->data() == orig);
	CHECK(static_cast<uint8_t*>(b0->data())[0] == 0xDE);
}

TEST_CASE("buffer_last_ref_destructor_no_crash") {
	Buffer* b1 = new Buffer(512);
	static_cast<uint8_t*>((*b1)->data())[0] = 123;
	void* ptr = (*b1)->data();

	Buffer* b2 = new Buffer(*b1); // second ref
	delete b1;                    // first ref gone; b2 keeps it alive

	CHECK((*b2)->data() == ptr);
	CHECK(static_cast<uint8_t*>((*b2)->data())[0] == 123);

	delete b2; // last ref — memory freed, no crash
}

TEST_CASE("buffer_chain_copy_and_move") {
	Buffer b0(64);
	static_cast<uint8_t*>(b0->data())[0] = 55;

	Buffer b1(b0);             // copy -> shares
	Buffer b2(std::move(b1));  // move -> b2 owns
	Buffer b3 = b2;            // copy -> shares
	Buffer b4 = std::move(b3); // move -> b4 owns

	CHECK(b4->data() == b0->data());
	CHECK(b4->size() == 64);
	CHECK(static_cast<uint8_t*>(b4->data())[0] == 55);
}

//& ============================================================
//& operator-> const correctness
//& ============================================================

TEST_CASE("buffer_arrow_op_const_overload") {
	Buffer b(32);
	static_cast<uint8_t*>(b->data())[0] = 7;

	const Buffer& cb = b;
	CHECK(cb->size() == 32);
	CHECK(cb->data() != nullptr);
	CHECK(static_cast<const uint8_t*>(cb->data())[0] == 7);
}
