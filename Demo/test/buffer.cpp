#define DOCTEST_CONFIG_COLORS
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <Tenzor.hpp>
#include <doctest/doctest.h>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

static std::mt19937 rng(123455);

// ============================================================
// BufferIMPL tests
// ============================================================

TEST_CASE("bufferimpl_basic_construction") {
	const uint64_t size = 256;
	TZ::mem::BufferIMPL buf(size);

	CHECK(buf.size() == size);
	CHECK(buf.data() != nullptr);
	CHECK(buf.device() == TZ::CPU);
}

TEST_CASE("bufferimpl_write_and_read") {
	const uint64_t size = 128;
	TZ::mem::BufferIMPL buf(size);

	uint8_t* ptr = static_cast<uint8_t*>(buf.data());
	for (uint64_t i = 0; i < size; ++i)
		ptr[i] = static_cast<uint8_t>(i % 256);

	for (uint64_t i = 0; i < size; ++i)
		CHECK(ptr[i] == static_cast<uint8_t>(i % 256));
}

TEST_CASE("bufferimpl_copy_constructor_clones_data") {
	const uint64_t size = 64;
	TZ::mem::BufferIMPL original(size);

	uint8_t* optr = static_cast<uint8_t*>(original.data());
	for (uint64_t i = 0; i < size; ++i)
		optr[i] = static_cast<uint8_t>(i + 1);

	TZ::mem::BufferIMPL copy(original);

	CHECK(copy.size() == original.size());
	CHECK(copy.data() != original.data());

	const uint8_t* cptr = static_cast<const uint8_t*>(copy.data());
	for (uint64_t i = 0; i < size; ++i)
		CHECK(cptr[i] == static_cast<uint8_t>(i + 1));
}

TEST_CASE("bufferimpl_copy_is_independent") {
	const uint64_t size = 32;
	TZ::mem::BufferIMPL original(size);

	uint8_t* optr = static_cast<uint8_t*>(original.data());
	memset(optr, 0xAA, size);

	TZ::mem::BufferIMPL copy(original);
	uint8_t* cptr = static_cast<uint8_t*>(copy.data());

	memset(cptr, 0xFF, size);

	for (uint64_t i = 0; i < size; ++i)
		CHECK(optr[i] == 0xAA);
}

TEST_CASE("bufferimpl_custom_allocator") {
	const uint64_t size = 512;
	TZ::mem::BufferIMPL buf(size, TZ::mem::DEFAULT_ALIGNMENT, &TZ::mem::Malloc::instance());

	CHECK(buf.size() == size);
	CHECK(buf.data() != nullptr);
	CHECK(buf.allocator() == &TZ::mem::Malloc::instance());
}

TEST_CASE("bufferimpl_retain_and_release") {
	const uint64_t size = 64;
	TZ::mem::BufferIMPL* buf = new TZ::mem::BufferIMPL(size);
}

// ============================================================
// Buffer tests
// ============================================================

TEST_CASE("buffer_default_construction") {
	TZ::mem::Buffer buf;

	CHECK(buf->size() == 0);
	CHECK(buf->device() == TZ::CPU);
}

TEST_CASE("buffer_sized_construction") {
	const uint64_t size = 1024;
	TZ::mem::Buffer buf(size);

	CHECK(buf->size() == size);
	CHECK(buf->data() != nullptr);
}

TEST_CASE("buffer_write_and_read") {
	const uint64_t size = 256;
	TZ::mem::Buffer buf(size);

	uint8_t* ptr = static_cast<uint8_t*>(buf->data());
	for (uint64_t i = 0; i < size; ++i)
		ptr[i] = static_cast<uint8_t>(i);

	const uint8_t* cptr = static_cast<const uint8_t*>(
	    static_cast<const TZ::mem::BufferIMPL*>(buf.operator->())->data());
	for (uint64_t i = 0; i < size; ++i)
		CHECK(cptr[i] == static_cast<uint8_t>(i));
}

TEST_CASE("buffer_copy_shares_impl") {
	const uint64_t size = 128;
	TZ::mem::Buffer a(size);

	uint8_t* ptr = static_cast<uint8_t*>(a->data());
	memset(ptr, 0x5A, size);

	TZ::mem::Buffer b(a);

	CHECK(b->data() == a->data());
	CHECK(b->size() == a->size());
}

TEST_CASE("buffer_copy_assignment_shares_impl") {
	const uint64_t size = 128;
	TZ::mem::Buffer a(size);
	TZ::mem::Buffer b(64);

	b = a;

	CHECK(b->data() == a->data());
	CHECK(b->size() == a->size());
}

TEST_CASE("buffer_self_assignment") {
	const uint64_t size = 64;
	TZ::mem::Buffer a(size);
	void* original_data = a->data();

	a = a;

	CHECK(a->data() == original_data);
	CHECK(a->size() == size);
}

TEST_CASE("buffer_move_constructor_transfers_ownership") {
	const uint64_t size = 256;
	TZ::mem::Buffer a(size);
	void* original_data = a->data();

	TZ::mem::Buffer b(std::move(a));

	CHECK(b->data() == original_data);
	CHECK(b->size() == size);
}

TEST_CASE("buffer_move_assignment_transfers_ownership") {
	const uint64_t size = 256;
	TZ::mem::Buffer a(size);
	void* original_data = a->data();

	TZ::mem::Buffer b;
	b = std::move(a);

	CHECK(b->data() == original_data);
	CHECK(b->size() == size);
}

TEST_CASE("buffer_move_self_assignment") {
	const uint64_t size = 64;
	TZ::mem::Buffer a(size);
	void* original_data = a->data();

	a = std::move(a);

	(void)original_data;
}

TEST_CASE("buffer_clone_is_deep_copy") {
	const uint64_t size = 128;
	TZ::mem::Buffer a(size);

	uint8_t* ptr = static_cast<uint8_t*>(a->data());
	memset(ptr, 0xBB, size);

	TZ::mem::Buffer b = a.clone();

	CHECK(b->data() != a->data());
	CHECK(b->size() == size);

	const uint8_t* cptr = static_cast<const uint8_t*>(b->data());
	for (uint64_t i = 0; i < size; ++i)
		CHECK(cptr[i] == 0xBB);
}

TEST_CASE("buffer_clone_independence") {
	const uint64_t size = 64;
	TZ::mem::Buffer a(size);

	uint8_t* aptr = static_cast<uint8_t*>(a->data());
	memset(aptr, 0x11, size);

	TZ::mem::Buffer b = a.clone();
	uint8_t* bptr = static_cast<uint8_t*>(b->data());
	memset(bptr, 0x22, size);

	for (uint64_t i = 0; i < size; ++i)
		CHECK(aptr[i] == 0x11);
}

TEST_CASE("buffer_multiple_copies_share_impl") {
	const uint64_t size = 64;
	TZ::mem::Buffer a(size);
	void* original_data = a->data();

	{
		TZ::mem::Buffer b(a);
		TZ::mem::Buffer c(b);
		TZ::mem::Buffer d(c);

		CHECK(b->data() == original_data);
		CHECK(c->data() == original_data);
		CHECK(d->data() == original_data);
	}

	CHECK(a->data() == original_data);
	CHECK(a->size() == size);

	uint8_t* ptr = static_cast<uint8_t*>(a->data());
	ptr[size - 1] = 0xFF;
	CHECK(ptr[size - 1] == 0xFF);
}

TEST_CASE("buffer_random_sizes") {
	const int n = 200;
	for (int i = 0; i < n; ++i) {
		uint64_t size = (rng() % (4 * 1024)) + 1;
		TZ::mem::Buffer buf(size);

		CHECK(buf->size() == size);
		CHECK(buf->data() != nullptr);

		uint8_t* ptr = static_cast<uint8_t*>(buf->data());
		ptr[0] = 0xDE;
		ptr[size - 1] = 0xAD;

		CHECK(ptr[0] == 0xDE);
		CHECK(ptr[size - 1] == 0xAD);
	}
}

TEST_CASE("buffer_device_is_cpu") {
	TZ::mem::Buffer buf(128);
	CHECK(buf->device() == TZ::CPU);
}

TEST_CASE("buffer_custom_allocator") {
	const uint64_t size = 256;
	TZ::mem::Buffer buf(size, &TZ::mem::Malloc::instance());

	CHECK(buf->size() == size);
	CHECK(buf->data() != nullptr);
	CHECK(buf->allocator() == &TZ::mem::Malloc::instance());
}

TEST_CASE("buffer_concurrent_copies") {
	const uint64_t size = 256;
	TZ::mem::Buffer source(size);

	uint8_t* sptr = static_cast<uint8_t*>(source->data());
	memset(sptr, 0xCC, size);

	const int threadCount = 8;

	std::vector<TZ::mem::Buffer> copies(threadCount, TZ::mem::Buffer());
	std::vector<std::thread> threads;

	for (int i = 0; i < threadCount; ++i) {
		threads.emplace_back([&, i]() { copies[i] = source; });
	}

	for (auto& t : threads)
		t.join();

	for (int i = 0; i < threadCount; ++i) {
		CHECK(copies[i]->data() == source->data());
		CHECK(copies[i]->size() == size);
	}
}
