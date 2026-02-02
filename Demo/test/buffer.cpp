#define DOCTEST_CONFIG_COLORS
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <Tenzor.hpp>
#include <doctest/doctest.h>
#include <random>
#include <thread>
#include <vector>

// TODO ==============================================
// TODO GO THROUG THIS, MAKE SURE EVERYTHING IS TESTED
// TODO ==============================================

static std::mt19937 rng(123455);

using namespace TZ::mem;

TEST_CASE("BufferIMPL_basic_allocation") {
	int n = 300;

	for (int i = 0; i < n; i++) {
		int sz = rng() % (1024 * 4) + 64;
		BufferIMPL buf(sz, 64);

		CHECK(buf.size() == sz);
		CHECK(buf.data() != nullptr);
		CHECK(reinterpret_cast<uintptr_t>(buf.data()) % 64 == 0);

		static_cast<uint8_t*>(buf.data())[sz - 1] = 0xAB;
	}
}


TEST_CASE("BufferIMPL_clone_deep_copy") {
	auto& alloc = TZ::mem::salloc::instance();

	BufferIMPL a(256, 32, &alloc);
	auto* pa = static_cast<uint8_t*>(a.data());
	for (int i = 0; i < 256; ++i)
		pa[i] = static_cast<uint8_t>(i);

	BufferIMPL b(a); // deep copy

	auto* pb = static_cast<uint8_t*>(b.data());

	CHECK(pa != pb); // different memory

	for (int i = 0; i < 256; ++i)
		CHECK(pa[i] == pb[i]);

	// mutate clone, original must not change
	pb[0] = 0xFF;
	CHECK(pa[0] != pb[0]);
}


TEST_CASE("BufferIMPL_refcount_basic") {
	auto& alloc = TZ::mem::salloc::instance();

	auto* impl = new BufferIMPL(128, 16, &alloc);

	impl->retain();
	impl->retain();

	CHECK(impl->release() == false);
	CHECK(impl->release() == false);
	CHECK(impl->release() == true); // last owner frees memory
}


TEST_CASE("BufferIMPL_refcount_multithreaded") {
	auto& alloc = TZ::mem::salloc::instance();

	auto* impl = new BufferIMPL(512, 64, &alloc);

	const int threads = 8;
	const int iters = 1000;

	auto worker = [&]() {
		for (int i = 0; i < iters; ++i) {
			impl->retain();
			impl->release();
		}
	};

	std::vector<std::thread> ts;
	for (int i = 0; i < threads; ++i)
		ts.emplace_back(worker);

	for (auto& t : ts)
		t.join();

	// final release
	CHECK(impl->release() == true);
}


//& ============================================================


TEST_CASE("Buffer_basic_lifetime") {
	Buffer b(256, 64);

	CHECK(b.operator->() != nullptr);
	CHECK(reinterpret_cast<uintptr_t>(b->data()) % 64 == 0);

	static_cast<uint8_t*>(b->data())[255] = 0xAA;
}


TEST_CASE("Buffer_copy_shares_storage") {
	Buffer a(128, 32);
	auto* p = static_cast<uint8_t*>(a->data());
	p[0] = 0x11;

	Buffer b = a; // retain

	CHECK(a->data() == b->data());

	static_cast<uint8_t*>(b->data())[0] = 0x22;
	CHECK(static_cast<uint8_t*>(a->data())[0] == 0x22);
}


TEST_CASE("Buffer_move_steals") {
	Buffer a(128, 32);
	void* p = a->data();

	Buffer b = std::move(a);

	CHECK(b->data() == p);
	CHECK(a.operator->() == nullptr);
}


TEST_CASE("Buffer_move_assignment") {
	Buffer a(128, 32);
	Buffer b(256, 32);

	void* pa = a->data();

	b = std::move(a);

	CHECK(b->data() == pa);
	CHECK(a.operator->() == nullptr);
}


TEST_CASE("Buffer_clone_deep_copy") {
	Buffer a(256, 64);
	auto* pa = static_cast<uint8_t*>(a->data());
	for (int i = 0; i < 256; ++i)
		pa[i] = static_cast<uint8_t>(i);

	Buffer b = a.clone();

	auto* pb = static_cast<uint8_t*>(b->data());

	CHECK(pa != pb);

	for (int i = 0; i < 256; ++i)
		CHECK(pa[i] == pb[i]);

	pb[0] = 0xFF;
	CHECK(pa[0] != pb[0]);
}


TEST_CASE("Buffer_copy_and_destroy_safe") {
	Buffer a(512, 64);
	{
		Buffer b = a;
		Buffer c = b;
		CHECK(a->data() == b->data());
		CHECK(b->data() == c->data());
	}
	// a must still be valid
	static_cast<uint8_t*>(a->data())[511] = 0xAA;
}


TEST_CASE("Buffer_multithreaded_copy") {
	Buffer base(1024, 64);

	const int threads = 8;
	const int iters = 1000;

	auto worker = [&]() {
		for (int i = 0; i < iters; ++i) {
			Buffer tmp = base;
			static_cast<uint8_t*>(tmp->data())[0] = 0x42;
		}
	};

	std::vector<std::thread> ts;
	for (int i = 0; i < threads; ++i)
		ts.emplace_back(worker);

	for (auto& t : ts)
		t.join();

	CHECK(static_cast<uint8_t*>(base->data())[0] == 0x42);
}