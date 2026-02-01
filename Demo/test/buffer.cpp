#define DOCTEST_CONFIG_COLORS
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <Tenzor.hpp>
#include <doctest/doctest.h>
#include <random>
#include <thread>
#include <vector>

static std::mt19937 rng(123455);

TEST_CASE("BufferIMPL_basic_allocation") {
	int n = 300;

	for (int i = 0; i < n; i++) {
		int sz = rng() % (1024 * 4) + 1;
		TZ::mem::BufferIMPL buf(sz, 64);

		CHECK(buf.size() == sz);
		CHECK(buf.data() != nullptr);
		CHECK(reinterpret_cast<uintptr_t>(buf.data()) % 64 == 0);

		static_cast<uint8_t*>(buf.data())[sz - 1] = 0xAB;
	}
}
