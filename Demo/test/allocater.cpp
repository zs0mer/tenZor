#define DOCTEST_CONFIG_COLORS
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <Tenzor.hpp>
#include <doctest/doctest.h>

TEST_CASE("allocater constructor/destruction") {
	int n = 17;
	for (int i = 0; i < n; i++) {
		TZ::mem::salloc s(std::pow(2, i));
	}
}

TEST_CASE("allocating") {
	TZ::mem::salloc s(1024 * 1024); // 1MB
	int n = 1000;
	for (int i = 0; i < n; i++) {
		int p = rand() % 1024;
		uint8_t* a = static_cast<uint8_t*>(s.allocate(p));
		a[0] = 255;
		s.deallocate((void*&)a, p);
	}
}