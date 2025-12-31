#define DOCTEST_CONFIG_COLORS
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <Tenzor.hpp>
#include <doctest/doctest.h>


TEST_CASE("allocater constructor/destruction") {
	TZ::mem::salloc* a = new TZ::mem::salloc(1024 * 1024 * 10); // 10MB
	TZ::mem::salloc& s = *a;
	int n = 17;
	for (int i = 0; i < n; i++) {
		TZ::mem::salloc l(std::pow(2, i));
	}
}

TEST_CASE("allocating_little1") {
	TZ::mem::salloc* a = new TZ::mem::salloc(1024 * 1024 * 10); // 10MB
	TZ::mem::salloc& s = *a;
	int n = 10000;
	for (int i = 0; i < n; i++) {
		int p = (rand() % 1024) + 1;
		uint8_t* a = static_cast<uint8_t*>(s.allocate(p));
		*(a + p - 1) = 255;
		s.deallocate((void*&)a, p);
	}
}

TEST_CASE("allocating_little2") {
	TZ::mem::salloc* a = new TZ::mem::salloc(1024 * 1024 * 10); // 10MB
	TZ::mem::salloc& s = *a;
	int n = 10000;
	std::vector<uint8_t*> v(n);
	std::vector<int> sizee(n);
	for (int i = 0; i < n; i++) {
		int p = rand() % 1024 + 1;
		v[i] = static_cast<uint8_t*>(s.allocate(p));
		*(v[i] + p - 1) = 255;
		sizee[i] = p;
	}
	for (int i = 0; i < n; i++) {
		s.deallocate((void*&)v[i], sizee[i]);
	}
}