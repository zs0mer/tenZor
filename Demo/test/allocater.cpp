#define DOCTEST_CONFIG_COLORS
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <Tenzor.hpp>
#include <doctest/doctest.h>


TEST_CASE("allocating_little1") {
	TZ::mem::salloc& s = TZ::mem::salloc::instance();
	int n = 50000;
	for (int i = 0; i < n; i++) {
		int p = (rand() % (1024 * 4)) + 1;
		uint8_t* a = static_cast<uint8_t*>(s.allocate(p));
		*(a + p - 1) = 255;
		s.deallocate((void*&)a, p);
	}
}

TEST_CASE("allocating_little2") {
	TZ::mem::salloc& s = TZ::mem::salloc::instance();
	int n = 50000;
	std::vector<uint8_t*> v(n);
	std::vector<int> sizee(n);
	for (int i = 0; i < n; i++) {
		int p = (rand() % (1024 * 4)) + 1;
		v[i] = static_cast<uint8_t*>(s.allocate(p));
		*(v[i] + p - 1) = 255;
		sizee[i] = p;
	}
	for (int i = 0; i < n; i++) {
		s.deallocate((void*&)v[i], sizee[i]);
	}
}

TEST_CASE("allocating_little3") {
	TZ::mem::salloc& s = TZ::mem::salloc::instance();
	int k = 10;
	int n = 1000;
	std::vector<uint8_t*> v(n);
	std::vector<int> sizee(n);

	for (int j = 0; j < k; j++) {
		for (int i = 0; i < n; i++) {
			int p = (rand() % (1024 * 4)) + 1;
			v[i] = static_cast<uint8_t*>(s.allocate(p));
			*(v[i] + p - 1) = 255;
			sizee[i] = p;
		}
		for (int i = 0; i < n; i++) {
			s.deallocate((void*&)v[i], sizee[i]);
		}
	}
}


//& ============================================================


TEST_CASE("allocating_middle1") {
	TZ::mem::salloc& s = TZ::mem::salloc::instance();
	int n = 10000;
	for (int i = 0; i < n; i++) {
		int p = (rand() % (1024 * 1024)) + 1024 * 4 + 1;
		uint8_t* a = static_cast<uint8_t*>(s.allocate(p));
		*(a + p - 1) = 255;
		s.deallocate((void*&)a, p);
	}
}

TEST_CASE("allocating_middle2") {
	TZ::mem::salloc& s = TZ::mem::salloc::instance();
	int n = 10000;
	std::vector<uint8_t*> v(n);
	std::vector<int> sizee(n);
	for (int i = 0; i < n; i++) {
		int p = (rand() % (1024 * 1024)) + 1024 * 4 + 1;
		v[i] = static_cast<uint8_t*>(s.allocate(p));
		*(v[i] + p - 1) = 255;
		sizee[i] = p;
	}
	for (int i = 0; i < n; i++) {
		s.deallocate((void*&)v[i], sizee[i]);
	}
}

TEST_CASE("allocating_middle3") {
	TZ::mem::salloc& s = TZ::mem::salloc::instance();
	int k = 100;
	int n = 100;
	std::vector<uint8_t*> v(n);
	std::vector<int> sizee(n);

	for (int j = 0; j < k; j++) {
		for (int i = 0; i < n; i++) {
			int p = (rand() % (1024 * 1024)) + 1024 * 4 + 1;
			v[i] = static_cast<uint8_t*>(s.allocate(p));
			*(v[i] + p - 1) = 255;
			sizee[i] = p;
		}
		for (int i = 0; i < n; i++) {
			s.deallocate((void*&)v[i], sizee[i]);
		}
	}
}


//& ============================================================


TEST_CASE("allocating_large1") {
	TZ::mem::salloc& s = TZ::mem::salloc::instance();
	int n = 50000;
	for (int i = 0; i < n; i++) {
		int p = (rand() % (1024 * 1024 * 500)) + 1024 * 1024 + 1;
		uint8_t* a = static_cast<uint8_t*>(s.allocate(p));
		*(a + p - 1) = 255;
		s.deallocate((void*&)a, p);
	}
}

TEST_CASE("allocating_large2") {
	TZ::mem::salloc& s = TZ::mem::salloc::instance();
	int n = 10;
	std::vector<uint8_t*> v(n);
	std::vector<int> sizee(n);
	for (int i = 0; i < n; i++) {
		int p = (rand() % (1024 * 1024 * 500)) + 1024 * 1024 + 1;
		v[i] = static_cast<uint8_t*>(s.allocate(p));
		*(v[i] + p - 1) = 255;
		sizee[i] = p;
	}
	for (int i = 0; i < n; i++) {
		s.deallocate((void*&)v[i], sizee[i]);
	}
}

TEST_CASE("allocating_large3") {
	TZ::mem::salloc& s = TZ::mem::salloc::instance();
	int k = 1000;
	int n = 10;
	std::vector<uint8_t*> v(n);
	std::vector<int> sizee(n);

	for (int j = 0; j < k; j++) {
		for (int i = 0; i < n; i++) {
			int p = (rand() % (1024 * 1024 * 500)) + 1024 * 1024 + 1;
			v[i] = static_cast<uint8_t*>(s.allocate(p));
			*(v[i] + p - 1) = 255;
			sizee[i] = p;
		}
		for (int i = 0; i < n; i++) {
			s.deallocate((void*&)v[i], sizee[i]);
		}
	}
}