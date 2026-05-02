#define DOCTEST_CONFIG_COLORS
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <random>
#include <thread>
#include <vector>

#include "Tenzor.hpp"

using namespace tz;
using namespace tz::mem;

static std::mt19937 rng(123455);


TEST_CASE("alloc_little1") {
	Salloc& s = Salloc::instance();
	int n = 1000;
	for (int i = 0; i < n; i++) {
		int p = (rng() % (1024 * 4)) + 1;
		uint8_t* a = static_cast<uint8_t*>(s.allocate(p, DEFAULT_ALIGNMENT));
		*(a + p - 1) = 255;
		s.deallocate((void*&)a, p);
	}
}

TEST_CASE("alloc_little2") {
	Salloc& s = Salloc::instance();
	int n = 1000;
	std::vector<uint8_t*> v(n);
	std::vector<int> sizee(n);
	for (int i = 0; i < n; i++) {
		int p = (rng() % (1024 * 4)) + 1;
		v[i] = static_cast<uint8_t*>(s.allocate(p, DEFAULT_ALIGNMENT));
		*(v[i] + p - 1) = 255;
		sizee[i] = p;
	}
	for (int i = 0; i < n; i++) {
		s.deallocate((void*&)v[i], sizee[i]);
	}
}

TEST_CASE("alloc_little3") {
	Salloc& s = Salloc::instance();
	int k = 10;
	int n = 100;
	std::vector<uint8_t*> v(n);
	std::vector<int> sizee(n);

	for (int j = 0; j < k; j++) {
		for (int i = 0; i < n; i++) {
			int p = (rng() % (1024 * 4)) + 1;
			v[i] = static_cast<uint8_t*>(s.allocate(p, DEFAULT_ALIGNMENT));
			*(v[i] + p - 1) = 255;
			sizee[i] = p;
		}
		for (int i = 0; i < n; i++) {
			s.deallocate((void*&)v[i], sizee[i]);
		}
	}
}

TEST_CASE("alloc_little_multy1") {
	auto& s = Salloc::instance();

	int threadCount = 8;

	auto worker = [&]() {
		int k = 10;
		int n = 100;
		std::vector<uint8_t*> v(n);
		std::vector<int> sizee(n);

		for (int j = 0; j < k; j++) {
			for (int i = 0; i < n; i++) {
				int p = (rng() % (1024 * 4)) + 1;
				v[i] = static_cast<uint8_t*>(s.allocate(p, DEFAULT_ALIGNMENT));
				*(v[i] + p - 1) = 255;
				sizee[i] = p;
			}
			for (int i = 0; i < n; i++) {
				s.deallocate((void*&)v[i], sizee[i]);
			}
		}
	};

	std::vector<std::thread> threads;
	for (int i = 0; i < threadCount; ++i)
		threads.emplace_back(worker);

	for (auto& t : threads)
		t.join();
}

TEST_CASE("alloc_little_multy2") {
	Salloc& s = Salloc::instance();

	int k = 1;
	int n = 1;
	std::vector<void*> shared(n);
	std::vector<int> sizes(n);

	for (int i = 0; i < k; i++) {
		std::thread producer([&] {
			for (int i = 0; i < n; ++i) {
				int sz = (rng() % (1024 * 4)) + 1;
				shared[i] = s.allocate(sz, DEFAULT_ALIGNMENT);
				sizes[i] = sz;
				static_cast<uint8_t*>(shared[i])[sz - 1] = 0xAA;
			}
		});
		producer.join();

		std::thread consumer([&] {
			for (int i = 0; i < n; ++i) {
				s.deallocate(shared[i], sizes[i]);
			}
		});

		consumer.join();
	}
}

//& ============================================================

TEST_CASE("alloc_middle1") {
	Salloc& s = Salloc::instance();
	int n = 500;
	for (int i = 0; i < n; i++) {
		int p = (rng() % (1024 * 1024)) + 1024 * 4 + 1;
		uint8_t* a = static_cast<uint8_t*>(s.allocate(p, DEFAULT_ALIGNMENT));
		*(a + p - 1) = 255;
		s.deallocate((void*&)a, p);
	}
}

TEST_CASE("alloc_middle2") {
	Salloc& s = Salloc::instance();
	int n = 500;
	std::vector<uint8_t*> v(n);
	std::vector<int> sizee(n);
	for (int i = 0; i < n; i++) {
		int p = (rng() % (1024 * 1024)) + 1024 * 4 + 1;
		v[i] = static_cast<uint8_t*>(s.allocate(p, DEFAULT_ALIGNMENT));
		*(v[i] + p - 1) = 255;
		sizee[i] = p;
	}
	for (int i = 0; i < n; i++) {
		s.deallocate((void*&)v[i], sizee[i]);
	}
}

TEST_CASE("alloc_middle3") {
	Salloc& s = Salloc::instance();
	int k = 5;
	int n = 100;
	std::vector<uint8_t*> v(n);
	std::vector<int> sizee(n);

	for (int j = 0; j < k; j++) {
		for (int i = 0; i < n; i++) {
			int p = (rng() % (1024 * 1024)) + 1024 * 4 + 1;
			v[i] = static_cast<uint8_t*>(s.allocate(p, DEFAULT_ALIGNMENT));
			*(v[i] + p - 1) = 255;
			sizee[i] = p;
		}
		for (int i = 0; i < n; i++) {
			s.deallocate((void*&)v[i], sizee[i]);
		}
	}
}

TEST_CASE("alloc_middle_multy1") {
	auto& s = Salloc::instance();

	int threadCount = 8;

	auto worker = [&]() {
		int k = 5;
		int n = 100;
		std::vector<uint8_t*> v(n);
		std::vector<int> sizee(n);

		for (int j = 0; j < k; j++) {
			for (int i = 0; i < n; i++) {
				int p = (rng() % (1024 * 1024)) + 1024 * 4 + 1;
				v[i] = static_cast<uint8_t*>(s.allocate(p, DEFAULT_ALIGNMENT));
				*(v[i] + p - 1) = 255;
				sizee[i] = p;
			}
			for (int i = 0; i < n; i++) {
				s.deallocate((void*&)v[i], sizee[i]);
			}
		}
	};

	std::vector<std::thread> threads;
	for (int i = 0; i < threadCount; ++i)
		threads.emplace_back(worker);

	for (auto& t : threads)
		t.join();
}

TEST_CASE("alloc_middle_multy2") {
	Salloc& s = Salloc::instance();

	int k = 5;
	int n = 200;
	std::vector<void*> shared(n);
	std::vector<int> sizes(n);

	for (int i = 0; i < k; i++) {
		std::thread producer([&] {
			for (int i = 0; i < n; ++i) {
				int sz = (rng() % (1024 * 1024)) + 1024 * 4 + 1;
				shared[i] = s.allocate(sz, DEFAULT_ALIGNMENT);
				sizes[i] = sz;
				static_cast<uint8_t*>(shared[i])[sz - 1] = 0xAA;
			}
		});
		producer.join();

		std::thread consumer([&] {
			for (int i = 0; i < n; ++i) {
				s.deallocate(shared[i], sizes[i]);
			}
		});

		consumer.join();
	}
}

//& ============================================================

TEST_CASE("alloc_large1") {
	Salloc& s = Salloc::instance();
	int n = 40;
	for (int i = 0; i < n; i++) {
		int p = (rng() % (1024 * 1024 * 500)) + 1024 * 1024 + 1;
		uint8_t* a = static_cast<uint8_t*>(s.allocate(p, DEFAULT_ALIGNMENT));
		*(a + p - 1) = 255;
		s.deallocate((void*&)a, p);
	}
}

TEST_CASE("alloc_large2") {
	Salloc& s = Salloc::instance();
	int n = 40;
	std::vector<uint8_t*> v(n);
	std::vector<int> sizee(n);
	for (int i = 0; i < n; i++) {
		int p = (rng() % (1024 * 1024 * 500)) + 1024 * 1024 + 1;
		v[i] = static_cast<uint8_t*>(s.allocate(p, DEFAULT_ALIGNMENT));
		*(v[i] + p - 1) = 255;
		sizee[i] = p;
	}
	for (int i = 0; i < n; i++) {
		s.deallocate((void*&)v[i], sizee[i]);
	}
}

TEST_CASE("alloc_large3") {
	Salloc& s = Salloc::instance();
	int k = 5;
	int n = 10;
	std::vector<uint8_t*> v(n);
	std::vector<int> sizee(n);

	for (int j = 0; j < k; j++) {
		for (int i = 0; i < n; i++) {
			int p = (rng() % (1024 * 1024 * 500)) + 1024 * 1024 + 1;
			v[i] = static_cast<uint8_t*>(s.allocate(p, DEFAULT_ALIGNMENT));
			*(v[i] + p - 1) = 255;
			sizee[i] = p;
		}
		for (int i = 0; i < n; i++) {
			s.deallocate((void*&)v[i], sizee[i]);
		}
	}
}

TEST_CASE("alloc_large_multy1") {
	auto& s = Salloc::instance();

	int threadCount = 8;

	auto worker = [&]() {
		int k = 3;
		int n = 7;
		std::vector<uint8_t*> v(n);
		std::vector<int> sizee(n);

		for (int j = 0; j < k; j++) {
			for (int i = 0; i < n; i++) {
				int p = (rng() % (1024 * 1024 * 500)) + 1024 * 1024 + 1;
				v[i] = static_cast<uint8_t*>(s.allocate(p, DEFAULT_ALIGNMENT));
				*(v[i] + p - 1) = 255;
				sizee[i] = p;
			}
			for (int i = 0; i < n; i++) {
				s.deallocate((void*&)v[i], sizee[i]);
			}
		}
	};

	std::vector<std::thread> threads;
	for (int i = 0; i < threadCount; ++i)
		threads.emplace_back(worker);

	for (auto& t : threads)
		t.join();
}

TEST_CASE("alloc_large_multy2") {
	Salloc& s = Salloc::instance();

	int k = 5;
	int n = 40;
	std::vector<void*> shared(n);
	std::vector<int> sizes(n);

	for (int i = 0; i < k; i++) {
		std::thread producer([&] {
			for (int i = 0; i < n; ++i) {
				int sz = (rng() % (1024 * 1024 * 500)) + 1024 * 1024 + 1;
				shared[i] = s.allocate(sz, DEFAULT_ALIGNMENT);
				sizes[i] = sz;
				static_cast<uint8_t*>(shared[i])[sz - 1] = 0xAA;
			}
		});
		producer.join();

		std::thread consumer([&] {
			for (int i = 0; i < n; ++i) {
				s.deallocate(shared[i], sizes[i]);
			}
		});

		consumer.join();
	}
}

//& ============================================================

TEST_CASE("alloc_zero") {
	auto& s = Salloc::instance();

	void* p = s.allocate(0, DEFAULT_ALIGNMENT);
	CHECK(p == nullptr);
}

TEST_CASE("alloc_alignment") {
	auto& s = Salloc::instance();
	int n = 100;
	int m = 40;
	std::vector<int> alignments = {8, 16, 32, 64};

	for (int i = 0; i < n; i++) {
		int u = rng() % (8 * 1024) + 64;
		u = u - u % 64;
		for (size_t align : alignments) {
			void* p = s.allocate(u, align);
			CHECK(reinterpret_cast<uintptr_t>(p) % align == 0);
			s.deallocate(p, u);
		}
	}

	for (int i = 0; i < m; i++) {
		int u = (rng() % 50 + 1) * 1024 * 1024;
		for (size_t align : alignments) {
			void* p = s.allocate(u, align);
			CHECK(p != nullptr);
			CHECK(reinterpret_cast<uintptr_t>(p) % align == 0);
			s.deallocate(p, u);
		}
	}
}

TEST_CASE("alloc_size_boundaries") {
	auto& s = Salloc::instance();

	std::vector<size_t> sizes = {
	    2,       4,       8,       16,       32,       64,       128,      256,      512,
	    1024,    2048,    4096,    2 - 1,    4 - 1,    8 - 1,    16 - 1,   32 - 1,   64 - 1,
	    128 - 1, 256 - 1, 512 - 1, 1024 - 1, 2048 - 1, 4096 - 1, 2 + 1,    4 + 1,    8 + 1,
	    16 + 1,  32 + 1,  64 + 1,  128 + 1,  256 + 1,  512 + 1,  1024 + 1, 2048 + 1, 4096 + 1};

	for (size_t sz : sizes) {
		void* p = s.allocate(sz, DEFAULT_ALIGNMENT);
		reinterpret_cast<uint8_t*>(p)[sz - 1] = 0xAA;
		s.deallocate(p, sz);
	}
}
