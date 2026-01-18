#define DOCTEST_CONFIG_COLORS
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <Tenzor.hpp>
#include <doctest/doctest.h>
#include <random>
#include <thread>
#include <vector>

static std::mt19937 rng(123456);


TEST_CASE("alloc_little1") {
	TZ::mem::salloc& s = TZ::mem::salloc::instance();
	int n = 50000;
	for (int i = 0; i < n; i++) {
		int p = (rng() % (1024 * 4)) + 1;
		uint8_t* a = static_cast<uint8_t*>(s.allocate(p));
		*(a + p - 1) = 255;
		s.deallocate((void*&)a, p);
	}
}

TEST_CASE("alloc_little2") {
	TZ::mem::salloc& s = TZ::mem::salloc::instance();
	int n = 50000;
	std::vector<uint8_t*> v(n);
	std::vector<int> sizee(n);
	for (int i = 0; i < n; i++) {
		int p = (rng() % (1024 * 4)) + 1;
		v[i] = static_cast<uint8_t*>(s.allocate(p));
		*(v[i] + p - 1) = 255;
		sizee[i] = p;
	}
	for (int i = 0; i < n; i++) {
		s.deallocate((void*&)v[i], sizee[i]);
	}
}

TEST_CASE("alloc_little3") {
	TZ::mem::salloc& s = TZ::mem::salloc::instance();
	int k = 10;
	int n = 1000;
	std::vector<uint8_t*> v(n);
	std::vector<int> sizee(n);

	for (int j = 0; j < k; j++) {
		for (int i = 0; i < n; i++) {
			int p = (rng() % (1024 * 4)) + 1;
			v[i] = static_cast<uint8_t*>(s.allocate(p));
			*(v[i] + p - 1) = 255;
			sizee[i] = p;
		}
		for (int i = 0; i < n; i++) {
			s.deallocate((void*&)v[i], sizee[i]);
		}
	}
}

TEST_CASE("alloc_little_multy") {
	auto& s = TZ::mem::salloc::instance();

	constexpr int threadCount = 8;
	constexpr int iterations = 10000;

	auto worker = [&]() {
		int k = 10;
		int n = 1000;
		std::vector<uint8_t*> v(n);
		std::vector<int> sizee(n);

		for (int j = 0; j < k; j++) {
			for (int i = 0; i < n; i++) {
				int p = (rng() % (1024 * 4)) + 1;
				v[i] = static_cast<uint8_t*>(s.allocate(p));
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


//& ============================================================

/*
TEST_CASE("alloc_middle1") {
    TZ::mem::salloc& s = TZ::mem::salloc::instance();
    int n = 10000;
    for (int i = 0; i < n; i++) {
        int p = 1024 * 4 + 1; // (rng() % (1024 * 1024)) + 1024 * 4 + 1;  685810 + 64
        uint8_t* a = static_cast<uint8_t*>(s.allocate(p));
        *(a) = 255; // + p - 1
        s.deallocate((void*&)a, p);
    }
}

TEST_CASE("alloc_middle2") {
    TZ::mem::salloc& s = TZ::mem::salloc::instance();
    int n = 10000;
    std::vector<uint8_t*> v(n);
    std::vector<int> sizee(n);
    for (int i = 0; i < n; i++) {
        int p = (rng() % (1024 * 1024)) + 1024 * 4 + 1;
        v[i] = static_cast<uint8_t*>(s.allocate(p));
        *(v[i]) = 255; //  + p - 1
        sizee[i] = p;
    }
    for (int i = 0; i < n; i++) {
        s.deallocate((void*&)v[i], sizee[i]);
    }
}

TEST_CASE("alloc_middle3") {
    TZ::mem::salloc& s = TZ::mem::salloc::instance();
    int k = 100;
    int n = 100;
    std::vector<uint8_t*> v(n);
    std::vector<int> sizee(n);

    for (int j = 0; j < k; j++) {
        for (int i = 0; i < n; i++) {
            int p = (rng() % (1024 * 1024)) + 1024 * 4 + 1;
            v[i] = static_cast<uint8_t*>(s.allocate(p));
            *(v[i]) = 255; //  + p - 1
            sizee[i] = p;
        }
        for (int i = 0; i < n; i++) {
            s.deallocate((void*&)v[i], sizee[i]);
        }
    }
}

TEST_CASE("alloc_middle_multy") {
    auto& s = TZ::mem::salloc::instance();

    constexpr int threadCount = 8;
    constexpr int iterations = 10000;

    auto worker = [&]() {
        int k = 100;
        int n = 100;
        std::vector<uint8_t*> v(n);
        std::vector<int> sizee(n);

        for (int j = 0; j < k; j++) {
            for (int i = 0; i < n; i++) {
                int p = (rng() % (1024 * 1024)) + 1024 * 4 + 1;
                v[i] = static_cast<uint8_t*>(s.allocate(p));
                *(v[i]) = 255; //  + p - 1
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

*/
//& ============================================================


TEST_CASE("alloc_large1") {
	TZ::mem::salloc& s = TZ::mem::salloc::instance();
	int n = 50000;
	for (int i = 0; i < n; i++) {
		int p = (rng() % (1024 * 1024 * 500)) + 1024 * 1024 + 1;
		uint8_t* a = static_cast<uint8_t*>(s.allocate(p));
		*(a + p - 1) = 255;
		s.deallocate((void*&)a, p);
	}
}

TEST_CASE("alloc_large2") {
	TZ::mem::salloc& s = TZ::mem::salloc::instance();
	int n = 10;
	std::vector<uint8_t*> v(n);
	std::vector<int> sizee(n);
	for (int i = 0; i < n; i++) {
		int p = (rng() % (1024 * 1024 * 500)) + 1024 * 1024 + 1;
		v[i] = static_cast<uint8_t*>(s.allocate(p));
		*(v[i] + p - 1) = 255;
		sizee[i] = p;
	}
	for (int i = 0; i < n; i++) {
		s.deallocate((void*&)v[i], sizee[i]);
	}
}

TEST_CASE("alloc_large3") {
	TZ::mem::salloc& s = TZ::mem::salloc::instance();
	int k = 1000;
	int n = 10;
	std::vector<uint8_t*> v(n);
	std::vector<int> sizee(n);

	for (int j = 0; j < k; j++) {
		for (int i = 0; i < n; i++) {
			int p = (rng() % (1024 * 1024 * 500)) + 1024 * 1024 + 1;
			v[i] = static_cast<uint8_t*>(s.allocate(p));
			*(v[i] + p - 1) = 255;
			sizee[i] = p;
		}
		for (int i = 0; i < n; i++) {
			s.deallocate((void*&)v[i], sizee[i]);
		}
	}
}

TEST_CASE("alloc_large_multy") {
	auto& s = TZ::mem::salloc::instance();

	constexpr int threadCount = 8;
	constexpr int iterations = 10000;

	auto worker = [&]() {
		int k = 10;
		int n = 100;
		std::vector<uint8_t*> v(n);
		std::vector<int> sizee(n);

		for (int j = 0; j < k; j++) {
			for (int i = 0; i < n; i++) {
				int p = (rng() % (1024 * 1024 * 500)) + 1024 * 1024 + 1;
				v[i] = static_cast<uint8_t*>(s.allocate(p));
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


//& ============================================================


TEST_CASE("alloc_zero") {
	auto& s = TZ::mem::salloc::instance();

	void* p = s.allocate(0);
	CHECK(p == nullptr);
}

TEST_CASE("alloc_alignment") {
	auto& s = TZ::mem::salloc::instance();
	int n = 100;

	for (int i = 0; i < n; i++) {
		int u = rng() % (8 * 1024) + 64;
		u = u - u % 64;
		for (size_t align : {8, 16, 32, 64}) {
			void* p = s.allocate(u, align);
			CHECK(p != nullptr);
			CHECK(reinterpret_cast<uintptr_t>(p) % align == 0);
			s.deallocate(p, u);
		}
	}


	int u = 50 * 1024 * 1024;
	for (size_t align : {8, 16, 32, 64}) {
		void* p = s.allocate(u, align);
		CHECK(p != nullptr);
		CHECK(reinterpret_cast<uintptr_t>(p) % align == 0);
		s.deallocate(p, u);
	}
}

TEST_CASE("alloc_size_boundaries") {
	auto& s = TZ::mem::salloc::instance();

	std::vector<size_t> sizes = {
	    2,       4,       8,       16,       32,       64,       128,      256,      512,
	    1024,    2048,    4096,    2 - 1,    4 - 1,    8 - 1,    16 - 1,   32 - 1,   64 - 1,
	    128 - 1, 256 - 1, 512 - 1, 1024 - 1, 2048 - 1, 4096 - 1, 2 + 1,    4 + 1,    8 + 1,
	    16 + 1,  32 + 1,  64 + 1,  128 + 1,  256 + 1,  512 + 1,  1024 + 1, 2048 + 1, 4096 + 1};

	for (size_t sz : sizes) {
		void* p = s.allocate(sz);
		CHECK(p != nullptr);
		reinterpret_cast<uint8_t*>(p)[sz - 1] = 0xAA;
		s.deallocate(p, 64);
	}
}
