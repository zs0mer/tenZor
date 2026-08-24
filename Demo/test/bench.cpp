#define DOCTEST_CONFIG_COLORS
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "TenZor.hpp"

using namespace tz;
using namespace tz::mem;
using namespace tz::grad;

// # helpers ======================================================================

template <class T>
static void doNotOptimize(T const& value) {
	asm volatile("" : : "r,m"(value) : "memory");
}

template <class F>
static double timeIt(F&& f) {
	auto t0 = std::chrono::steady_clock::now();
	f();
	auto t1 = std::chrono::steady_clock::now();
	return std::chrono::duration<double>(t1 - t0).count();
}

static void report(const std::string name, double secs, double iters, double flops = 0) {
	double ns = secs / iters * 1e9;

	std::cout << name << "[" << std::fixed << std::setprecision(2);

	std::cout << ns << "] ns/op";

	if (flops > 0)
		std::cout << " - [" << flops * iters / secs / 1e9 << "] GFLOP/s";

	std::cout << "\n";
}

static std::mt19937 RNG(12345);

// # allocators ===================================================================

static void benchAllocator(const char* name, Allocator& a, uint64_t minSize, uint64_t maxSize,
                           uint64_t batch, uint64_t rounds) {
	std::uniform_int_distribution<uint64_t> dist(minSize, maxSize);
	std::vector<uint64_t> sizes(batch);
	for (auto& s : sizes)
		s = dist(RNG);

	std::vector<void*> ptrs(batch);

	// warmup
	for (uint64_t i = 0; i < batch; i++)
		ptrs[i] = a.allocate(sizes[i], DEFAULT_ALIGNMENT);
	for (uint64_t i = 0; i < batch; i++)
		a.deallocate(ptrs[i], sizes[i]);

	double secs = timeIt([&] {
		for (uint64_t r = 0; r < rounds; r++) {
			for (uint64_t i = 0; i < batch; i++) {
				ptrs[i] = a.allocate(sizes[i], DEFAULT_ALIGNMENT);
				static_cast<uint8_t*>(ptrs[i])[sizes[i] - 1] = 88;
			}
			for (uint64_t i = 0; i < batch; i++)
				a.deallocate(ptrs[i], sizes[i]);
			doNotOptimize(ptrs);
		}
	});

	report(name, secs, double(rounds) * batch);
}

TEST_CASE("alloc_small") {
	benchAllocator("small Salloc (1B-4KB)", Salloc::instance(), 1, 4096, 256, 2000);
	benchAllocator("small Malloc (1B-4KB)", Malloc::instance(), 1, 4096, 256, 2000);
}

TEST_CASE("alloc_medium") {
	benchAllocator("medium Salloc (8KB-512KB)", Salloc::instance(), 8 * 1024, 512 * 1024, 32, 300);
	benchAllocator("medium Malloc (8KB-512KB)", Malloc::instance(), 8 * 1024, 512 * 1024, 32, 300);
}

TEST_CASE("alloc_large") {
	benchAllocator("large Salloc (2MB-16MB)", Salloc::instance(), 2 << 20, 16 << 20, 8, 100);
	benchAllocator("large Malloc (2MB-16MB)", Malloc::instance(), 2 << 20, 16 << 20, 8, 100);
}

TEST_CASE("tensor_churn") {
	double secs = timeIt([&] {
		for (int i = 0; i < 100000; i++) {
			Vector<float> v(256, CPU);
			v.at(255) = float(i);
			doNotOptimize(v.at(0));
		}
	});
	report("Vector<float>(256) create+destroy", secs, 100000);
}

// # elementwise ==================================================================

static constexpr uint64_t N = 1 << 24;

TEST_CASE("elementwise_cpu") {
	Vector<float> a(N, CPU), b(N, CPU);
	a.apply(impl::RrandomUniform<float>(-1.f, 1.f));
	b.apply(impl::RrandomUniform<float>(-1.f, 1.f));

	a += b; // warmup
	double secs = timeIt([&] {
		for (int i = 0; i < 20; i++)
			a += b;
		doNotOptimize(a.at(0));
	});

	report("CPU a += b (~16M floats, 64MB)", secs, 20);
}


TEST_CASE("elementwise_gpu") {
	Vector<float> a(N, GPU), b(N, GPU);
	a.apply(impl::RrandomUniform<float>(-1.f, 1.f), impl::AnyDevice{});
	b.apply(impl::RrandomUniform<float>(-1.f, 1.f), impl::AnyDevice{});

	a += b; // warmup
	double secs = timeIt([&] {
		for (int i = 0; i < 100; i++)
			a += b;
		auto r = a.copyTo(CPU);
		doNotOptimize(r.at(0));
	});
	report("GPU a += b (~16M floats, 64MB)", secs, 100);
}


// # matmul =======================================================================

static void benchMatmulCPU(uint64_t n, int iters) {
	Matrix<float> A(n, n, CPU), B(n, n, CPU);
	A.apply(impl::RrandomUniform<float>(-1.f, 1.f));
	B.apply(impl::RrandomUniform<float>(-1.f, 1.f));

	Matrix<float> C = matmul(A, B); // warmup
	double secs = timeIt([&] {
		for (int i = 0; i < iters; i++) {
			Matrix<float> D = matmul(A, B);
			doNotOptimize(D.at(0, 0));
		}
	});

	report("CPU matmul " + std::to_string(n) + "x" + std::to_string(n), secs, iters,
	       2.0 * n * n * n);
}


static void benchMatmulGPU(uint64_t n, int iters) {
	Matrix<float> A(n, n, GPU), B(n, n, GPU);
	A.apply(impl::RrandomUniform<float>(-1.f, 1.f), impl::AnyDevice{});
	B.apply(impl::RrandomUniform<float>(-1.f, 1.f), impl::AnyDevice{});

	Matrix<float> C = matmul(A, B); // warmup
	double secs = timeIt([&] {
		for (int i = 0; i < iters; i++) {
			Matrix<float> D = matmul(A, B);
			doNotOptimize(D);
		}
		auto r = C.copyTo(CPU);
		doNotOptimize(r.at(0, 0));
	});

	report("GPU matmul " + std::to_string(n) + "x" + std::to_string(n), secs, iters,
	       2.0 * n * n * n);
}


TEST_CASE("matmul") {
	benchMatmulCPU(256, 20);
	benchMatmulCPU(512, 20);

	benchMatmulGPU(256, 50);
	benchMatmulGPU(512, 50);
	benchMatmulGPU(1024, 20);
}

// # training step ================================================================

static GradVector<float> makeVec(std::initializer_list<float> vals, Device dev) {
	Vector<float> v(vals.size(), CPU);

	for (uint64_t i = 0; i < vals.size(); i++)
		v.at(i) = vals.begin()[i];
	return GradVector<float>(v.copyTo(dev));
}

static void benchTrainStep(const char* name, Device dev, int steps) {
	GradVector<float> x = makeVec({1.f, 0.f}, dev);
	GradVector<float> y = makeVec({1.f}, dev);

	Linear<float> L1(2, 8, dev), L2(8, 8, dev), L3(8, 1, dev);
	SGD<float> opt({}, 0.5f);
	opt.add(L1.parameters());
	opt.add(L2.parameters());
	opt.add(L3.parameters());

	auto step = [&] {
		auto z1 = L1(x);
		auto h1 = fn::sigmoid<float, GradVector<float>>(z1);
		auto z2 = L2(h1);
		auto h2 = fn::sigmoid<float, GradVector<float>>(z2);
		auto z3 = L3(h2);
		auto out = fn::sigmoid<float, GradVector<float>>(z3);
		auto diff = fn::subtract<float, GradVector<float>>(out, y);
		auto sq = fn::multiply<float, GradVector<float>>(diff, diff);
		auto loss = fn::sum<float>(sq);
		loss.backward();
		opt.step();
	};

	step(); // warmup
	double secs = timeIt([&] {
		for (int i = 0; i < steps; i++)
			step();
	});
	report(name, secs, steps);
}

TEST_CASE("train_step") {
	benchTrainStep("SGD train step CPU (2-8-8-1 nn)", CPU, 500);

	benchTrainStep("SGD train step GPU (2-8-8-1 nn)", GPU, 500);
}
