#include "tensor.hpp"
#define DOCTEST_CONFIG_COLORS
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "Tenzor.hpp"

using namespace TZ;
using namespace TZ::impl;

// # ============================================================

TEST_CASE("gpu_alloc") {
	void* ptr = cuda::allocGPU(1024);
	CHECK(ptr != nullptr);
	cuda::freeGPU(ptr, 1024);
}

TEST_CASE("gpu_alloc_zero") {
	CHECK_NOTHROW([] {
		void* ptr = cuda::allocGPU(0);
		cuda::freeGPU(ptr, 0);
	}());
}

TEST_CASE("gpu_transfer") {
	const int N = 16;
	int host_in[N], host_out[N];
	for (int i = 0; i < N; i++)
		host_in[i] = i * 3;

	void* dev = cuda::allocGPU(N * sizeof(int));
	cuda::copyToGPU(dev, host_in, N * sizeof(int));
	cuda::copyToCPU(host_out, dev, N * sizeof(int));
	cuda::freeGPU(dev, N * sizeof(int));

	for (int i = 0; i < N; i++)
		CHECK(host_out[i] == i * 3);
}

TEST_CASE("gpu_memcopy") {
	const int N = 32;
	int host_in[N], host_out[N];
	for (int i = 0; i < N; i++)
		host_in[i] = i + 7;

	void* src = cuda::allocGPU(N * sizeof(int));
	void* dst = cuda::allocGPU(N * sizeof(int));

	cuda::copyToGPU(src, host_in, N * sizeof(int));
	cuda::memCopyOnGPU(dst, src, N * sizeof(int));
	cuda::copyToCPU(host_out, dst, N * sizeof(int));

	cuda::freeGPU(src, N * sizeof(int));
	cuda::freeGPU(dst, N * sizeof(int));

	for (int i = 0; i < N; i++)
		CHECK(host_out[i] == i + 7);
}

// # ============================================================

TEST_CASE("gpu_galloc") {
	mem::Galloc& g = mem::Galloc::instance();
	CHECK(g.device() == GPU);

	const int N = 16;
	int host_in[N], host_out[N];
	for (int i = 0; i < N; i++)
		host_in[i] = i * 3;

	void* dev = g.allocate(256, 64);
	cuda::copyToGPU(dev, host_in, N * sizeof(int));
	cuda::copyToCPU(host_out, dev, N * sizeof(int));
	g.deallocate(dev, 256);

	for (int i = 0; i < N; i++)
		CHECK(host_out[i] == i * 3);
}

// # ============================================================

TEST_CASE("gpu_buffer_construction") {
	mem::Buffer buf(64 * sizeof(float), &mem::Galloc::instance());
	CHECK(buf->device() == GPU);
	CHECK(buf->size() == 64 * sizeof(float));
	CHECK(buf->data() != nullptr);
}

TEST_CASE("gpu_buffer_clone") {
	const int N = 8;
	float host_in[N], host_out[N];
	for (int i = 0; i < N; i++)
		host_in[i] = (float)i * 1.5f;

	mem::Buffer buf(N * sizeof(float), &mem::Galloc::instance());
	cuda::copyToGPU(buf->data(), host_in, N * sizeof(float));

	mem::Buffer clone = buf.clone();
	CHECK(clone->data() != buf->data());

	cuda::copyToCPU(host_out, clone->data(), N * sizeof(float));
	for (int i = 0; i < N; i++)
		CHECK(host_out[i] == doctest::Approx((float)i * 1.5f));
}

// # ============================================================

TEST_CASE("gpu_tensor_construction") {
	TensorIMPL<float> t({4, 4}, GPU);
	CHECK(t.device() == GPU);
	CHECK(t.size() == 16);
	CHECK(t.dim() == 2);
	CHECK(t.dense());
}

TEST_CASE("gpu_tensor_copyTo") {
	const int N = 6;
	TensorIMPL<int> cpu_t({N}, CPU);
	for (int i = 0; i < N; i++)
		cpu_t.rawData()[i] = i * 10;

	TensorIMPL<int> gpu_t = cpu_t.copyTo(GPU);
	CHECK(gpu_t.device() == GPU);
	CHECK(gpu_t.size() == N);

	TensorIMPL<int> back = gpu_t.copyTo(CPU);
	CHECK(back.device() == CPU);
	for (int i = 0; i < N; i++)
		CHECK(back.rawData()[i] == i * 10);
}

TEST_CASE("gpu_tensor_copyTo_nondense") {
	TensorIMPL<int> base({2, 3}, CPU);
	for (int i = 0; i < 6; i++)
		base.rawData()[i] = i;

	uint64_t sh[] = {3, 2}, st[] = {1, 3};
	TensorIMPL<int> tr(2, sh, st, 0, base.buffer());
	CHECK(!tr.dense());

	TensorIMPL<int> gpu_t = tr.copyTo(GPU);
	CHECK(gpu_t.device() == GPU);
	CHECK(gpu_t.dense());

	TensorIMPL<int> back = gpu_t.copyTo(CPU);
	for (uint64_t r = 0; r < 3; r++)
		for (uint64_t c = 0; c < 2; c++)
			CHECK(back.at({r, c}) == tr.at({r, c}));
}

TEST_CASE("gpu_tensor_clone") {
	TensorIMPL<float> cpu_t({3, 3}, CPU);
	for (int i = 0; i < 9; i++)
		cpu_t.rawData()[i] = (float)i;

	TensorIMPL<float> gpu_t = cpu_t.copyTo(GPU);
	TensorIMPL<float> gpu_clone = gpu_t.clone();
	CHECK(gpu_clone.device() == GPU);
	CHECK(gpu_clone.rawData() != gpu_t.rawData());

	TensorIMPL<float> back = gpu_clone.copyTo(CPU);
	for (int i = 0; i < 9; i++)
		CHECK(back.rawData()[i] == doctest::Approx((float)i));
}

// # ============================================================

TEST_CASE("gpu_apply1_negate") {
	const int N = 8;
	TensorIMPL<int> cpu_t({N}, CPU);
	for (int i = 0; i < N; i++)
		cpu_t.rawData()[i] = i + 1;

	TensorIMPL<int> gpu_t = cpu_t.copyTo(GPU);
	TensorIMPL<int>::apply(gpu_t, Negate<int>{}, AnyDevice{});

	TensorIMPL<int> back = gpu_t.copyTo(CPU);
	for (int i = 0; i < N; i++)
		CHECK(back.rawData()[i] == -(i + 1));
}

TEST_CASE("gpu_apply1_set") {
	TensorIMPL<float> cpu_t({4, 4}, CPU);
	TensorIMPL<float> gpu_t = cpu_t.copyTo(GPU);
	TensorIMPL<float>::apply(gpu_t, Set<float>(3.14f), AnyDevice{});

	TensorIMPL<float> back = gpu_t.copyTo(CPU);
	for (int i = 0; i < 16; i++)
		CHECK(back.rawData()[i] == doctest::Approx(3.14f));
}

TEST_CASE("gpu_apply1_sum") {
	const int N = 100;
	TensorIMPL<int> cpu_t({N}, CPU);
	for (int i = 0; i < N; i++)
		cpu_t.rawData()[i] = 1;

	TensorIMPL<int> gpu_t = cpu_t.copyTo(GPU);

	TensorIMPL<int> cpu_acc(0, nullptr, CPU);
	cpu_acc.get() = 0;
	TensorIMPL<int> gpu_acc = cpu_acc.copyTo(GPU);

	TensorIMPL<int>::apply(gpu_t, Sum<int>(gpu_acc.data()), AnyDevice{});

	TensorIMPL<int> result = gpu_acc.copyTo(CPU);
	CHECK(result.get() == N);
}

TEST_CASE("gpu_apply1_nondense") {
	TensorIMPL<int> cpu_base({2, 3}, CPU);
	for (int i = 0; i < 6; i++)
		cpu_base.rawData()[i] = i + 1;

	TensorIMPL<int> gpu_base = cpu_base.copyTo(GPU);

	uint64_t sh[] = {3, 2}, st[] = {1, 3};
	TensorIMPL<int> gpu_tr(2, sh, st, 0, gpu_base.buffer());
	CHECK(!gpu_tr.dense());

	TensorIMPL<int>::apply(gpu_tr, Negate<int>{}, AnyDevice{});

	TensorIMPL<int> back_base = gpu_base.copyTo(CPU);
	for (int i = 0; i < 6; i++)
		CHECK(back_base.rawData()[i] == -(i + 1));
}

// # ============================================================

TEST_CASE("gpu_apply2_copy") {
	const int N = 10;
	TensorIMPL<int> src({N}, CPU);
	for (int i = 0; i < N; i++)
		src.rawData()[i] = i * 5;

	TensorIMPL<int> gpu_src = src.copyTo(GPU);
	TensorIMPL<int> gpu_dst({N}, GPU);
	TensorIMPL<int>::apply(gpu_src, gpu_dst, Copy<int>{}, AnyDevice{});

	TensorIMPL<int> back = gpu_dst.copyTo(CPU);
	for (int i = 0; i < N; i++)
		CHECK(back.rawData()[i] == i * 5);
}

TEST_CASE("gpu_apply2_add_scalar") {
	const int N = 5;
	TensorIMPL<float> src({N}, CPU);
	for (int i = 0; i < N; i++)
		src.rawData()[i] = (float)i;

	TensorIMPL<float> gpu_src = src.copyTo(GPU);
	TensorIMPL<float> gpu_dst({N}, GPU);
	TensorIMPL<float>::apply(gpu_src, gpu_dst, AddScalar<float>(10.f), AnyDevice{});

	TensorIMPL<float> back = gpu_dst.copyTo(CPU);
	for (int i = 0; i < N; i++)
		CHECK(back.rawData()[i] == doctest::Approx((float)i + 10.f));
}

TEST_CASE("gpu_apply2_multiply_scalar") {
	const int N = 4;
	TensorIMPL<double> src({N}, CPU);
	for (int i = 0; i < N; i++)
		src.rawData()[i] = (double)(i + 1);

	TensorIMPL<double> gpu_src = src.copyTo(GPU);
	TensorIMPL<double> gpu_dst({N}, GPU);
	TensorIMPL<double>::apply(gpu_src, gpu_dst, MultiplyScalar<double>(2.0), AnyDevice{});

	TensorIMPL<double> back = gpu_dst.copyTo(CPU);
	for (int i = 0; i < N; i++)
		CHECK(back.rawData()[i] == doctest::Approx((double)(i + 1) * 2.0));
}

TEST_CASE("gpu_apply2_broadcasted") {
	TensorIMPL<int> cpu_vec({3}, CPU);
	cpu_vec.rawData()[0] = 1;
	cpu_vec.rawData()[1] = 2;
	cpu_vec.rawData()[2] = 3;

	TensorIMPL<int> gpu_vec = cpu_vec.copyTo(GPU);

	uint64_t bsh[] = {4, 3}, bst[] = {0, 1};
	TensorIMPL<int> gpu_bcast(2, bsh, bst, 0, gpu_vec.buffer());
	CHECK(gpu_bcast.broadcasted());

	TensorIMPL<int> gpu_dst({4, 3}, GPU);
	TensorIMPL<int>::apply(gpu_bcast, gpu_dst, Copy<int>{}, AnyDevice{});

	TensorIMPL<int> back = gpu_dst.copyTo(CPU);
	for (uint64_t r = 0; r < 4; r++)
		for (uint64_t c = 0; c < 3; c++)
			CHECK(back.at({r, c}) == (int)(c + 1));
}

TEST_CASE("gpu_apply2_nondense") {
	TensorIMPL<int> cpu_src({2, 3}, CPU);
	for (int i = 0; i < 6; i++)
		cpu_src.rawData()[i] = i;

	TensorIMPL<int> gpu_src = cpu_src.copyTo(GPU);

	uint64_t sh[] = {3, 2}, st[] = {1, 3};
	TensorIMPL<int> gpu_tr(2, sh, st, 0, gpu_src.buffer());

	TensorIMPL<int> gpu_dst({3, 2}, GPU);
	TensorIMPL<int>::apply(gpu_tr, gpu_dst, Copy<int>{}, AnyDevice{});

	TensorIMPL<int> back = gpu_dst.copyTo(CPU);
	CHECK(back.at({0, 0}) == 0);
	CHECK(back.at({0, 1}) == 3);
	CHECK(back.at({1, 0}) == 1);
	CHECK(back.at({1, 1}) == 4);
	CHECK(back.at({2, 0}) == 2);
	CHECK(back.at({2, 1}) == 5);
}

// # ============================================================

TEST_CASE("gpu_apply3_add") {
	const int N = 8;
	TensorIMPL<int> a({N}, CPU), b({N}, CPU);
	for (int i = 0; i < N; i++) {
		a.rawData()[i] = i;
		b.rawData()[i] = i * 2;
	}

	TensorIMPL<int> ga = a.copyTo(GPU);
	TensorIMPL<int> gb = b.copyTo(GPU);
	TensorIMPL<int> gc({N}, GPU);
	TensorIMPL<int>::apply(ga, gb, gc, Add<int>{}, AnyDevice{});

	TensorIMPL<int> back = gc.copyTo(CPU);
	for (int i = 0; i < N; i++)
		CHECK(back.rawData()[i] == i + i * 2);
}

TEST_CASE("gpu_apply3_subtract") {
	const int N = 6;
	TensorIMPL<float> a({N}, CPU), b({N}, CPU);
	for (int i = 0; i < N; i++) {
		a.rawData()[i] = (float)(i + 10);
		b.rawData()[i] = (float)i;
	}

	TensorIMPL<float> ga = a.copyTo(GPU);
	TensorIMPL<float> gb = b.copyTo(GPU);
	TensorIMPL<float> gc({N}, GPU);
	TensorIMPL<float>::apply(ga, gb, gc, Subtract<float>{}, AnyDevice{});

	TensorIMPL<float> back = gc.copyTo(CPU);
	for (int i = 0; i < N; i++)
		CHECK(back.rawData()[i] == doctest::Approx(10.f));
}

TEST_CASE("gpu_apply3_nondense") {
	TensorIMPL<int> cpu_base({2, 2}, CPU);
	cpu_base.rawData()[0] = 1;
	cpu_base.rawData()[1] = 2;
	cpu_base.rawData()[2] = 3;
	cpu_base.rawData()[3] = 4;

	TensorIMPL<int> gpu_base = cpu_base.copyTo(GPU);

	uint64_t sh[] = {2, 2}, st[] = {1, 2};
	TensorIMPL<int> gpu_tr(2, sh, st, 0, gpu_base.buffer());

	TensorIMPL<int> b_cpu({2, 2}, CPU);
	TensorIMPL<int>::apply(b_cpu, Set<int>{10}, AnyDevice{});
	TensorIMPL<int> gb = b_cpu.copyTo(GPU);
	TensorIMPL<int> gc({2, 2}, GPU);

	TensorIMPL<int>::apply(gpu_tr, gb, gc, Add<int>{}, AnyDevice{});

	TensorIMPL<int> back = gc.copyTo(CPU);
	CHECK(back.at({0, 0}) == 11);
	CHECK(back.at({0, 1}) == 13);
	CHECK(back.at({1, 0}) == 12);
	CHECK(back.at({1, 1}) == 14);
}

// # ============================================================

TEST_CASE("gpu_dot1") {
	Vector<int> a = {1, 2, 3};
	Vector<int> b = {4, 5, 6};
	Vector<int> ga = a.copyTo(GPU);
	Vector<int> gb = b.copyTo(GPU);
	Scalar<int> r = dot(ga, gb);
	// result comes back on GPU — bring to CPU
	Scalar<int> r_cpu = r.copyTo(CPU);
	CHECK(r_cpu.get() == 32);
}

TEST_CASE("gpu_dot2") {
	Vector<float> a = {1.f, 2.f, 3.f, 4.f};
	Vector<float> b = {4.f, 3.f, 2.f, 1.f};
	Vector<float> ga = a.copyTo(GPU);
	Vector<float> gb = b.copyTo(GPU);
	Scalar<float> r = dot(ga, gb);
	Scalar<float> r_cpu = r.copyTo(CPU);
	CHECK(r_cpu.get() == doctest::Approx(20.f));
}

TEST_CASE("gpu_dot3") {
	Vector<double> a = {1.0, 0.0, -1.0};
	Vector<double> b = {2.0, 3.0, 4.0};
	Vector<double> ga = a.copyTo(GPU);
	Vector<double> gb = b.copyTo(GPU);
	Scalar<double> r = dot(ga, gb);
	Scalar<double> r_cpu = r.copyTo(CPU);
	CHECK(r_cpu.get() == doctest::Approx(-2.0));
}

TEST_CASE("gpu_dot4") {
	const int N = 1024;
	Vector<float> a(N, CPU), b(N, CPU);
	a.setAll(1.f);
	b.setAll(1.f);
	Vector<float> ga = a.copyTo(GPU);
	Vector<float> gb = b.copyTo(GPU);
	Scalar<float> r = dot(ga, gb);
	Scalar<float> r_cpu = r.copyTo(CPU);
	CHECK(r_cpu.get() == doctest::Approx((float)N));
}

TEST_CASE("gpu_dot_zero") {
	Vector<float> a(8, GPU), b(8, GPU);
	a.setAll(0.f);
	b.setAll(0.f);
	Scalar<float> r = dot(a, b);
	Scalar<float> r_cpu = r.copyTo(CPU);
	CHECK(r_cpu.get() == doctest::Approx(0.f));
}

TEST_CASE("gpu_dot_error") {
	Vector<float> a(3, GPU), b(4, GPU);
	CHECK_THROWS_AS(dot(a, b), std::runtime_error);
}

TEST_CASE("gpu_dot_error") {
	Vector<float> a(3, CPU), b(3, GPU);
	CHECK_THROWS_AS(dot(a, b), std::runtime_error);
}

// # ============================================================

TEST_CASE("gpu_matmul1") {
	Matrix<float> a = {{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}};
	Matrix<float> b = {{7.f, 8.f}, {9.f, 10.f}, {11.f, 12.f}};

	Matrix<float> ga = a.copyTo(GPU);
	Matrix<float> gb = b.copyTo(GPU);
	Matrix<float> gc = matmul(ga, gb);
	Matrix<float> back = gc.copyTo(CPU);

	CHECK(back.rows() == 2);
	CHECK(back.cols() == 2);
	CHECK(back.at(0, 0) == doctest::Approx(58.f));
	CHECK(back.at(0, 1) == doctest::Approx(64.f));
	CHECK(back.at(1, 0) == doctest::Approx(139.f));
	CHECK(back.at(1, 1) == doctest::Approx(154.f));
}

TEST_CASE("gpu_matmul2") {
	Matrix<double> a = {{1.0, 2.0}, {3.0, 4.0}};
	Matrix<double> b = {{5.0, 6.0}, {7.0, 8.0}};

	Matrix<double> ga = a.copyTo(GPU);
	Matrix<double> gb = b.copyTo(GPU);
	Matrix<double> gc = matmul(ga, gb);
	Matrix<double> back = gc.copyTo(CPU);

	CHECK(back.at(0, 0) == doctest::Approx(19.0));
	CHECK(back.at(0, 1) == doctest::Approx(22.0));
	CHECK(back.at(1, 0) == doctest::Approx(43.0));
	CHECK(back.at(1, 1) == doctest::Approx(50.0));
}

TEST_CASE("gpu_matmul3") {
	const int N = 4;
	Matrix<float> a(N, N, CPU);
	for (uint64_t i = 0; i < N; i++)
		for (uint64_t j = 0; j < N; j++)
			a.at(i, j) = (float)(i * N + j + 1);

	Matrix<float> id(N, N, CPU);
	id.setAll(0.f);
	for (int i = 0; i < N; i++)
		id.at(i, i) = 1.f;

	Matrix<float> ga = a.copyTo(GPU);
	Matrix<float> gid = id.copyTo(GPU);
	Matrix<float> gc = matmul(ga, gid);
	Matrix<float> back = gc.copyTo(CPU);

	for (uint64_t i = 0; i < N; i++)
		for (uint64_t j = 0; j < N; j++)
			CHECK(back.at(i, j) == doctest::Approx(a.at(i, j)));
}

TEST_CASE("gpu_matmul4") {
	Matrix<float> M = {{1.f, 2.f}, {3.f, 4.f}};
	Matrix<float> N = {{5.f, 6.f}, {7.f, 8.f}};
	Matrix<float> Mt = M.transpose();

	Matrix<float> gMt = Mt.copyTo(GPU);
	Matrix<float> gN = N.copyTo(GPU);
	Matrix<float> gc = matmul(gMt, gN);
	Matrix<float> back = gc.copyTo(CPU);

	CHECK(back.at(0, 0) == doctest::Approx(26.f));
	CHECK(back.at(0, 1) == doctest::Approx(30.f));
	CHECK(back.at(1, 0) == doctest::Approx(38.f));
	CHECK(back.at(1, 1) == doctest::Approx(44.f));
}

TEST_CASE("gpu_matmul_error1") {
	Matrix<float> a(2, 3, GPU);
	Matrix<float> b(4, 2, GPU);
	CHECK_THROWS_AS(matmul(a, b), std::runtime_error);
}

TEST_CASE("gpu_matmul_error2") {
	Matrix<float> a(2, 2, CPU);
	Matrix<float> b(2, 2, GPU);
	CHECK_THROWS_AS(matmul(a, b), std::runtime_error);
}

TEST_CASE("gpu_matmul_error3") {
	Matrix<int> a(2, 2, GPU);
	Matrix<int> b(2, 2, GPU);
	CHECK_THROWS_AS(matmul(a, b), std::runtime_error);
}
