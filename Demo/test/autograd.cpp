#include "math_classes.hpp"
#include <cstdint>
#define DOCTEST_CONFIG_COLORS
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "TenZor.hpp"

using namespace tz;
using namespace tz::grad;
using namespace tz::grad::fn;

// helpers
static float scalarGradVal(GradScalar<float>& s) {
	return Scalar<float>(s.grad().tensor_()).get();
}

// # ================================================================================

TEST_CASE("grad_add") {
	GradScalar<float> a(Scalar<float>(2.f)), b(Scalar<float>(3.f));
	auto z = add<float>(a, b);
	z.backward();
	CHECK(scalarGradVal(a) == doctest::Approx(1.f));
	CHECK(scalarGradVal(b) == doctest::Approx(1.f));
}

TEST_CASE("grad_subtract") {
	GradScalar<float> a(Scalar<float>(5.f)), b(Scalar<float>(3.f));
	auto z = subtract<float>(a, b);
	z.backward();
	CHECK(scalarGradVal(a) == doctest::Approx(1.f));
	CHECK(scalarGradVal(b) == doctest::Approx(-1.f));
}

TEST_CASE("grad_multiply") {
	GradScalar<float> a(Scalar<float>(3.f)), b(Scalar<float>(4.f));
	auto z = multiply<float>(a, b);
	z.backward();
	CHECK(scalarGradVal(a) == doctest::Approx(4.f));
	CHECK(scalarGradVal(b) == doctest::Approx(3.f));
}

TEST_CASE("grad_divide") {
	GradScalar<float> a(Scalar<float>(6.f)), b(Scalar<float>(2.f));
	auto z = divide<float>(a, b);
	z.backward();
	CHECK(scalarGradVal(a) == doctest::Approx(0.5f));
	CHECK(scalarGradVal(b) == doctest::Approx(-1.5f));
}

TEST_CASE("grad_negate") {
	GradScalar<float> a(Scalar<float>(3.f));
	auto z = negate<float>(a);
	z.backward();
	CHECK(scalarGradVal(a) == doctest::Approx(-1.f));
}

// # ================================================================================

TEST_CASE("grad_exp") {
	GradScalar<float> a(Scalar<float>(0.f));
	auto z = exp<float>(a);
	z.backward();
	CHECK(z.val().get() == doctest::Approx(1.f));
	CHECK(scalarGradVal(a) == doctest::Approx(1.f));
}

TEST_CASE("grad_exp_nonzero") {
	GradScalar<float> a(Scalar<float>(1.f));
	auto z = exp<float>(a);
	z.backward();
	CHECK(scalarGradVal(a) == doctest::Approx(std::exp(1.f)));
}

TEST_CASE("grad_log") {
	GradScalar<float> a(Scalar<float>(2.f));
	auto z = log<float>(a);
	z.backward();
	CHECK(scalarGradVal(a) == doctest::Approx(0.5f));
}

TEST_CASE("grad_pow") {
	GradScalar<float> a(Scalar<float>(2.f));
	auto z = pow<float>(a, Scalar<float>(3.f));
	z.backward();
	CHECK(scalarGradVal(a) == doctest::Approx(12.f));
}

// # ================================================================================

TEST_CASE("grad_sigmoid") {
	GradScalar<float> a(Scalar<float>(0.f));
	auto z = fn::sigmoid<float>(a);
	z.backward();
	CHECK(scalarGradVal(a) == doctest::Approx(0.25f));
}

TEST_CASE("grad_relu") {
	GradScalar<float> a(Scalar<float>(2.f));
	auto z = fn::relu<float>(a);
	z.backward();
	CHECK(scalarGradVal(a) == doctest::Approx(1.f));

	GradScalar<float> b(Scalar<float>(-1.f));
	auto zb = fn::relu<float>(b);
	zb.backward();
	CHECK(scalarGradVal(b) == doctest::Approx(0.f));
}

TEST_CASE("grad_tanh") {
	GradScalar<float> a(Scalar<float>(0.f));
	auto z = fn::tanh<float>(a);
	z.backward();
	CHECK(scalarGradVal(a) == doctest::Approx(1.f));
}

TEST_CASE("grad_softmax") {
	GradVector<float> a(Vector<float>{1.f, 2.f, 3.f});
	auto z = fn::softmax<float>(a);

	z.backward();
	auto g = a.grad();
	for (uint64_t i = 0; i < 3; i++)
		CHECK(std::isfinite(g.at({i})));
}

TEST_CASE("grad_matmul") {
	GradMatrix<float> A(Matrix<float>{{1.f, 2.f}, {3.f, 4.f}});
	GradMatrix<float> B(Matrix<float>{{1.f, 0.f}, {0.f, 1.f}});
	auto C = fn::matmul<float>(A, B);
	C.backward();

	CHECK(A.grad().at({0, 0}) == doctest::Approx(1.f));
	CHECK(A.grad().at({1, 1}) == doctest::Approx(1.f));
}

TEST_CASE("grad_dot") {
	GradVector<float> a(Vector<float>{1.f, 2.f, 3.f});
	GradVector<float> b(Vector<float>{4.f, 5.f, 6.f});
	auto z = fn::dot<float>(a, b);
	z.backward();

	CHECK(a.grad().at({0}) == doctest::Approx(4.f));
	CHECK(b.grad().at({0}) == doctest::Approx(1.f));
}

TEST_CASE("grad_transpose") {
	GradMatrix<float> A(Matrix<float>{{1.f, 2.f}, {3.f, 4.f}});
	auto At = fn::transpose<float>(A);
	At.backward();

	CHECK(A.grad().at({0, 0}) == doctest::Approx(1.f));
}

// # ================================================================================

TEST_CASE("chain_log_exp") {
	GradScalar<float> a(Scalar<float>(3.f));
	auto v = exp<float>(a);
	auto z = log<float>(v);
	z.backward();
	CHECK(scalarGradVal(a) == doctest::Approx(1.f));
}

TEST_CASE("chain_square") {
	GradScalar<float> a(Scalar<float>(4.f));
	auto z = multiply<float>(a, a);
	z.backward();
	CHECK(scalarGradVal(a) == doctest::Approx(8.f));
}

TEST_CASE("chain_simple") {
	GradScalar<float> a(Scalar<float>(2.f));
	GradScalar<float> b(Scalar<float>(5.f));
	GradScalar<float> c(Scalar<float>(3.f));
	auto g = multiply<float>(a, c);
	auto z = add<float>(g, b);
	z.backward();
	CHECK(scalarGradVal(a) == doctest::Approx(3.f));
	CHECK(scalarGradVal(b) == doctest::Approx(1.f));
}

TEST_CASE("chain_simple2") {
	GradScalar<float> a(Scalar<float>(1.f));
	GradScalar<float> b(Scalar<float>(2.f));
	GradScalar<float> c(Scalar<float>(4.f));
	auto g = add<float>(a, b);
	auto z = multiply<float>(g, c);
	z.backward();
	CHECK(scalarGradVal(a) == doctest::Approx(4.f));
	CHECK(scalarGradVal(b) == doctest::Approx(4.f));
	CHECK(scalarGradVal(c) == doctest::Approx(3.f));
}

TEST_CASE("chain_pow") {
	GradScalar<float> a(Scalar<float>(2.f));
	auto a2 = pow<float>(a, Scalar<float>(2.f));
	auto z = pow<float>(a2, Scalar<float>(2.f));
	z.backward();
	CHECK(scalarGradVal(a) == doctest::Approx(32.f));
}

TEST_CASE("chain_linear") {
	Linear<float> L(3, 2);
	GradVector<float> x(Vector<float>{1.f, 1.f, 1.f});
	auto out = L(x);
	auto loss = sum<float>(out);
	loss.backward();

	auto wgrad = L.parameters()[0]->grad().tensor_();
	for (uint64_t r = 0; r < 2; r++)
		for (uint64_t c = 0; c < 3; c++)
			CHECK(wgrad.at({r, c}) == doctest::Approx(1.f));
}

TEST_CASE("chain_linear2") {
	Linear<float> L1(2, 4);
	Linear<float> L2(4, 1);
	GradVector<float> x(Vector<float>{1.f, 1.f});
	auto z1 = L1(x);
	auto h = relu<float>(z1);
	auto out = L2(h);
	auto loss = sum<float>(out);
	loss.backward();
	auto w1grad = L1.parameters()[0]->grad().tensor_();
	auto w2grad = L2.parameters()[0]->grad().tensor_();

	bool w1_nonzero = false, w2_nonzero = false;
	for (uint64_t i = 0; i < 2; i++)
		for (uint64_t j = 0; j < 4; j++)
			if (std::abs(w1grad.at({j, i})) > 1e-6f)
				w1_nonzero = true;
	for (uint64_t i = 0; i < 4; i++)
		if (std::abs(w2grad.at({0, i})) > 1e-6f)
			w2_nonzero = true;
	CHECK(w1_nonzero);
	CHECK(w2_nonzero);
}

// # ================================================================================

TEST_CASE("grad_xor") {
	Vector<float> vx[4], vy[4];
	float xi[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
	float yi[4] = {0, 1, 1, 0};

	for (uint64_t i = 0; i < 4; i++) {
		vx[i] = Vector<float>(2, CPU);
		vx[i].at(0) = xi[i][0];
		vx[i].at(1) = xi[i][1];
		vy[i] = Vector<float>(1, CPU);
		vy[i].at(0) = yi[i];
	}

	GradVector<float> x[4], y[4];
	for (uint64_t i = 0; i < 4; i++) {
		x[i] = vx[i];
		y[i] = vy[i];
	}

	Linear<float> L1(2, 8);
	Linear<float> L2(8, 1);

	auto train = [&](auto& opt) {
		for (uint64_t epoch = 0; epoch < 2000; epoch++) {
			for (uint64_t i = 0; i < 4; i++) {
				auto z1 = L1(x[i]);
				auto h = sigmoid<float>(z1);
				auto z2 = L2(h);
				auto out = sigmoid<float>(z2);
				auto diff = subtract<float, GradVector<float>>(out, y[i]);
				auto sq = multiply<float, GradVector<float>>(diff, diff);
				auto loss = sum<float>(sq);
				loss.backward();
			}
			opt.step();
		}
	};

	auto check = [&]() {
		for (uint64_t i = 0; i < 4; i++) {
			auto z1 = L1(x[i]);
			auto h = sigmoid<float>(z1);
			auto z2 = L2(h);
			auto out = sigmoid<float>(z2);
			CHECK((out.val().at(0)) == doctest::Approx(yi[i]).epsilon(0.2));
		}
	};

	SUBCASE("SGD") {
		SGD<float> opt({}, 0.5f);
		opt.add(L1.parameters());
		opt.add(L2.parameters());
		train(opt);
		check();
	}

	SUBCASE("SGDmomentum") {
		SGDmomentum<float> opt({}, 0.1f, 0.9f);
		opt.add(L1.parameters());
		opt.add(L2.parameters());
		train(opt);
		check();
	}

	SUBCASE("RMSProp") {
		RMSProp<float> opt({}, 0.01f, 0.9f);
		opt.add(L1.parameters());
		opt.add(L2.parameters());
		train(opt);
		check();
	}

	SUBCASE("Adam") {
		Adam<float> opt({}, 0.01f, 0.9f, 0.999f);
		opt.add(L1.parameters());
		opt.add(L2.parameters());
		train(opt);
		check();
	}
}
