#define DOCTEST_CONFIG_COLORS
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "TenZor.hpp"

using namespace tz;
using namespace tz::grad;
using namespace tz::grad::fn;

// helpers
static float grad_of(GradScalar<float>& s) {
	return Scalar<float>(s.grad().tensor_()).get();
}

// # ================================================================================

TEST_CASE("grad_add_scalars") {
	GradScalar<float> a(Scalar<float>(2.f)), b(Scalar<float>(3.f));
	auto z = add<float>(a, b);
	z.backward();
	CHECK(grad_of(a) == doctest::Approx(1.f));
	CHECK(grad_of(b) == doctest::Approx(1.f));
}

TEST_CASE("grad_subtract_scalars") {
	GradScalar<float> a(Scalar<float>(5.f)), b(Scalar<float>(3.f));
	auto z = subtract<float>(a, b);
	z.backward();
	CHECK(grad_of(a) == doctest::Approx(1.f));
	CHECK(grad_of(b) == doctest::Approx(-1.f));
}

TEST_CASE("grad_multiply_scalars") {
	GradScalar<float> a(Scalar<float>(3.f)), b(Scalar<float>(4.f));
	auto z = multiply<float>(a, b);
	z.backward();
	CHECK(grad_of(a) == doctest::Approx(4.f));
	CHECK(grad_of(b) == doctest::Approx(3.f));
}

TEST_CASE("grad_divide_scalars") {
	GradScalar<float> a(Scalar<float>(6.f)), b(Scalar<float>(2.f));
	auto z = divide<float>(a, b);
	z.backward();
	CHECK(grad_of(a) == doctest::Approx(0.5f));
	CHECK(grad_of(b) == doctest::Approx(-1.5f));
}

TEST_CASE("grad_negate_scalar") {
	GradScalar<float> a(Scalar<float>(3.f));
	auto z = negate<float>(a);
	z.backward();
	CHECK(grad_of(a) == doctest::Approx(-1.f));
}

// # ================================================================================

TEST_CASE("grad_exp") {
	GradScalar<float> a(Scalar<float>(0.f));
	auto z = exp<float>(a);
	z.backward();
	CHECK(z.val().get() == doctest::Approx(1.f));
	CHECK(grad_of(a) == doctest::Approx(1.f));
}

TEST_CASE("grad_exp_nonzero") {
	GradScalar<float> a(Scalar<float>(1.f));
	auto z = exp<float>(a);
	z.backward();
	CHECK(grad_of(a) == doctest::Approx(std::exp(1.f)));
}

TEST_CASE("grad_log") {
	GradScalar<float> a(Scalar<float>(2.f));
	auto z = log<float>(a);
	z.backward();
	CHECK(grad_of(a) == doctest::Approx(0.5f));
}

TEST_CASE("grad_pow") {
	GradScalar<float> a(Scalar<float>(2.f));
	auto z = pow<float>(a, Scalar<float>(3.f));
	z.backward();
	CHECK(grad_of(a) == doctest::Approx(12.f));
}

// # ================================================================================

TEST_CASE("grad_chain_mul_add") {
	// z = a*b + c  →  dz/da=b=4, dz/db=a=3, dz/dc=1
	GradScalar<float> a(Scalar<float>(3.f)), b(Scalar<float>(4.f)), c(Scalar<float>(1.f));
	auto t = multiply<float>(a, b);
	auto z = add<float>(t, c);
	z.backward();
	CHECK(grad_of(a) == doctest::Approx(4.f));
	CHECK(grad_of(b) == doctest::Approx(3.f));
	CHECK(grad_of(c) == doctest::Approx(1.f));
}

TEST_CASE("grad_shared_input") {
	// z = a + a = 2a  →  dz/da = 2
	GradScalar<float> a(Scalar<float>(5.f));
	auto z = add<float>(a, a);
	z.backward();
	CHECK(grad_of(a) == doctest::Approx(2.f));
}

TEST_CASE("grad_chain_exp_mul") {
	// z = exp(a) * b,  dz/da = exp(a)*b,  dz/db = exp(a)
	GradScalar<float> a(Scalar<float>(0.f)), b(Scalar<float>(3.f));
	auto e = exp<float>(a);
	auto z = multiply<float>(e, b);
	z.backward();
	CHECK(grad_of(a) == doctest::Approx(3.f)); // exp(0)*3
	CHECK(grad_of(b) == doctest::Approx(1.f)); // exp(0)
}

// # ================================================================================

TEST_CASE("grad_sum_vector") {
	Vector<float> v(4, CPU);
	v.setAll(1.f);
	GradVector<float> a(v);
	auto z = sum<float>(a);
	z.backward();
	// dsum/dv[i] = 1 for all i
	for (int i = 0; i < 4; i++)
		CHECK(a.grad().tensor_().data()[i] == doctest::Approx(1.f));
}

TEST_CASE("grad_vector_add") {
	Vector<float> va(3, CPU), vb(3, CPU);
	va.setAll(1.f);
	vb.setAll(2.f);
	GradVector<float> a(va), b(vb);
	auto z = add<float>(a, b);
	auto sz = sum<float>(z);
	sz.backward();
	for (int i = 0; i < 3; i++) {
		CHECK(a.grad().tensor_().data()[i] == doctest::Approx(1.f));
		CHECK(b.grad().tensor_().data()[i] == doctest::Approx(1.f));
	}
}

// # ================================================================================

TEST_CASE("grad_sigmoid") {
	// sigma(0)=0.5, sigma'(0)=0.25
	Vector<float> v(1, CPU);
	v.setAll(0.f);
	GradVector<float> a(v);
	auto z = sigmoid<float>(a);
	auto sz = sum<float>(z);
	sz.backward();
	CHECK(a.grad().tensor_().data()[0] == doctest::Approx(0.25f));
}

TEST_CASE("grad_relu_positive") {
	Vector<float> v(3, CPU);
	v.at(0) = 1.f;
	v.at(1) = 2.f;
	v.at(2) = 3.f;
	GradVector<float> a(v);
	auto z = relu<float>(a);
	auto sz = sum<float>(z);
	sz.backward();
	for (int i = 0; i < 3; i++)
		CHECK(a.grad().tensor_().data()[i] == doctest::Approx(1.f));
}

TEST_CASE("grad_relu_negative") {
	Vector<float> v(3, CPU);
	v.at(0) = -1.f;
	v.at(1) = -2.f;
	v.at(2) = -3.f;
	GradVector<float> a(v);
	auto z = relu<float>(a);
	auto sz = sum<float>(z);
	sz.backward();
	for (int i = 0; i < 3; i++)
		CHECK(a.grad().tensor_().data()[i] == doctest::Approx(0.f));
}

TEST_CASE("grad_tanh") {
	// tanh(0)=0, tanh'(0)=1
	Vector<float> v(1, CPU);
	v.setAll(0.f);
	GradVector<float> a(v);
	auto z = tanh<float>(a);
	auto sz = sum<float>(z);
	sz.backward();
	CHECK(a.grad().tensor_().data()[0] == doctest::Approx(1.f));
}

// # ================================================================================

TEST_CASE("grad_matmul") {
	// C = A*B, loss = sum(C)
	// dL/dA = dL/dC * B^T = ones * B^T
	// dL/dB = A^T * dL/dC = A^T * ones
	Matrix<float> ma = {{1.f, 2.f}, {3.f, 4.f}};
	Matrix<float> mb = {{1.f, 0.f}, {0.f, 1.f}}; // identity
	GradMatrix<float> A(ma), B(mb);
	auto C = matmul<float>(A, B);
	auto sz = sum<float>(C);
	sz.backward();

	// dL/dA = ones * I^T = ones
	for (int i = 0; i < 4; i++)
		CHECK(A.grad().tensor_().data()[i] == doctest::Approx(1.f));
}

// # ================================================================================

TEST_CASE("grad_linear_shapes") {
	Linear<float> L(4, 3);
	Vector<float> in(4, CPU);
	in.setAll(1.f);
	GradVector<float> x(in);
	auto out = L(x);
	CHECK(out.val().size() == 3);
}

TEST_CASE("grad_sgd_step") {
	// single param, manually check update
	GradScalar<float> p(Scalar<float>(10.f));
	SGD<float> opt({&p}, 0.1f);

	// fake a gradient
	GradScalar<float> q = add<float>(p, p); // dq/dp = 2
	q.backward();

	CHECK(grad_of(p) == doctest::Approx(2.f));
	opt.step();
	CHECK(p.val().get() == doctest::Approx(9.8f)); // 10 - 0.1*2
}

TEST_CASE("grad_zero_grad") {
	GradScalar<float> p(Scalar<float>(1.f));
	auto z = add<float>(p, p);
	z.backward();
	CHECK(grad_of(p) == doctest::Approx(2.f));
	p.zeroGrad();
	CHECK(grad_of(p) == doctest::Approx(0.f));
}

// # ================================================================================

TEST_CASE("grad_xor") {
	Vector<float> vx[4], vy[4];
	float xi[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
	float yi[4] = {0, 1, 1, 0};

	for (int i = 0; i < 4; i++) {
		vx[i] = Vector<float>(2, CPU);
		vx[i].at(0) = xi[i][0];
		vx[i].at(1) = xi[i][1];
		vy[i] = Vector<float>(1, CPU);
		vy[i].at(0) = yi[i];
	}

	GradVector<float> x[4], y[4];
	for (int i = 0; i < 4; i++) {
		x[i] = vx[i];
		y[i] = vy[i];
	}

	Linear<float> L1(2, 8);
	Linear<float> L2(8, 1);

	auto train = [&](auto& opt) {
		for (int epoch = 0; epoch < 500; epoch++) {
			for (int i = 0; i < 4; i++) {
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
		for (int i = 0; i < 4; i++) {
			auto z1 = L1(x[i]);
			auto h = sigmoid<float>(z1);
			auto z2 = L2(h);
			auto out = sigmoid<float>(z2);
			CHECK(std::round(out.val().at(0)) == doctest::Approx(yi[i]));
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
