#pragma once
#include "grad_classes.hpp"
#include "math_classes.hpp"
#include "tensor.hpp"
#include "utils.hpp"

namespace tz::grad::fn {

// # elementary functions =======================================================================

template <class T, class GradType>
GradType add(const GradType& a, const GradType& b) {
	a.addConsumer();
	b.addConsumer();
	return GradType::fromOp(a.val() + b.val(), [&](const auto& upstream) {
		a.propagate(upstream);
		b.propagate(upstream);
	});
}

template <class T, class GradType>
GradType subtract(const GradType& a, const GradType& b) {
	a.addConsumer();
	b.addConsumer();
	return GradType::fromOp(a.val() - b.val(), [&](const auto& upstream) {
		a.propagate(upstream);
		b.propagate(-upstream);
	});
}

template <class T, class GradType>
GradType multiply(const GradType& a, const GradType& b) {
	a.addConsumer();
	b.addConsumer();
	return GradType::fromOp(a.val() * b.val(), [&](const auto& upstream) {
		a.propagate(upstream * b.val());
		b.propagate(upstream * a.val());
	});
}

template <class T, class GradType>
GradType divide(const GradType& a, const GradType& b) {
	a.addConsumer();
	b.addConsumer();
	return GradType::fromOp(a.val() / b.val(), [&](const auto& upstream) {
		a.propagate(upstream / b.val());
		b.propagate(-upstream * a.val() / (b.val() * b.val()));
	});
}

template <class T, class GradType>
GradType negate(const GradType& a) {
	a.addConsumer();
	return GradType::fromOp(-a.val(), [&](const auto& upstream) { a.propagate(-upstream); });
}

template <class T>
GradScalar<T> sum(GradVector<T>& a) {
	a.addConsumer();
	return GradScalar<T>::fromOp(a.val().sum(), [&](const auto& upstream) {
		a.propagate(upstream.broadcast(a.val().size()));
	});
}

template <class T>
GradScalar<T> sum(GradMatrix<T>& a) {
	a.addConsumer();
	return GradScalar<T>::fromOp(a.val().sum(), [&](const auto& upstream) {
		a.propagate(upstream.broadcast(a.val().rows(), a.val().cols()));
	});
}

template <class T>
GradScalar<T> sum(GradTensor<T>& a) {
	a.addConsumer();
	return GradScalar<T>::fromOp(a.val().sum(), [&](const auto& upstream) {
		a.propagate(upstream.broadcast(a.val().dim(), a.val().shape()));
	});
}

template <class T, class GradType>
GradType exp(const GradType& a) {
	auto saved = a.val().exp();
	a.addConsumer();
	return GradType::fromOp(std::move(saved),
	                        [&a, saved](const auto& upstream) { a.propagate(upstream * saved); });
}

template <class T, class GradType>
GradType log(const GradType& a) {
	a.addConsumer();
	return GradType::fromOp(a.val().log(),
	                        [&](const auto& upstream) { a.propagate(upstream / a.val()); });
}

template <class T, class GradType>
GradType pow(const GradType& a, const Scalar<T>& p) {
	a.addConsumer();
	return GradType::fromOp(a.val().pow(p), [&a, p](const auto& upstream) {
		a.propagate(upstream * p * a.val().pow(p - Scalar<T>(1)));
	});
}

template <class T>
GradMatrix<T> matmul(const GradMatrix<T>& a, const GradMatrix<T>& b) {
	a.addConsumer();
	b.addConsumer();
	return GradMatrix<T>::fromOp(matmul(a.val(), b.val()), [&](const auto& upstream) {
		a.propagate(matmul(upstream, b.val().transpose()));
		b.propagate(matmul(a.val().transpose(), upstream));
	});
}

template <class T>
GradVector<T> matmul(const GradMatrix<T>& a, const GradVector<T>& b) {
	a.addConsumer();
	b.addConsumer();
	return GradVector<T>::fromOp(matmul(a.val(), Matrix<T>(b.val())), [&](const auto& upstream) {
		a.propagate(matmul(Matrix<T>(upstream), b.val().transpose()));
		b.propagate(matmul(a.val().transpose(), Matrix<T>(upstream)));
	});
}

template <class T>
GradScalar<T> dot(const GradVector<T>& a, const GradVector<T>& b) {
	a.addConsumer();
	b.addConsumer();
	return GradScalar<T>::fromOp(dot(a.val(), b.val()), [&](const auto& upstream) {
		a.propagate(b.val() * upstream);
		b.propagate(a.val() * upstream);
	});
}

template <class T, class GradType>
GradType transpose(const GradType& a) {
	a.addConsumer();
	return GradType::fromOp(a.val().transpose(),
	                        [&](const auto& upstream) { a.propagate(upstream.transpose()); });
}

// # complex functions =======================================================================

template <class T, class GradType>
GradType sigmoid(GradType& a) {
	auto out = createSame(a.val());
	out.apply(a.val(), tz::impl::Sigmoid<T>{}, tz::impl::AnyDevice{});
	auto saved = out;

	a.addConsumer();
	return GradType::fromOp(std::move(out), [&a, saved](const auto& upstream) {
		auto grad = createSame(upstream);
		grad.apply(upstream, saved, tz::impl::SigmoidGrad<T>{}, tz::impl::AnyDevice{});
		a.propagate(grad);
	});
}

template <class T, class GradType>
GradType relu(GradType& a) {
	auto out = createSame(a.val());
	out.apply(a.val(), tz::impl::Relu<T>{}, tz::impl::AnyDevice{});

	a.addConsumer();
	return GradType::fromOp(std::move(out), [&](const auto& upstream) {
		auto grad = createSame(upstream);
		grad.apply(upstream, a.val(), tz::impl::ReluGrad<T>{}, tz::impl::AnyDevice{});
		a.propagate(grad);
	});
}

template <class T, class GradType>
GradType tanh(GradType& a) {
	auto out = createSame(a.val());
	out.apply(a.val(), tz::impl::Tanh<T>{}, tz::impl::AnyDevice{});
	auto saved = out;

	a.addConsumer();
	return GradType::fromOp(std::move(out), [&a, saved](const auto& upstream) {
		auto grad = createSame(upstream);
		grad.apply(upstream, saved, tz::impl::TanhGrad<T>{}, tz::impl::AnyDevice{});
		a.propagate(grad);
	});
}

template <class T, class GradType>
GradType softmax(GradType& a) {
	auto e = a.val().exp();
	auto out = e / e.sum();

	auto saved = out;
	a.addConsumer();
	return GradType::fromOp(std::move(out), [&a, saved](const auto& upstream) {
		auto grad = saved * (upstream - dot(upstream, saved));
		a.propagate(grad);
	});
}

// # delete some rvalues =====================================================================

#define TZ_DELETE_RVALUE(func)                                                                     \
	template <class T>                                                                             \
	GradScalar<T> func(GradScalar<T>&&) = delete;                                                  \
	template <class T>                                                                             \
	GradVector<T> func(GradVector<T>&&) = delete;                                                  \
	template <class T>                                                                             \
	GradMatrix<T> func(GradMatrix<T>&&) = delete;

TZ_DELETE_RVALUE(log)
TZ_DELETE_RVALUE(exp)
TZ_DELETE_RVALUE(negate)
TZ_DELETE_RVALUE(sigmoid)
TZ_DELETE_RVALUE(relu)
TZ_DELETE_RVALUE(tanh)
TZ_DELETE_RVALUE(softmax)

#define TZ_DELETE_RVALUE_BINARY(func)                                                              \
	template <class T>                                                                             \
	GradScalar<T> func(GradScalar<T>&&, GradScalar<T>&&) = delete;                                 \
	template <class T>                                                                             \
	GradVector<T> func(GradVector<T>&&, GradVector<T>&&) = delete;                                 \
	template <class T>                                                                             \
	GradMatrix<T> func(GradMatrix<T>&&, GradMatrix<T>&&) = delete;                                 \
	template <class T>                                                                             \
                                                                                                   \
	GradScalar<T> func(const GradScalar<T>&, GradScalar<T>&&) = delete;                            \
	template <class T>                                                                             \
	GradVector<T> func(const GradVector<T>&, GradVector<T>&&) = delete;                            \
	template <class T>                                                                             \
	GradMatrix<T> func(const GradMatrix<T>&, GradMatrix<T>&&) = delete;                            \
                                                                                                   \
	template <class T>                                                                             \
	GradScalar<T> func(GradScalar<T>&&, const GradScalar<T>&) = delete;                            \
	template <class T>                                                                             \
	GradVector<T> func(GradVector<T>&&, const GradVector<T>&) = delete;                            \
	template <class T>                                                                             \
	GradMatrix<T> func(GradMatrix<T>&&, const GradMatrix<T>&) = delete;


TZ_DELETE_RVALUE_BINARY(add)
TZ_DELETE_RVALUE_BINARY(subtract)
TZ_DELETE_RVALUE_BINARY(multiply)
TZ_DELETE_RVALUE_BINARY(divide)
TZ_DELETE_RVALUE_BINARY(pow)


template <class T>
GradMatrix<T> matmul(GradMatrix<T>&&, GradMatrix<T>&&) = delete;
template <class T>
GradMatrix<T> matmul(const GradMatrix<T>&, GradMatrix<T>&&) = delete;
template <class T>
GradMatrix<T> matmul(GradMatrix<T>&&, const GradMatrix<T>&) = delete;

template <class T>
GradVector<T> matmul(GradMatrix<T>&&, GradVector<T>&&) = delete;
template <class T>
GradVector<T> matmul(const GradMatrix<T>&&, GradVector<T>&&) = delete;
template <class T>
GradVector<T> matmul(GradMatrix<T>&&, const GradVector<T>&) = delete;

template <class T>
GradScalar<T> dot(GradVector<T>&&, GradVector<T>&&) = delete;
template <class T>
GradScalar<T> dot(const GradVector<T>&, GradVector<T>&&) = delete;
template <class T>
GradScalar<T> dot(GradVector<T>&&, const GradVector<T>&) = delete;

} // namespace tz::grad::fn
