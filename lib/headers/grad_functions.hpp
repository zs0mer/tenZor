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
	a.addConsumer();
	return GradType::fromOp(a.val().exp(),
	                        [&](const auto& upstream) { a.propagate(upstream * a.val().exp()); });
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
	return GradType::fromOp(a.val().pow(p), [&](const auto& upstream) {
		a.propagate(upstream * p * a.val().pow(p - 1));
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
GradVector<T> dot(const GradVector<T>& a, const GradVector<T>& b) {
	a.addConsumer();
	b.addConsumer();
	return GradVector<T>::fromOp(dot(a.val(), b.val()), [&](const auto& upstream) {
		a.propagate(b.val() * upstream.val());
		b.propagate(a.val() * upstream.val());
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
	auto saved = out.val();

	a.addConsumer();
	return GradType::fromOp(std::move(out), [&a, saved](const auto& upstream) {
		auto grad = createSame(upstream);
		grad.apply(upstream, saved, tz::impl::TanhGrad<T>{}, tz::impl::AnyDevice{});
		a.propagate(grad);
	});
}

template <class T, class GradType>
GradType softmax(GradType& a) {
	auto out = createSame(a);
	out.apply(a.val(), tz::impl::Softmax<T>{a.val().exp().sum().get()}, tz::impl::AnyDevice{});
	auto saved = out.val();

	a.addConsumer();
	return GradType::fromOp(std::move(out), [&a, saved](const auto& upstream) {
		auto grad = createSame(upstream);
		grad.apply(upstream, saved, tz::impl::SoftmaxGrad<T>{dot(upstream, saved).get()},
		           tz::impl::AnyDevice{});
		a.propagate(grad);
	});
}

} // namespace tz::grad::fn
