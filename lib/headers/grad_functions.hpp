#pragma once
#include "grad_tensor.hpp"

namespace tz::grad {


template <class T, class GradType>
GradType add(GradType& a, GradType& b) {
	a.addConsumer();
	b.addConsumer();
	return GradType::fromOp(a.val() + b.val(), [&](const auto& upstream) {
		a.propagate(upstream);
		b.propagate(upstream);
	});
}

template <class T, class GradType>
GradType multiply(GradType& a, GradType& b) {
	a.addConsumer();
	b.addConsumer();
	return GradType::fromOp(a.val() * b.val(), [&](const auto& upstream) {
		a.propagate(upstream * b.val());
		b.propagate(upstream * a.val());
	});
}


} // namespace tz::grad
