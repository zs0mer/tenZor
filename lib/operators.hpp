#pragma once

namespace TZ {

template <typename T>
Tensor<T> Tensor<T>::doOp(const Tensor<T>& a, const Tensor<T>& b, std::function<T(T, T)> func) {
	_CHECK(a.size() != b.size(), "a and b do not have the same shape");

	// TODO SDadadddsdasdsadsada

	return out;
}

}; // namespace TZ