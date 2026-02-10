#pragma once

namespace TZ {

template <class T>
T& Tensor<T>::at(const std::vector<uint64_t>& idx) {}

template <class T>
const T& Tensor<T>::at(const std::vector<uint64_t>& idx) const {}

template <class T>
Tensor<T> Tensor<T>::operator[](const uint64_t idx) const {}

template <class T>
Tensor<T> Tensor<T>::clone() const {
	return Tensor(shape_, data_.clone(), strides_, offset_);
}

template <class T>
T& Tensor<T>::scalarVal() {
	_CHECK(!shape_.empty() && !data_->data(), "Not a scalar");
	return data_->data()[0];
}

template <class T>
const T& Tensor<T>::scalarVal() const {
	_CHECK(!shape_.empty() && !data_->data(), "Not a scalar");
	return data_->data()[0];
}

} // namespace TZ