#pragma once

namespace TZ {

template <class T>
uint64_t Tensor<T>::dim() const {
	return shape_.size();
}

template <class T>
const std::vector<uint64_t>& Tensor<T>::shape() const {
	return shape_;
}

template <class T>
uint64_t Tensor<T>::numel() const {
	if (!data_->data())
		return 0;
	uint64_t elements = 1;

	for (uint64_t i : shape_)
		elements *= i;

	return elements;
};

template <class T>
T* Tensor<T>::data(const bool fullBuffer) {
	if (fullBuffer)
		return static_cast<T*>(data_->data());
	else
		return static_cast<T*>(data_->data()) + offset_;
}

template <class T>
const T* Tensor<T>::data(const bool fullBuffer) const {
	if (fullBuffer)
		return static_cast<const T*>(data_->data());
	else
		return static_cast<const T*>(data_->data()) + offset_;
}

template <class T>
bool Tensor<T>::empty() const {
	return numel() == 0;
}

template <class T>
bool Tensor<T>::isContiguous(const bool softCheck) const {
	if (offset_ != 0 && !softCheck)
		return false;
	if (shape_.empty())
		return true;

	uint64_t expected = 1;
	for (int64_t i = static_cast<int64_t>(shape_.size()) - 1; i >= 0; --i) {
		if (strides_[i] != expected)
			return false;
		expected *= shape_[i];
	}

	return true;
}

template <class T>
bool Tensor<T>::isScalar() const {
	return shape_.empty() && numel() == 1;
}


} // namespace TZ
