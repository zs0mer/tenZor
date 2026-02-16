#pragma once

namespace TZ {

template <class T>
uint64_t Tensor<T>::dim() const {
	return dim_;
}

template <class T>
const std::vector<uint64_t>& Tensor<T>::shape() const {
	return shape_;
}

template <class T>
uint64_t Tensor<T>::size() const {
	if (!data_->data())
		return 0;
	uint64_t elements = 1;

	for (uint8_t i = 0; i < dim_; i++)
		elements *= shape_[i];

	return elements;
};

template <class T>
T* Tensor<T>::data() {
	if (!data_->data())
		return nullptr;
	return static_cast<T*>(data_->data()) + offset_;
}

template <class T>
const T* Tensor<T>::data() const {
	if (!data_->data())
		return nullptr;
	return static_cast<const T*>(data_->data()) + offset_;
}

template <class T>
T* Tensor<T>::rawData() {
	if (!data_->data())
		return nullptr;
	return static_cast<T*>(data_->data());
}

template <class T>
const T* Tensor<T>::rawData() const {
	if (!data_->data())
		return nullptr;
	return static_cast<const T*>(data_->data());
}

template <class T>
bool Tensor<T>::isContiguous(const bool softCheck) const {
	if (offset_ != 0 && !softCheck)
		return false;
	if (shape_.empty())
		return true;

	uint64_t expected = 1;
	for (int64_t i = dim_; i-- > 0;) {
		if (strides_[i] != expected)
			return false;
		expected *= shape_[i];
	}

	return true;
}

template <class T>
bool Tensor<T>::scalar() const {
	return shape_.empty() && size() == 1;
}

template <class T>
bool Tensor<T>::empty() const {
	return size() == 0;
}

template <class T>
bool Tensor<T>::indexable() const {
	return !scalar() && !empty();
}

} // namespace TZ
