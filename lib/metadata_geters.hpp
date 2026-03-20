#pragma once

namespace TZ {

template <class T>
uint64_t Tensor<T>::dim() const {
	return dim_;
}

template <class T>
const uint64_t* Tensor<T>::shape() const {
	return shape_.begin();
}

template <class T>
const uint64_t* Tensor<T>::strides() const {
	return strides_.begin();
}

template <class T>
uint64_t Tensor<T>::size() const {
	if (c_sizeCached_)
		return c_sizeValue_;

	if (dim_ == 0 && !data_->data()) {
		c_sizeCached_ = true;
		c_sizeValue_ = 0;
		return 0;
	}

	uint64_t elements = 1;

	for (uint8_t i = 0; i < dim_; i++)
		elements *= shape_[i];

	c_sizeCached_ = true;
	c_sizeValue_ = elements;

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
bool Tensor<T>::isContiguous() const {
	return isDense() && offset_ == 0;
}

template <class T>
bool Tensor<T>::isDense() const {
	if (c_isDenseCached_)
		return c_isDenseValue_;


	if (empty()) {
		c_isDenseCached_ = true;
		c_isDenseValue_ = true;
		return true;
	}

	uint64_t expected = 1;
	for (int64_t i = dim_; i-- > 0;) {
		if (strides_[i] != expected) {
			c_isDenseCached_ = true;
			c_isDenseValue_ = false;
			return false;
		}

		expected *= shape_[i];
	}
	c_isDenseCached_ = true;
	c_isDenseValue_ = true;

	return true;
}

template <class T>
bool Tensor<T>::scalar() const {
	return dim_ == 0 && size() == 1;
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
