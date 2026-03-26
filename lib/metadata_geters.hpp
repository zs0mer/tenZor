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
const uint64_t* Tensor<T>::_strides() const {
	return strides_.begin();
}

template <class T>
uint64_t Tensor<T>::size() const {
	if (c_size_.cached)
		return c_size_.value;

	if (dim_ == 0 && !data_->data()) {
		c_size_.set(0);
		return 0;
	}

	uint64_t elements = 1;

	for (uint8_t i = 0; i < dim_; i++)
		elements *= shape_[i];

	c_size_.set(elements);

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
T* Tensor<T>::_rawData() {
	if (!data_->data())
		return nullptr;
	return static_cast<T*>(data_->data());
}

template <class T>
const T* Tensor<T>::_rawData() const {
	if (!data_->data())
		return nullptr;
	return static_cast<const T*>(data_->data());
}

template <class T>
uint64_t Tensor<T>::_offset() const {
	return offset_;
}

template <class T>
bool Tensor<T>::dense() const {
	if (c_dense.cached)
		return c_dense.value;


	if (empty()) {
		c_dense.set(true);
		return true;
	}

	uint64_t expected = 1;
	for (int64_t i = dim_; i-- > 0;) {
		if (strides_[i] != expected) {
			c_dense.set(false);
			return false;
		}

		expected *= shape_[i];
	}
	c_dense.set(true);

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

template <class T>
mem::Allocator& Tensor<T>::allocator() const {
	return *data_->allocator();
}

template <class T>
mem::Device Tensor<T>::device() const {
	return data_->allocator()->device();
}

template <class T>
mem::Buffer Tensor<T>::_buffer() {
	return data_;
}

template <class T>
const mem::Buffer Tensor<T>::_buffer() const {
	return data_;
}

} // namespace TZ
