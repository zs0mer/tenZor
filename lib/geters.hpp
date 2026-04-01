#pragma once

namespace TZ {

template <class T>
T& Tensor<T>::at(const std::vector<uint64_t>& idx) {
	TZ_CHECK(!indexable(), "not indexable");
	TZ_CHECK(idx.size() != dim_, "incorrect number of indices");

	uint64_t linearIdx = offset_;

	for (size_t i = 0; i < dim_; ++i) {
		TZ_CHECK(idx[i] >= shape_[i], "index out of bounds");
		linearIdx += idx[i] * strides_[i];
	}

	return static_cast<T*>(data_->data())[linearIdx];
}

template <class T>
const T& Tensor<T>::at(const std::vector<uint64_t>& idx) const {
	TZ_CHECK(!indexable(), "not indexable");
	TZ_CHECK(idx.size() != dim_, "incorrect number of indices");

	uint64_t linearIdx = offset_;

	for (size_t i = 0; i < dim_; ++i) {
		TZ_CHECK(idx[i] >= shape_[i], "index out of bounds");
		linearIdx += idx[i] * strides_[i];
	}

	return static_cast<const T*>(data_->data())[linearIdx];
}

template <class T>
T& Tensor<T>::at(const uint64_t* idx) {
	uint64_t linearIdx = offset_;

	for (size_t i = 0; i < dim_; ++i) {
		TZ_CHECK(idx[i] >= shape_[i], "index out of bounds");
		linearIdx += idx[i] * strides_[i];
	}

	return static_cast<T*>(data_->data())[linearIdx];
}

template <class T>
const T& Tensor<T>::at(const uint64_t* idx) const {
	uint64_t linearIdx = offset_;

	for (size_t i = 0; i < dim_; ++i) {
		TZ_CHECK(idx[i] >= shape_[i], "index out of bounds");
		linearIdx += idx[i] * strides_[i];
	}

	return static_cast<const T*>(data_->data())[linearIdx];
}

template <class T>
Tensor<T> Tensor<T>::operator[](const uint64_t idx) {
	TZ_CHECK(!indexable(), "not indexable");
	TZ_CHECK(idx >= shape_[0], "index out of bounds");

	return Tensor<T>(dim_ - 1, &shape_[1], &strides_[1], offset_ + idx * strides_[0], data_);
}

template <class T>
const Tensor<T> Tensor<T>::operator[](const uint64_t idx) const {
	TZ_CHECK(!indexable(), "not indexable");
	TZ_CHECK(idx >= shape_[0], "index out of bounds");

	return Tensor<T>(dim_ - 1, &shape_[1], &strides_[1], offset_ + idx * strides_[0], data_);
}

template <class T>
Tensor<T> Tensor<T>::clone() const {
	Tensor<T> out(dim_, shape_.data(), *data_->allocator());

	if (dense()) {
		std::memcpy(out.data(), this->data(), size() * sizeof(T));
		return out;
	}

	apply(*this, out, [](const T& a, T& b) { b = a; });

	return out;
}

template <class T>
T& Tensor<T>::get() {
	TZ_CHECK(!scalar(), "Not a scalar");
	return *(static_cast<T*>(data_->data()) + offset_);
}

template <class T>
const T& Tensor<T>::get() const {
	TZ_CHECK(!scalar(), "Not a scalar");
	return *(static_cast<const T*>(data_->data()) + offset_);
}

} // namespace TZ