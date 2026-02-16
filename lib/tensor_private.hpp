#pragma once

namespace TZ {

template <class T>
Tensor<T>::Tensor(const std::vector<uint64_t>& shape, const mem::Buffer data,
                  const std::vector<uint64_t>& strides, const uint64_t offset)
    : shape_(shape), data_(data), strides_(strides), offset_(offset) {}

template <class T>
void Tensor<T>::ComputeStrides() {
	uint64_t k = 1;
	strides_.resize(dim_);

	for (int64_t i = dim_; i-- > 0;) {
		strides_[i] = k;
		k *= shape_[i];
	}
}


template <class T>
template <class K>
void Tensor<T>::getSTDVecShape(const K& k, std::vector<uint64_t>& shape, uint8_t& currDim) {
	return;
}

template <class T>
template <class K>
void Tensor<T>::getSTDVecShape(const std::vector<K>& v, std::vector<uint64_t>& shape,
                               uint8_t& currDim) {
	shape[currDim++] = v.size();
	if (v.empty())
		return;
	getSTDVecShape(v[0], shape, currDim);
}


template <class T>
template <class K>
void Tensor<T>::falttenSTDVec(const K& k, T* dst, uint64_t& offset) {
	dst[offset++] = static_cast<T>(k);
}

template <class T>
template <class K>
void Tensor<T>::falttenSTDVec(const std::vector<K>& v, T* dst, uint64_t& offset) {
	for (const auto& i : v)
		falttenSTDVec(i, dst, offset);
}

}; // namespace TZ