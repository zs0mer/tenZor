#pragma once

namespace TZ {

template <class T>
void Tensor<T>::ComputeStrides() {
	uint64_t k = 1;

	for (int64_t i = dim_; i-- > 0;) {
		strides_[i] = k;
		k *= shape_[i];
	}
}


template <class T>
template <class K>
void Tensor<T>::getSTDVecShape(const K& k, uint64_t* const shape, uint8_t& currDim) {
	return;
}

template <class T>
template <class K>
void Tensor<T>::getSTDVecShape(const std::vector<K>& v, uint64_t* const shape, uint8_t& currDim) {
	shape[currDim++] = v.size();
	_CHECK(currDim > MAX_DIM, "tensor dimension exceeds MAX_DIMS");
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