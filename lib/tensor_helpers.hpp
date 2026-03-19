#pragma once

namespace TZ {

template <class T>
void Tensor<T>::computeStrides() {
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


template <class T>
std::ostream& operator<<(std::ostream& os, const Tensor<T>& t) {
	if (t.empty()) {
		os << "[]";
		return os;
	}

	if (t.scalar()) {
		os << t.get();
		return os;
	}

	if (t.dim() == 1) {
		os << "[";
		for (int i = 0; i < t.size() - 1; i++)
			os << t[i].get() << ", ";

		os << t[t.size() - 1].get() << "]";
		return os;
	}

	os << "[\n";
	for (int i = 0; i < t.shape()[0] - 1; i++)
		os << t[i] << ",\n";
	os << t[t.shape()[0] - 1] << "\n]";
	return os;
}

template <typename Derived, typename T>
std::ostream& operator<<(std::ostream& os, const TensorWrapper<Derived, T>& t) {
	os << t.tensor();
	return os;
}

}; // namespace TZ