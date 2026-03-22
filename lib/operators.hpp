#pragma once

namespace TZ {

template <class T>
template <typename Func>
void Tensor<T>::apply(Tensor<T>& a, Func func) {
	const uint64_t n = a.size();


	if (a.dense()) {
		T* ptr = a.data();
		for (uint64_t i = 0; i < n; i++)
			func(ptr[i]);
		return;
	}

	std::array<uint64_t, MAX_DIM> counters = {};

	T* base = a.rawData();
	uint64_t linearIdx = a.offset_;

	for (uint64_t i = 0; i < n; i++) {
		func(base[linearIdx]);

		for (uint8_t d = a.dim_; d-- > 0;) {
			linearIdx += a.strides_[d];
			if (++counters[d] < a.shape_[d])
				break;

			linearIdx -= counters[d] * a.strides_[d];
			counters[d] = 0;
		}
	}
}

template <class T>
template <typename Func>
void Tensor<T>::apply(const Tensor<T>& a, Tensor<T>& b, Func func) {
	_CHECK(!isSameShape(a, b), "not same size tensors in apply");

	const uint64_t n = a.size();

	if (a.dense() && b.dense()) {
		const T* ap = a.data();
		T* bp = b.data();

		for (uint64_t i = 0; i < n; i++)
			func(ap[i], bp[i]);

		return;
	}

	std::array<uint64_t, MAX_DIM> counters = {};

	const T* baseA = a.rawData();
	T* baseB = b.rawData();
	uint64_t linearIdxA = a.offset_;
	uint64_t linearIdxB = b.offset_;

	for (uint64_t i = 0; i < n; i++) {
		func(baseA[linearIdxA], baseB[linearIdxB]);

		for (uint8_t d = a.dim_; d-- > 0;) {
			linearIdxA += a.strides_[d];
			linearIdxB += b.strides_[d];
			if (++counters[d] < a.shape_[d])
				break;

			linearIdxA -= counters[d] * a.strides_[d];
			linearIdxB -= counters[d] * b.strides_[d];

			counters[d] = 0;
		}
	}
}

template <class T>
template <typename Func>
void Tensor<T>::apply(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& c, Func func) {
	_CHECK(!isSameShape(a, b) || !isSameShape(c, b), "not same size tensors in apply");

	const uint64_t n = a.size();

	if (a.dense() && b.dense() && c.dense()) {
		const T* ap = a.data();
		const T* bp = b.data();
		T* cp = c.data();

		for (uint64_t i = 0; i < n; i++)
			func(ap[i], bp[i], cp[i]);

		return;
	}

	std::array<uint64_t, MAX_DIM> counters = {};

	const T* baseA = a.rawData();
	const T* baseB = b.rawData();
	T* baseC = c.rawData();

	uint64_t linearIdxA = a.offset_;
	uint64_t linearIdxB = b.offset_;
	uint64_t linearIdxC = c.offset_;


	for (uint64_t i = 0; i < n; i++) {
		func(baseA[linearIdxA], baseB[linearIdxB], baseC[linearIdxC]);

		for (uint8_t d = a.dim_; d-- > 0;) {
			linearIdxA += a.strides_[d];
			linearIdxB += b.strides_[d];
			linearIdxC += c.strides_[d];

			if (++counters[d] < a.shape_[d])
				break;

			linearIdxA -= counters[d] * a.strides_[d];
			linearIdxB -= counters[d] * b.strides_[d];
			linearIdxC -= counters[d] * c.strides_[d];

			counters[d] = 0;
		}
	}
}


template <class T>
template <typename Func>
void Tensor<T>::apply(Func func) {
	Tensor<T>::apply(*this, func);
}

template <class T>
template <typename Func>
void Tensor<T>::apply(const Tensor<T>& a, Func func) {
	Tensor<T>::apply(a, *this, func);
}

template <class T>
template <typename Func>
void Tensor<T>::apply(const Tensor<T>& a, const Tensor<T>& b, Func func) {
	Tensor<T>::apply(a, b, *this, func);
}

}; // namespace TZ