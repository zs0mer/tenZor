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

namespace math {

template <class T>
Scalar<T> dot(const Vector<T>& a, const Vector<T>& b) {
	_CHECK(a.size() != b.size(), "not the same size vectors in dot");
	Scalar<T> out(0);
	uint64_t n = a.size();

	if (a.dense() && b.dense()) {
		const T* ap = a.data();
		const T* bp = b.data();

		for (uint64_t i = 0; i < n; i++)
			out.get() += ap[i] * bp[i];
	} else {
		for (uint64_t i = 0; i < n; i++)
			out.get() += a.at(i) * b.at(i);
	}

	return out;
}

template <class T>
Matrix<T> matmul(const Matrix<T>& a, const Matrix<T>& b) {
	_CHECK(a.cols() != b.rows(), "can't multiply the matrixes");
	Matrix<T> out(a.rows(), b.cols(), a.tensor().allocator());

	for (uint64_t i = 0; i < a.rows(); i++) {
		for (uint64_t j = 0; j < b.cols(); j++) {
			out[i][j] = dot(a.row(i), b.col(j));
		}
	}
	return out;
}

template <class T>
Matrix<T> transpose(const Matrix<T>& m) {
	std::array<uint64_t, 2> strides = {m.tensor().strides()[1], m.tensor().strides()[0]};
	std::array<uint64_t, 2> shape = {m.cols(), m.rows()};
	return Matrix<T>(
	    Tensor<T>(2, shape.data(), strides.data(), m.tensor().offset(), m.tensor().buffer()));
}

template <class T>
// If the matrix has intregers inside it, it won't give an acurate anwser
Scalar<T> det(const Matrix<T>& m) {
	//^ https://en.wikipedia.org/wiki/Gaussian_elimination
	_CHECK(m.rows() != m.cols(), "matrix must be square");

	const uint64_t n = m.rows();

	Matrix<T> A = m.clone();

	T det = 1;

	for (uint64_t k = 0; k < n; k++) {
		// 1. Find pivot row
		uint64_t pivot = k;
		for (uint64_t i = k + 1; i < n; i++) {
			if (std::abs(A.at(i, k)) > std::abs(A.at(pivot, k))) {
				pivot = i;
			}
		}

		// 2. If pivot is zero return 0
		if (A.at(pivot, k) == T(0)) {
			return Scalar<T>(0);
		}

		// 3. Swap rows if needed
		if (pivot != k) {
			A.swapRow(k, pivot);
			det = -det;
		}

		// 4. Eliminate below rows k-th column
		for (uint64_t i = k + 1; i < n; i++) {
			T factor = A.at(i, k) / A.at(k, k);

			for (uint64_t j = k + 1; j < n; j++) {
				A.at(i, j) -= factor * A.at(k, j);
			}
		}

		// 5. Multiply diagonal
		det *= A.at(k, k);
	}

	return Scalar<T>(det);
}
}; // namespace math
}; // namespace TZ