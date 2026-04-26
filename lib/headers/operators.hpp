#pragma once

#include <array>
#include <cmath>
#include <cstdint>

#include "allocator.hpp"
#include "kernel_functions.hpp"
#include "math_classes.hpp"
#include "tensor.hpp"
#include "tensor_impl.hpp"
#include "utils.hpp"

namespace TZ {
namespace internal {

// these are the CPU impl
template <class T>
template <typename Func>
void TensorIMPL<T>::apply(TensorIMPL<T>& a, Func func) {
	if (a.empty())
		return;
	if (a.device() == GPU) {
		cuda::applyGPU(a.getCudaTensor(), func);
		return;
	}

	const uint64_t n = a.size();
	T* base = a.rawData();
	uint64_t offset = a.offset_;

	if (a.dense()) {
#pragma omp parallel for if (n > 1000)
		for (uint64_t i = 0; i < n; i++) {
			func(base[offset + i]);
		}
	} else {
		std::array<uint64_t, MAX_DIM> counters = {};
		uint64_t linearIdx = offset;

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
}

// these are the CPU impl
template <class T>
template <typename Func>
void TensorIMPL<T>::apply(const TensorIMPL<T>& a, TensorIMPL<T>& b, Func func) {
	if (a.empty())
		return;
	TZ_CHECK(isSameShape(a, b), "not same size tensors in apply");
	TZ_CHECK(a.device() == b.device(), "not same device tensors in apply");
	if (a.device() == GPU) {
		cuda::applyGPU(a.getCudaTensor(), b.getCudaTensor(), func);
		return;
	}

	const uint64_t n = a.size();
	const T* baseA = a.rawData();
	T* baseB = b.rawData();


	if (a.dense() && b.dense()) {
#pragma omp parallel for if (n > 10000)
		for (uint64_t i = 0; i < n; i++) {
			func(baseA[a.offset_ + i], baseB[b.offset_ + i]);
		}
	} else {
		std::array<uint64_t, MAX_DIM> counters = {};
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
}

// these are the CPU impl
template <class T>
template <typename Func>
void TensorIMPL<T>::apply(const TensorIMPL<T>& a, const TensorIMPL<T>& b, TensorIMPL<T>& c,
                          Func func) {
	if (a.empty())
		return;
	TZ_CHECK(isSameShape(a, b) && isSameShape(c, b), "not same size tensors in apply");
	TZ_CHECK(a.device() == b.device() && b.device() == c.device(),
	         "not same device tensors in apply");
	if (a.device() == GPU) {
		cuda::applyGPU(a.getCudaTensor(), b.getCudaTensor(), c.getCudaTensor(), func);
		return;
	}

	const uint64_t n = a.size();
	const T* baseA = a.rawData();
	const T* baseB = b.rawData();
	T* baseC = c.rawData();

	if (a.dense() && b.dense() && c.dense()) {
#pragma omp parallel for if (n > 10000)
		for (uint64_t i = 0; i < n; i++) {
			func(baseA[a.offset_ + i], baseB[b.offset_ + i], baseC[c.offset_ + i]);
		}
	} else {
		std::array<uint64_t, MAX_DIM> counters = {};
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
}

// # --------------------

template <class T>
template <typename Func>
void TensorIMPL<T>::apply(Func func) {
	TensorIMPL<T>::apply(*this, func);
}

template <class T>
template <typename Func>
void TensorIMPL<T>::apply(const TensorIMPL<T>& a, Func func) {
	TensorIMPL<T>::apply(a, *this, func);
}

template <class T>
template <typename Func>
void TensorIMPL<T>::apply(const TensorIMPL<T>& a, const TensorIMPL<T>& b, Func func) {
	TensorIMPL<T>::apply(a, b, *this, func);
}

}; // namespace internal

// # ===========================================================================

// does a normal dot product beetwen two vectors
template <class T>
Scalar<T> dot(const Vector<T>& a, const Vector<T>& b) {
	TZ_CHECK(a.size() == b.size(), "not the same size vectors in dot");
	TZ_CHECK(a.device() == b.device(), "not same device vectors in dot");
	if (a.device() == GPU)
		return Scalar<T>(cuda::dot(a.tensor_().getCudaTensor(), b.tensor_().getCudaTensor()));

	uint64_t n = a.size();
	T sum = 0;

	if (a.tensor_().dense() && b.tensor_().dense()) {
		const T* ap = a.tensor_().data();
		const T* bp = b.tensor_().data();

#pragma omp parallel for reduction(+ : sum)
		for (uint64_t i = 0; i < n; i++) {
			sum += ap[i] * bp[i];
		}
	} else {
		for (uint64_t i = 0; i < n; i++)
			sum += a.at(i) * b.at(i);
	}

	return Scalar<T>(sum);
}

// does a normal matrix multiplication
template <class T>
Matrix<T> matmul(const Matrix<T>& a, const Matrix<T>& b) {
	TZ_CHECK(a.cols() == b.rows(), "can't multiply the matrixes");
	TZ_CHECK(a.device() == b.device(), "not same device matrixes in matmul");
	Matrix<T> out(a.rows(), b.cols(), a.device());

	if (a.device() == GPU) {
		Matrix<T> inA = a;
		Matrix<T> inB = b;
		if (a.tensor_().strides()[1] != 1)
			inA = transpose(a);
		if (b.tensor_().strides()[1] != 1)
			inA = transpose(b);

		cuda::matmul(inA.tensor_().getCudaTensor(), inB.tensor_().getCudaTensor(),
		             out.tensor_().getCudaTensor());
		return out;
	}

#pragma omp parallel for
	for (uint64_t i = 0; i < a.rows(); i++) {
		for (uint64_t j = 0; j < b.cols(); j++) {
			out.at(i, j) = 0;
		}
		for (uint64_t k = 0; k < a.cols(); k++) {
			for (uint64_t j = 0; j < b.cols(); j++) {
				out.at(i, j) += a.at(i, k) * b.at(k, j);
			}
		}
	}
	return out;
}

// returns the transposed Matrix
template <class T>
Matrix<T> transpose(const Matrix<T>& m) {
	std::array<uint64_t, 2> strides = {m.tensor_().strides()[1], m.tensor_().strides()[0]};
	std::array<uint64_t, 2> shape = {m.cols(), m.rows()};
	return Matrix<T>(internal::TensorIMPL<T>(2, shape.data(), strides.data(), m.tensor_().offset(),
	                                         m.tensor_().buffer()));
}

// ! don't use with integers
// ! can't run on GPU
// returns the determinant of the Matrix
template <class T>
Scalar<T> det(const Matrix<T>& m) {
	//^ https://en.wikipedia.org/wiki/Gaussian_elimination
	TZ_CHECK(m.rows() == m.cols(), "matrix must be square");

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
#pragma omp parallel for
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
}; // namespace TZ
