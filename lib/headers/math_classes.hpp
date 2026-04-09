#pragma once

#include <vector>
#include <cstdint>
#include <array>

#include "allocator.hpp"
#include "tensor.hpp"
#include "utils.hpp"

namespace TZ {

template <typename T>
class Scalar;

template <typename T>
class Vector;

template <typename T>
class Matrix;

template <typename T>
class Tensor;

namespace internal {

// # tensorWrapper =============================================================

template <typename Derived, typename T>
class TensorWrapper {
  protected:
	internal::TensorIMPL<T> t_;

  public:
	// # seters ===========================================================================

	TensorWrapper() = default;

	TensorWrapper(const internal::TensorIMPL<T>& t) : t_(t) {}

	TensorWrapper(internal::TensorIMPL<T>&& t) : t_(std::move(t)) {}

	TensorWrapper(const std::vector<uint64_t>& shape, Device device = CPU) : t_(shape, device) {}

	TensorWrapper(const uint8_t dim, const uint64_t* shape, Device device = CPU)
	    : t_(dim, shape, device) {}


	TensorWrapper(const TensorWrapper&) = default;

	TensorWrapper& operator=(const TensorWrapper&) = default;

	TensorWrapper(TensorWrapper&&) = default;

	TensorWrapper& operator=(TensorWrapper&&) = default;

	Tensor<T> toTensor() {
		return Tensor<T>(this->tensor_());
	}

	// # metadata geters ===========================================================================

	uint64_t dim() const {
		return t_.dim();
	}

	const uint64_t* shape() const {
		return t_.shape();
	}

	bool empty() const {
		return t_.empty();
	}

	bool device() const {
		return t_.device();
	}

	// # geters ===========================================================================

	// returns the inner tensor
	internal::TensorIMPL<T>& tensor_() {
		return t_;
	}

	// returns the inner tensor
	const internal::TensorIMPL<T>& tensor_() const {
		return t_;
	}

	// clones the object
	Derived clone() const {
		return Derived(this->t_.clone());
	}

	Tensor<T> operator[](uint64_t idx) const {
		return Tensor<T>(this->t_[idx]);
	}

	// # operations ===========================================================================

	Derived operator+(const TensorWrapper& other) const {
		Derived out(t_.dim(), t_.shape(), t_.allocator());
		internal::TensorIMPL<T>::apply(this->t_, other.t_, out.t_,
		                               [](const T& a, const T& b, T& c) { c = a + b; });
		return out;
	}

	Derived operator-(const TensorWrapper& other) const {
		Derived out(t_.dim(), t_.shape(), t_.allocator());
		internal::TensorIMPL<T>::apply(this->t_, other.t_, out.t_,
		                               [](const T& a, const T& b, T& c) { c = a - b; });
		return out;
	}

	Derived operator-() const {
		Derived out = this->clone();
		internal::TensorIMPL<T>::apply(out.t_, [](T& a) { a = -a; });
		return out;
	}


	Derived operator+(const Scalar<T>& s) const {
		Derived out(t_.dim(), t_.shape(), t_.allocator());
		internal::TensorIMPL<T>::apply(this->t_, out.t_,
		                               [&](const T& a, T& b) { b = a + s.get(); });
		return out;
	}

	Derived operator-(const Scalar<T>& s) const {
		Derived out(t_.dim(), t_.shape(), t_.allocator());
		internal::TensorIMPL<T>::apply(this->t_, out.t_,
		                               [&](const T& a, T& b) { b = a - s.get(); });
		return out;
	}

	Derived operator*(const Scalar<T>& s) const {
		Derived out(t_.dim(), t_.shape(), t_.allocator());
		internal::TensorIMPL<T>::apply(this->t_, out.t_,
		                               [&](const T& a, T& b) { b = a * s.get(); });
		return out;
	}


	Derived operator+=(const TensorWrapper& other) {
		internal::TensorIMPL<T>::apply(other.t_, this->t_, [](const T& a, T& b) { b += a; });
		return Derived(this->t_);
	}

	Derived operator-=(const TensorWrapper& other) {
		internal::TensorIMPL<T>::apply(other.t_, this->t_, [](const T& a, T& b) { b -= a; });
		return Derived(this->t_);
	}

	Derived operator+=(const Scalar<T>& s) {
		internal::TensorIMPL<T>::apply(this->t_, [&](T& a) { a += s.get(); });
		return Derived(this->t_);
	}

	Derived operator-=(const Scalar<T>& s) {
		internal::TensorIMPL<T>::apply(this->t_, [&](T& a) { a -= s.get(); });
		return Derived(this->t_);
	}

	Derived operator*=(const Scalar<T>& s) {
		internal::TensorIMPL<T>::apply(this->t_, [&](T& a) { a *= s.get(); });
		return Derived(this->t_);
	}

	// sets everything to a given value
	void setAll(const T& s) {
		internal::TensorIMPL<T>::apply(this->t_, [&](T& a) { a = s; });
	}

	// sums everything
	Scalar<T> sum() {
		Scalar<T> s = 0;
		internal::TensorIMPL<T>::apply(this->t_, [&](const T& a) { s.get() += a; });
		return s;
	}
};


template <typename Derived, typename T>
std::ostream& operator<<(std::ostream& os, const TensorWrapper<Derived, T>& t) {
	os << t.tensor_();
	return os;
}

}; // namespace internal

// # Tensor =============================================================

template <typename T>
class Tensor : public internal::TensorWrapper<Tensor<T>, T> {
	using Base = internal::TensorWrapper<Tensor<T>, T>;

  public:
	using Base::Base;

	// un-nests a nested std::vector to a Tensor
	// has to be right shape
	// the device can only be the CPU
	template <class NestedVector>
	static Tensor<T> fromSTDVec(const std::vector<NestedVector>& v) {
		Device device = CPU;
		uint8_t currDim = 0;
		std::array<uint64_t, internal::MAX_DIM> shape;

		getSTDVecShape(v, shape.data(), currDim);

		internal::TensorIMPL<T> t(currDim, shape.data(), device);

		uint64_t offset = 0;
		falttenSTDVec(v, t.data(), offset);
		return Tensor<T>(t);
	}


  private:
	// base case
	template <class K>
	static void getSTDVecShape(const K& k, uint64_t* const shape, uint8_t& currDim) {
		return;
	}

	template <class K>
	static void getSTDVecShape(const std::vector<K>& v, uint64_t* const shape, uint8_t& currDim) {
		shape[currDim++] = v.size();
		TZ_CHECK(currDim > internal::MAX_DIM, "tensor dimension exceeds MAX_DIMS");
		if (v.empty())
			return;
		getSTDVecShape(v[0], shape, currDim);
	}
	// base case
	template <class K>
	static void falttenSTDVec(const K& k, T* dst, uint64_t& offset) {
		dst[offset++] = static_cast<T>(k);
	}

	template <class K>
	static void falttenSTDVec(const std::vector<K>& v, T* dst, uint64_t& offset) {
		for (const auto& i : v)
			falttenSTDVec(i, dst, offset);
	}
};

// # Scalar =====================================================================

template <typename T>
class Scalar : public internal::TensorWrapper<Scalar<T>, T> {
	using Base = internal::TensorWrapper<Scalar<T>, T>;

  public:
	using Base::Base;

	// # constructors ---------------

	// standard constructor with tensor
	Scalar(internal::TensorIMPL<T>& t) : Base(t) {
		TZ_CHECK(this->t_.dim() != 0, "not a Scalar in the TZ::Scalar");
	}

	// standard constructor with tensor
	Scalar(Tensor<T>& t) : Base(t.tensor_()) {
		TZ_CHECK(this->t_.dim() != 0, "not a Scalar in the TZ::Scalar");
	}

	// standard constructor with tensor
	Scalar(internal::TensorIMPL<T>&& t) : Base(std::move(t)) {
		TZ_CHECK(this->t_.dim() != 0, "not a Scalar in the TZ::Scalar");
	}

	// standard constructor with a T class
	Scalar(const T& val, Device device = CPU) {
		set(val, device);
	}

	// # methods --------------------

	// makes a new scalar with the class T
	void set(const T& val, Device device = CPU) {
		this->t_.set(0, nullptr, device);
		TZ_CHECK(this->t_.dim() != 0, "not a Scalar in the TZ::Scalar");
		this->t_.get() = val;
	}

	// returns the value
	T& get() {
		return this->t_.get();
	}

	// returns the value
	const T& get() const {
		return this->t_.get();
	}

	Scalar& operator=(const T& val) {
		this->t_.get() = val;
		return *this;
	}

	Scalar& operator=(const Scalar& other) {
		this->t_.get() = other.get();
		return *this;
	}

	operator T() const {
		return this->t_.get();
	}
};

// # Vector =====================================================================

template <typename T>
class Vector : public internal::TensorWrapper<Vector<T>, T> {
	using Base = internal::TensorWrapper<Vector<T>, T>;

  public:
	using Base::Base;

	// # constructors ---------------

	// standard constructor with tensor
	Vector(const internal::TensorIMPL<T>& t) : Base(t) {
		TZ_CHECK(this->t_.dim() != 1, "not a Vector in the TZ::Vector");
	}

	// standard constructor with tensor
	Vector(const Tensor<T>& t) : Base(t.tensor_()) {
		TZ_CHECK(this->t_.dim() != 1, "not a Vector in the TZ::Vector");
	}

	// standard constructor with tensor
	Vector(internal::TensorIMPL<T>&& t) : Base(std::move(t)) {
		TZ_CHECK(this->t_.dim() != 1, "not a Vector in the TZ::Vector");
	}

	// standard constructor with size of the Vector
	Vector(const uint64_t size, Device device = CPU) {
		set(size, device);
	}

	// # methods --------------------

	// standard set function
	void set(const uint64_t size, Device device = CPU) {
		this->t_.set(1, &size, device);
		TZ_CHECK(this->t_.dim() != 1, "not a Vector in the TZ::Vector");
	}

	// returns the value at the given index
	T& at(const uint64_t idx) {
		TZ_CHECK(idx >= size(), "index out of bounds in matrix");
		return this->t_.data()[idx * this->t_.strides()[0]];
	}

	// returns the value at the given index
	const T& at(const uint64_t idx) const {
		TZ_CHECK(idx >= size(), "index out of bounds in matrix");
		return this->t_.data()[idx * this->t_.strides()[0]];
	}

	// returns the scalar at the given index
	Scalar<T> operator[](uint64_t idx) const {
		return Scalar<T>(this->t_[idx]);
	}

	// returns the size of the Vector
	uint64_t size() const {
		return this->shape()[0];
	}
};

// # Matrix =====================================================================

template <typename T>
class Matrix : public internal::TensorWrapper<Matrix<T>, T> {
	using Base = internal::TensorWrapper<Matrix<T>, T>;

  public:
	using Base::Base;

	// # constructors ---------------

	// standard constructor with tensor
	Matrix(const internal::TensorIMPL<T>& t) : Base(t) {
		TZ_CHECK(this->t_.dim() != 2, "not a Matrix in the TZ::Matrix");
	}

	// standard constructor with tensor
	Matrix(const Tensor<T>& t) : Base(t.tensor_()) {
		TZ_CHECK(this->t_.dim() != 2, "not a Matrix in the TZ::Matrix");
	}

	// standard constructor with tensor
	Matrix(internal::TensorIMPL<T>&& t) : Base(std::move(t)) {
		TZ_CHECK(this->t_.dim() != 2, "not a Matrix in the TZ::Matrix");
	}

	// standard constructor with the size of the rows, and columns
	Matrix(const uint64_t rows, const uint64_t cols, Device device = CPU) {
		set(rows, cols, device);
	}

	// # methods --------------------

	// standard set function with the size of the rows, and columns
	void set(const uint64_t rows, const uint64_t cols, Device device = CPU) {
		std::array<uint64_t, 2> shape = {rows, cols};
		this->t_.set(2, shape.data(), device);
		TZ_CHECK(this->t_.dim() != 2, "not a Matrix in the TZ::Matrix");
	}

	// returns the value at the given index
	T& at(const uint64_t i, const uint64_t j) {
		TZ_CHECK(i >= rows() || j >= cols(), "index out of bounds in matrix");
		return this->t_.data()[i * this->t_.strides()[0] + j * this->t_.strides()[1]];
	}

	// returns the value at the given index
	const T& at(const uint64_t i, const uint64_t j) const {
		TZ_CHECK(i >= rows() || j >= cols(), "index out of bounds in matrix");
		return this->t_.data()[i * this->t_.strides()[0] + j * this->t_.strides()[1]];
	}

	// returns the i'th row as a Vector
	Vector<T> operator[](uint64_t idx) const {
		return Vector<T>(this->t_[idx]);
	}

	// get the number of elements in the Matrix
	uint64_t size() const {
		return this->t_.size();
	}

	// get the number of rows in the Matrix
	uint64_t rows() const {
		return this->t_.shape()[0];
	}

	// get the number of collumns in the Matrix
	uint64_t cols() const {
		return this->t_.shape()[1];
	}

	// returns the i'th row
	Vector<T> row(const uint64_t i) {
		TZ_CHECK(i >= this->t_.shape()[0], "row index out of bounds");

		return Vector<T>(this->t_[i]);
	}

	// returns the i'th row
	const Vector<T> row(const uint64_t i) const {
		TZ_CHECK(i >= this->t_.shape()[0], "row index out of bounds");

		return Vector<T>(this->t_[i]);
	}

	// returns the j'th collumn
	Vector<T> col(const uint64_t j) {
		TZ_CHECK(j >= this->t_.shape()[1], "column index out of bounds");

		return Vector<T>(internal::TensorIMPL<T>(1, &this->t_.shape()[0], &this->t_.strides()[0],
		                                         j * this->t_.strides()[1] + this->t_.offset(),
		                                         this->t_.buffer()));
	}

	// returns the j'th collumn
	const Vector<T> col(const uint64_t j) const {
		TZ_CHECK(j >= this->t_.shape()[1], "column index out of bounds");

		return Vector<T>(internal::TensorIMPL<T>(1, &this->t_.shape()[0], &this->t_.strides()[0],
		                                         j * this->t_.strides()[1] + this->t_.offset(),
		                                         this->t_.buffer()));
	}

	// swap two rows, given by the indexes
	void swapRow(uint64_t i, uint64_t j) {
		TZ_CHECK(i >= rows() || j >= rows(), "out of bounds index in swapRow");

		if (i == j)
			return;

		for (uint64_t c = 0; c < cols(); c++)
			std::swap(at(i, c), at(j, c));
	}

	// swap two collumns, given by the indexes
	void swapCol(uint64_t i, uint64_t j) {
		TZ_CHECK(i >= cols() || j >= cols(), "out of bounds index in swapCol");

		if (i == j)
			return;

		for (uint64_t c = 0; c < rows(); c++)
			std::swap(at(c, i), at(c, i));
	}
};

} // namespace TZ
