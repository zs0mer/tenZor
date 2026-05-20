#pragma once

#include <cstdint>
#include <array>
#include <initializer_list>

#include "allocator.hpp"
#include "tensor.hpp"
#include "utils.hpp"

namespace tz {

template <class T>
class Scalar;

template <class T>
class Vector;

template <class T>
class Matrix;

template <class T>
class Tensor;

namespace impl {

// # tensorWrapper =============================================================

template <class Derived, class T>
class TensorWrapper {
  protected:
	impl::TensorIMPL<T> t_;

  public:
	// # seters ===========================================================================

	TensorWrapper() = default;

	TensorWrapper(const impl::TensorIMPL<T>& t) : t_(t) {}

	TensorWrapper(impl::TensorIMPL<T>&& t) : t_(std::move(t)) {}

	TensorWrapper(const std::initializer_list<uint64_t>& shape, Device device = CPU)
	    : t_(shape, device) {}

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

	uint64_t size() const {
		return t_.size();
	}

	bool empty() const {
		return t_.empty();
	}

	Device device() const {
		return t_.device();
	}

	// # geters ===========================================================================

	// returns the inner tensor
	const impl::TensorIMPL<T>& tensor_() const {
		return t_;
	}

	// returns the inner tensor
	impl::TensorIMPL<T>& tensor_() {
		return t_;
	}

	// clones the object
	Derived clone() const {
		return Derived(this->t_.clone());
	}

	Tensor<T> operator[](uint64_t idx) const {
		return Tensor<T>(this->t_[idx]);
	}

	void copyDataFrom(const TensorWrapper& other) {
		this->t_.copyDataFrom(other.tensor_());
	}

	// # operations ===========================================================================

	// calls func(this[i]) for all elements of the tensor
	template <class Func, class IsGPUAvalable = CPUOnly,
	          class = std::enable_if_t<IsPolicy<IsGPUAvalable>::value>>
	void apply(Func func, IsGPUAvalable constraint = {}) {
		impl::TensorIMPL<T>::apply(this->t_, func, constraint);
	}

	// calls func(const other[i], this[i]) for all elements of the tensor
	template <class Func, class IsGPUAvalable = CPUOnly,
	          class = std::enable_if_t<IsPolicy<IsGPUAvalable>::value>>
	void apply(const impl::TensorWrapper<Derived, T>& other, Func func,
	           IsGPUAvalable constraint = {}) {
		impl::TensorIMPL<T>::apply(other.t_, this->t_, func, constraint);
	}

	// calls func(const a[i], const b[i], this[i]) for all elements of the tensor
	template <class Func, class IsGPUAvalable = CPUOnly,
	          class = std::enable_if_t<IsPolicy<IsGPUAvalable>::value>>
	void apply(const impl::TensorWrapper<Derived, T>& a, const impl::TensorWrapper<Derived, T>& b,
	           Func func, IsGPUAvalable constraint = {}) {
		impl::TensorIMPL<T>::apply(a.t_, b.t_, this->t_, func, constraint);
	}


	Derived operator+(const TensorWrapper& other) const {
		Derived out(t_.dim(), t_.shape(), device());
		impl::TensorIMPL<T>::apply(this->t_, other.t_, out.t_, impl::Add<T>{}, AnyDevice{});
		return out;
	}

	Derived operator-(const TensorWrapper& other) const {
		Derived out(t_.dim(), t_.shape(), device());
		impl::TensorIMPL<T>::apply(this->t_, other.t_, out.t_, impl::Subtract<T>{}, AnyDevice{});
		return out;
	}

	Derived operator*(const TensorWrapper& other) const {
		Derived out(t_.dim(), t_.shape(), device());
		impl::TensorIMPL<T>::apply(this->t_, other.t_, out.t_, impl::Multiply<T>{}, AnyDevice{});
		return out;
	}

	Derived operator-() const {
		Derived out = this->clone();
		impl::TensorIMPL<T>::apply(out.t_, impl::Negate<T>{}, AnyDevice{});
		return out;
	}


	Derived operator+(const Scalar<T>& s) const {
		Derived out(t_.dim(), t_.shape(), device());
		impl::TensorIMPL<T>::apply(this->t_, out.t_, impl::AddScalar<T>(s.get()), AnyDevice{});
		return out;
	}

	Derived operator-(const Scalar<T>& s) const {
		Derived out(t_.dim(), t_.shape(), device());
		impl::TensorIMPL<T>::apply(this->t_, out.t_, impl::SubtractScalar<T>(s.get()), AnyDevice{});
		return out;
	}

	Derived operator*(const Scalar<T>& s) const {
		Derived out(t_.dim(), t_.shape(), device());
		impl::TensorIMPL<T>::apply(this->t_, out.t_, impl::MultiplyScalar<T>(s.get()), AnyDevice{});
		return out;
	}

	Derived operator/(const Scalar<T>& s) const {
		Derived out(t_.dim(), t_.shape(), device());
		impl::TensorIMPL<T>::apply(this->t_, out.t_, impl::DivideScalar<T>(s.get()), AnyDevice{});
		return out;
	}

	Derived& operator+=(const TensorWrapper& other) {
		impl::TensorIMPL<T>::apply(this->t_, other.t_, this->t_, impl::Add<T>{}, AnyDevice{});
		return static_cast<Derived&>(*this);
	}

	Derived& operator-=(const TensorWrapper& other) {
		impl::TensorIMPL<T>::apply(this->t_, other.t_, this->t_, impl::Subtract<T>{}, AnyDevice{});
		return static_cast<Derived&>(*this);
	}

	Derived& operator*=(const TensorWrapper& other) {
		impl::TensorIMPL<T>::apply(this->t_, other.t_, this->t_, impl::Multiply<T>{}, AnyDevice{});
		return static_cast<Derived&>(*this);
	}

	Derived& operator+=(const Scalar<T>& s) {
		impl::TensorIMPL<T>::apply(this->t_, this->t_, impl::AddScalar<T>(s.get()), AnyDevice{});
		return static_cast<Derived&>(*this);
	}

	Derived& operator-=(const Scalar<T>& s) {
		impl::TensorIMPL<T>::apply(this->t_, this->t_, impl::SubtractScalar<T>(s.get()),
		                           AnyDevice{});
		return static_cast<Derived&>(*this);
	}

	Derived& operator*=(const Scalar<T>& s) {
		impl::TensorIMPL<T>::apply(this->t_, this->t_, impl::MultiplyScalar<T>(s.get()),
		                           AnyDevice{});
		return static_cast<Derived&>(*this);
	}

	Derived& operator/=(const Scalar<T>& s) {
		impl::TensorIMPL<T>::apply(this->t_, this->t_, impl::DivideScalar<T>(s.get()), AnyDevice{});
		return static_cast<Derived&>(*this);
	}

	// sets everything to a given value
	void setAll(const T& s) {
		impl::TensorIMPL<T>::apply(this->t_, impl::Set<T>(s), AnyDevice{});
	}

	// sums everything
	Scalar<T> sum() const {
		Scalar<T> s = TensorIMPL<T>(0, nullptr, device());
		s.setAll(0);
		impl::TensorIMPL<T>::apply(this->t_, impl::Sum<T>(s.tensor_().data()), AnyDevice{});
		return s;
	}

	Derived copyTo(Device device) {
		return Derived(this->t_.copyTo(device));
	}
};


template <class Derived, class T>
std::ostream& operator<<(std::ostream& os, const TensorWrapper<Derived, T>& t) {
	os << t.tensor_();
	return os;
}

}; // namespace impl

// # Tensor =============================================================

template <class T>
class Tensor : public impl::TensorWrapper<Tensor<T>, T> {
	using Base = impl::TensorWrapper<Tensor<T>, T>;

  public:
	using Base::Base;


	// un-nests a nested std::vector to a Tensor
	// has to be right shape
	// the device can only be the CPU
	template <class NestedVector>
	static Tensor<T> fromSTDVec(const std::vector<NestedVector>& v) {
		uint8_t currDim = 0;
		std::array<uint64_t, impl::MAX_DIM> shape;

		getSTDVecShape(v, shape.data(), currDim);

		impl::TensorIMPL<T> t(currDim, shape.data(), CPU);

		uint64_t offset = 0;
		flattenSTDVec(v, t.data(), offset);
		return Tensor<T>(t);
	}

	void broadcastTo(const std::initializer_list<uint64_t>& targetShape) {
		this->t_ = this->t_.broadcast(targetShape);
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
		TZ_CHECK(currDim <= impl::MAX_DIM, "tensor dimension exceeds MAX_DIMS");
		if (v.empty())
			return;
		getSTDVecShape(v[0], shape, currDim);
	}
	// base case
	template <class K>
	static void flattenSTDVec(const K& k, T* dst, uint64_t& offset) {
		dst[offset++] = static_cast<T>(k);
	}

	template <class K>
	static void flattenSTDVec(const std::vector<K>& v, T* dst, uint64_t& offset) {
		for (const auto& i : v)
			flattenSTDVec(i, dst, offset);
	}
};

// # Scalar =====================================================================

template <class T>
class Scalar : public impl::TensorWrapper<Scalar<T>, T> {
	using Base = impl::TensorWrapper<Scalar<T>, T>;

  public:
	// # constructors ---------------

	Scalar() = default;

	// standard constructor with tensor
	Scalar(impl::TensorIMPL<T>& t) : Base(t) {
		TZ_CHECK(this->t_.dim() == 0, "not a Scalar in the TZ::Scalar");
	}

	// standard constructor with tensor
	Scalar(Tensor<T>& t) : Base(t.tensor_()) {
		TZ_CHECK(this->t_.dim() == 0, "not a Scalar in the TZ::Scalar");
	}

	// standard constructor with tensor
	Scalar(impl::TensorIMPL<T>&& t) : Base(std::move(t)) {
		TZ_CHECK(this->t_.dim() == 0, "not a Scalar in the TZ::Scalar");
	}

	// standard constructor with a T class
	Scalar(const T& val) {
		set(val, CPU);
	}

	Scalar(const uint8_t dim, const uint64_t* shape, Device device = CPU)
	    : Base(dim, shape, device) {
		TZ_CHECK(dim == 0, "not a Scalar in the TZ::Scalar");
	}

	// # methods --------------------

	// makes a new scalar with the class T
	void set(const T& val, Device device = CPU) {
		this->t_.set(0, nullptr, device);
		TZ_CHECK(this->t_.dim() == 0, "not a Scalar in the TZ::Scalar");
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

	Vector<T> broadcast(uint64_t size) const {
		std::array<uint64_t, 1> shape = {size};
		std::array<uint64_t, 1> strides = {0};
		return Vector<T>(impl::TensorIMPL<T>(1, shape.data(), strides.data(), this->t_.offset(),
		                                     this->t_.buffer()));
	}

	Matrix<T> broadcast(uint64_t rows, uint64_t cols) const {
		std::array<uint64_t, 2> shape = {rows, cols};
		std::array<uint64_t, 2> strides = {0, 0};
		return Matrix<T>(impl::TensorIMPL<T>(2, shape.data(), strides.data(), this->t_.offset(),
		                                     this->t_.buffer()));
	}
};

// # Vector =====================================================================

template <class T>
class Vector : public impl::TensorWrapper<Vector<T>, T> {
	using Base = impl::TensorWrapper<Vector<T>, T>;

  public:
	// # constructors ---------------

	Vector() = default;

	// initializer list constructor
	Vector(const std::initializer_list<T>& t) : Base({t.size()}, CPU) {
		for (uint64_t i = 0; i < t.size(); i++)
			this->t_.data()[i * this->t_.strides()[0]] = t.begin()[i];
	}

	// standard constructor with tensor
	Vector(const impl::TensorIMPL<T>& t) : Base(t) {
		TZ_CHECK(this->t_.dim() == 1, "not a Vector in the TZ::Vector");
	}

	// standard constructor with tensor
	Vector(const Tensor<T>& t) : Base(t.tensor_()) {
		TZ_CHECK(this->t_.dim() == 1, "not a Vector in the TZ::Vector");
	}

	// standard constructor with tensor
	Vector(impl::TensorIMPL<T>&& t) : Base(std::move(t)) {
		TZ_CHECK(this->t_.dim() == 1, "not a Vector in the TZ::Vector");
	}

	// standard constructor with size of the Vector
	Vector(const uint64_t size, Device device = CPU) {
		set(size, device);
	}

	// initializer with matrix, the matrix has to be a column vector
	Vector(const Matrix<T>& t)
	    : Base(impl::TensorIMPL<T>(1, t.shape(), t.tensor_().strides(), t.tensor_().offset(),
	                               t.tensor_().buffer())) {
		TZ_CHECK(t.cols() == 1, "not a convertable Matrix in the TZ::Vector constructor");
	}

	Vector(const uint8_t dim, const uint64_t* shape, Device device = CPU)
	    : Base(dim, shape, device) {
		TZ_CHECK(dim == 1, "not a Scalar in the TZ::Vector");
	}

	// # methods --------------------

	// standard set function
	void set(const uint64_t size, Device device = CPU) {
		this->t_.set(1, &size, device);
		TZ_CHECK(this->t_.dim() == 1, "not a Vector in the TZ::Vector");
	}

	// returns the value at the given index
	T& at(const uint64_t idx) {
		TZ_CHECK(idx < size(), "index out of bounds in matrix");
		return this->t_.data()[idx * this->t_.strides()[0]];
	}

	// returns the value at the given index
	const T& at(const uint64_t idx) const {
		TZ_CHECK(idx < size(), "index out of bounds in matrix");
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

	// returns a transposed Matrix view of the Vector
	Matrix<T> transpose() const {
		uint64_t stride = this->t_.strides()[0];

		// this->size() * stride because we want the new tensor to be dense
		std::array<uint64_t, 2> strides = {this->size() * stride, stride};
		std::array<uint64_t, 2> shape = {1, this->size()};
		return Matrix<T>(impl::TensorIMPL<T>(2, shape.data(), strides.data(),
		                                     this->tensor_().offset(), this->tensor_().buffer()));
	}

	Matrix<T> broadcast(uint64_t rows, uint64_t cols) const {
		std::array<uint64_t, 2> shape = {rows, cols};
		std::array<uint64_t, 2> strides = {0, this->t_.strides()[0]};
		return Matrix<T>(impl::TensorIMPL<T>(2, shape.data(), strides.data(), this->t_.offset(),
		                                     this->t_.buffer()));
	}
};

// # Matrix =====================================================================

template <class T>
class Matrix : public impl::TensorWrapper<Matrix<T>, T> {
	using Base = impl::TensorWrapper<Matrix<T>, T>;

  public:
	// # constructors ---------------

	Matrix() = default;

	// initializer list constructor
	Matrix(const std::initializer_list<std::initializer_list<T>>& t)
	    : Base({t.size(), t.size() == 0 ? 0 : t.begin()[0].size()}, CPU) {
		for (uint64_t i = 0; i < t.size(); i++)
			for (uint64_t j = 0; j < t.begin()[i].size(); j++)
				this->t_.data()[i * this->t_.strides()[0] + j * this->t_.strides()[1]] =
				    t.begin()[i].begin()[j];
	}

	// standard constructor with tensor
	Matrix(const impl::TensorIMPL<T>& t) : Base(t) {
		TZ_CHECK(this->t_.dim() == 2, "not a Matrix in the TZ::Matrix");
	}

	// standard constructor with tensor
	Matrix(const Tensor<T>& t) : Base(t.tensor_()) {
		TZ_CHECK(this->t_.dim() == 2, "not a Matrix in the TZ::Matrix");
	}

	// standard constructor with tensor
	Matrix(impl::TensorIMPL<T>&& t) : Base(std::move(t)) {
		TZ_CHECK(this->t_.dim() == 2, "not a Matrix in the TZ::Matrix");
	}

	// standard constructor with the size of the rows, and columns
	Matrix(const uint64_t rows, const uint64_t cols, Device device = CPU) {
		set(rows, cols, device);
	}

	// initializer with a Vector
	Matrix(const Vector<T>& t) : Base(t.transpose().transpose().tensor_()) {}

	Matrix(const uint8_t dim, const uint64_t* shape, Device device = CPU)
	    : Base(dim, shape, device) {
		TZ_CHECK(dim == 2, "not a Scalar in the TZ::Matrix");
	}

	// # methods --------------------

	// standard set function with the size of the rows, and columns
	void set(const uint64_t rows, const uint64_t cols, Device device = CPU) {
		std::array<uint64_t, 2> shape = {rows, cols};
		this->t_.set(2, shape.data(), device);
		TZ_CHECK(this->t_.dim() == 2, "not a Matrix in the TZ::Matrix");
	}

	// returns the value at the given index
	T& at(const uint64_t i, const uint64_t j) {
		TZ_CHECK(i < rows() && j < cols(), "index out of bounds in matrix");
		return this->t_.data()[i * this->t_.strides()[0] + j * this->t_.strides()[1]];
	}

	// returns the value at the given index
	const T& at(const uint64_t i, const uint64_t j) const {
		TZ_CHECK(i < rows() && j < cols(), "index out of bounds in matrix");
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
		TZ_CHECK(i < this->t_.shape()[0], "row index out of bounds");

		return Vector<T>(this->t_[i]);
	}

	// returns the i'th row
	const Vector<T> row(const uint64_t i) const {
		TZ_CHECK(i < this->t_.shape()[0], "row index out of bounds");

		return Vector<T>(this->t_[i]);
	}

	// returns the j'th collumn
	Vector<T> col(const uint64_t j) {
		TZ_CHECK(j < this->t_.shape()[1], "column index out of bounds");

		return Vector<T>(impl::TensorIMPL<T>(1, &this->t_.shape()[0], &this->t_.strides()[0],
		                                     j * this->t_.strides()[1] + this->t_.offset(),
		                                     this->t_.buffer()));
	}

	// returns the j'th collumn
	const Vector<T> col(const uint64_t j) const {
		TZ_CHECK(j < this->t_.shape()[1], "column index out of bounds");

		return Vector<T>(impl::TensorIMPL<T>(1, &this->t_.shape()[0], &this->t_.strides()[0],
		                                     j * this->t_.strides()[1] + this->t_.offset(),
		                                     this->t_.buffer()));
	}

	// swap two rows, given by the indexes
	void swapRow(uint64_t i, uint64_t j) {
		TZ_CHECK(i < rows() && j < rows(), "out of bounds index in swapRow");

		if (i == j)
			return;

		for (uint64_t c = 0; c < cols(); c++)
			std::swap(at(i, c), at(j, c));
	}

	// swap two collumns, given by the indexes
	void swapCol(uint64_t i, uint64_t j) {
		TZ_CHECK(i < cols() && j < cols(), "out of bounds index in swapCol");

		if (i == j)
			return;

		for (uint64_t c = 0; c < rows(); c++)
			std::swap(at(c, i), at(c, j));
	}

	// returns the transposed Matrix
	Matrix<T> transpose() const {
		std::array<uint64_t, 2> strides = {this->tensor_().strides()[1],
		                                   this->tensor_().strides()[0]};
		std::array<uint64_t, 2> shape = {this->cols(), this->rows()};
		return Matrix<T>(impl::TensorIMPL<T>(2, shape.data(), strides.data(),
		                                     this->tensor_().offset(), this->tensor_().buffer()));
	}
};

} // namespace tz
