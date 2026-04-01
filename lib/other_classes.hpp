#pragma once

namespace TZ {

template <typename T>
class Scalar;

template <typename T>
class Vector;

template <typename T>
class Matrix;

//& _tensorWrapper =============================================================

template <typename Derived, typename T>
class _tensorWrapper {
  protected:
	Tensor<T> t_;

  public:
	//& seters ===========================================================================

	_tensorWrapper() = default;

	_tensorWrapper(const Tensor<T>& t) : t_(t) {}

	_tensorWrapper(Tensor<T>&& t) : t_(std::move(t)) {}

	_tensorWrapper(const std::vector<uint64_t>& shape,
	               mem::Allocator& allocator = config::defaultAllocator())
	    : t_(shape, allocator) {}

	_tensorWrapper(const uint8_t dim, const uint64_t* shape,
	               mem::Allocator& allocator = config::defaultAllocator())
	    : t_(dim, shape, allocator) {}


	_tensorWrapper(const _tensorWrapper&) = default;

	_tensorWrapper& operator=(const _tensorWrapper&) = default;

	_tensorWrapper(_tensorWrapper&&) = default;

	_tensorWrapper& operator=(_tensorWrapper&&) = default;

	//& metadata geters ===========================================================================

	uint64_t dim() const {
		return t_.dim();
	}

	const uint64_t* shape() const {
		return t_.shape();
	}

	T* data() {
		return t_.data();
	}

	const T* data() const {
		return t_.data();
	}

	bool empty() const {
		return t_.empty();
	}

	bool dense() const {
		return t_.dense();
	}

	//& geters ===========================================================================

	// returns the inner tensor
	Tensor<T>& tensor() {
		return t_;
	}

	// returns the inner tensor
	const Tensor<T>& tensor() const {
		return t_;
	}

	// clones the object
	Derived clone() const {
		return Derived(this->t_.clone());
	}

	//& operations ===========================================================================

	Derived operator+(const _tensorWrapper& other) const {
		Derived out(t_.dim(), t_.shape(), t_.allocator());
		Tensor<T>::apply(this->t_, other.t_, out.t_,
		                 [](const T& a, const T& b, T& c) { c = a + b; });
		return out;
	}

	Derived operator-(const _tensorWrapper& other) const {
		Derived out(t_.dim(), t_.shape(), t_.allocator());
		Tensor<T>::apply(this->t_, other.t_, out.t_,
		                 [](const T& a, const T& b, T& c) { c = a - b; });
		return out;
	}

	Derived operator-() const {
		Derived out = this->clone();
		Tensor<T>::apply(out.t_, [](T& a) { a = -a; });
		return out;
	}


	Derived operator+(const Scalar<T>& s) const {
		Derived out(t_.dim(), t_.shape(), t_.allocator());
		Tensor<T>::apply(this->t_, out.t_, [&](const T& a, T& b) { b = a + s.get(); });
		return out;
	}

	Derived operator-(const Scalar<T>& s) const {
		Derived out(t_.dim(), t_.shape(), t_.allocator());
		Tensor<T>::apply(this->t_, out.t_, [&](const T& a, T& b) { b = a - s.get(); });
		return out;
	}

	Derived operator*(const Scalar<T>& s) const {
		Derived out(t_.dim(), t_.shape(), t_.allocator());
		Tensor<T>::apply(this->t_, out.t_, [&](const T& a, T& b) { b = a * s.get(); });
		return out;
	}


	Derived operator+=(const _tensorWrapper& other) {
		Tensor<T>::apply(other.t_, this->t_, [](const T& a, T& b) { b += a; });
		return Derived(this->t_);
	}

	Derived operator-=(const _tensorWrapper& other) {
		Tensor<T>::apply(other.t_, this->t_, [](const T& a, T& b) { b -= a; });
		return Derived(this->t_);
	}

	Derived operator+=(const Scalar<T>& s) {
		Tensor<T>::apply(this->t_, [&](T& a) { a += s.get(); });
		return Derived(this->t_);
	}

	Derived operator-=(const Scalar<T>& s) {
		Tensor<T>::apply(this->t_, [&](T& a) { a -= s.get(); });
		return Derived(this->t_);
	}

	Derived operator*=(const Scalar<T>& s) {
		Tensor<T>::apply(this->t_, [&](T& a) { a *= s.get(); });
		return Derived(this->t_);
	}

	// sets everything to a given value
	void setAll(const T& s) {
		Tensor<T>::apply(this->t_, [&](T& a) { a = s; });
	}

	// sums everything
	Scalar<T> sum() {
		Scalar<T> s = 0;
		Tensor<T>::apply(this->t_, [&](const T& a) { s.get() += a; });
		return s;
	}
};

//& Scalar =====================================================================

template <typename T>
class Scalar : public _tensorWrapper<Scalar<T>, T> {
	using Base = _tensorWrapper<Scalar<T>, T>;

  public:
	using Base::Base;

	//& constructors ---------------

	// standard constructor with tensor
	Scalar(const Tensor<T>& t) : Base(t) {
		TZ_CHECK(this->t_.dim() != 0, "not a Scalar in the TZ::Scalar");
	}

	// standard constructor with tensor
	Scalar(Tensor<T>&& t) : Base(std::move(t)) {
		TZ_CHECK(this->t_.dim() != 0, "not a Scalar in the TZ::Scalar");
	}

	// standard constructor with a T class
	Scalar(const T& val, mem::Allocator& allocator = config::defaultAllocator()) {
		set(val, allocator);
	}

	//& methods --------------------

	// makes a new scalar with the class T
	void set(const T& val, mem::Allocator& allocator = config::defaultAllocator()) {
		this->t_.set(0, nullptr, allocator);
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

//& Vector =====================================================================

template <typename T>
class Vector : public _tensorWrapper<Vector<T>, T> {
	using Base = _tensorWrapper<Vector<T>, T>;

  public:
	using Base::Base;

	//& constructors ---------------

	// standard constructor with tensor
	Vector(const Tensor<T>& t) : Base(t) {
		TZ_CHECK(this->t_.dim() != 1, "not a Vector in the TZ::Vector");
	}

	// standard constructor with tensor
	Vector(Tensor<T>&& t) : Base(std::move(t)) {
		TZ_CHECK(this->t_.dim() != 1, "not a Vector in the TZ::Vector");
	}

	// standard constructor with size of the Vector
	Vector(const uint64_t size, mem::Allocator& allocator = config::defaultAllocator()) {
		set(size, allocator);
	}

	//& methods --------------------

	// standard set function
	void set(const uint64_t size, mem::Allocator& allocator = config::defaultAllocator()) {
		this->t_.set(1, &size, allocator);
		TZ_CHECK(this->t_.dim() != 1, "not a Vector in the TZ::Vector");
	}

	// returns the value at the given index
	T& at(const uint64_t idx) {
		TZ_CHECK(idx >= size(), "index out of bounds in matrix");
		return this->t_.data()[idx * this->t_._strides()[0]];
	}

	// returns the value at the given index
	const T& at(const uint64_t idx) const {
		TZ_CHECK(idx >= size(), "index out of bounds in matrix");
		return this->t_.data()[idx * this->t_._strides()[0]];
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

//& Matrix =====================================================================

template <typename T>
class Matrix : public _tensorWrapper<Matrix<T>, T> {
	using Base = _tensorWrapper<Matrix<T>, T>;

  public:
	using Base::Base;

	//& constructors ---------------

	// standard constructor with tensor
	Matrix(const Tensor<T>& t) : Base(t) {
		TZ_CHECK(this->t_.dim() != 2, "not a Matrix in the TZ::Matrix");
	}

	// standard constructor with tensor
	Matrix(Tensor<T>&& t) : Base(std::move(t)) {
		TZ_CHECK(this->t_.dim() != 2, "not a Matrix in the TZ::Matrix");
	}

	// standard constructor with the size of the rows, and columns
	Matrix(const uint64_t rows, const uint64_t cols,
	       mem::Allocator& allocator = config::defaultAllocator()) {
		set(rows, cols, allocator);
	}

	//& methods --------------------

	// standard set function with the size of the rows, and columns
	void set(const uint64_t rows, const uint64_t cols,
	         mem::Allocator& allocator = config::defaultAllocator()) {
		std::array<uint64_t, 2> shape = {rows, cols};
		this->t_.set(2, shape.data(), allocator);
		TZ_CHECK(this->t_.dim() != 2, "not a Matrix in the TZ::Matrix");
	}

	// returns the value at the given index
	T& at(const uint64_t i, const uint64_t j) {
		TZ_CHECK(i >= rows() || j >= cols(), "index out of bounds in matrix");
		return this->t_.data()[i * this->t_._strides()[0] + j * this->t_._strides()[1]];
	}

	// returns the value at the given index
	const T& at(const uint64_t i, const uint64_t j) const {
		TZ_CHECK(i >= rows() || j >= cols(), "index out of bounds in matrix");
		return this->t_.data()[i * this->t_._strides()[0] + j * this->t_._strides()[1]];
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

		return Vector<T>(Tensor<T>(1, &this->t_.shape()[0], &this->t_._strides()[0],
		                           j * this->t_._strides()[1] + this->t_._offset(),
		                           this->t_._buffer()));
	}

	// returns the j'th collumn
	const Vector<T> col(const uint64_t j) const {
		TZ_CHECK(j >= this->t_.shape()[1], "column index out of bounds");

		return Vector<T>(Tensor<T>(1, &this->t_.shape()[0], &this->t_._strides()[0],
		                           j * this->t_._strides()[1] + this->t_._offset(),
		                           this->t_._buffer()));
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