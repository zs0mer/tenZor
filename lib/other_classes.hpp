#pragma once

namespace TZ {

template <typename T>
class Scalar;

template <typename T>
class Vector;

template <typename T>
class Matrix;


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
	               mem::Allocator& allocator = DEFAULT_ALLOCATOR)
	    : t_(shape, allocator) {}

	_tensorWrapper(const uint8_t dim, const uint64_t* shape,
	               mem::Allocator& allocator = DEFAULT_ALLOCATOR)
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

	uint64_t size() const {
		return t_.size();
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

	//& geters ===========================================================================

	Tensor<T>& tensor() {
		return t_;
	}

	const Tensor<T>& tensor() const {
		return t_;
	}

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

	void setAll(const T& s) {
		Tensor<T>::apply(this->t_, [&](T& a) { a = s; });
	}
};

template <typename T>
class Scalar : public _tensorWrapper<Scalar<T>, T> {
	using Base = _tensorWrapper<Scalar<T>, T>;

  public:
	using Base::Base;

	Scalar(const Tensor<T>& t) : Base(t) {
		_CHECK(this->t_.dim() != 0, "not a Scalar in the TZ::Scalar");
	}

	Scalar(Tensor<T>&& t) : Base(std::move(t)) {
		_CHECK(this->t_.dim() != 0, "not a Scalar in the TZ::Scalar");
	}

	Scalar(const T& val, mem::Allocator& allocator = DEFAULT_ALLOCATOR) {
		set(val, allocator);
	}


	void set(const T& val, mem::Allocator& allocator = DEFAULT_ALLOCATOR) {
		this->t_.set(0, nullptr, allocator);
		_CHECK(this->t_.dim() != 0, "not a Scalar in the TZ::Scalar");
		this->t_.get() = val;
	}

	T& get() {
		return this->t_.get();
	}

	const T& get() const {
		return this->t_.get();
	}
};

template <typename T>
class Vector : public _tensorWrapper<Vector<T>, T> {
	using Base = _tensorWrapper<Vector<T>, T>;

  public:
	using Base::Base;

	Vector(const Tensor<T>& t) : Base(t) {
		_CHECK(this->t_.dim() != 1, "not a Vector in the TZ::Vector");
	}

	Vector(Tensor<T>&& t) : Base(std::move(t)) {
		_CHECK(this->t_.dim() != 1, "not a Vector in the TZ::Vector");
	}

	Vector(const uint64_t size, mem::Allocator& allocator = DEFAULT_ALLOCATOR) {
		set(size, allocator);
	}


	void set(const uint64_t size, mem::Allocator& allocator = DEFAULT_ALLOCATOR) {
		this->t_.set(1, &size, allocator);
		_CHECK(this->t_.dim() != 1, "not a Vector in the TZ::Vector");
	}

	T& at(const uint64_t idx) {
		return this->t_.data()[idx * this->t_.strides()[0]];
	}

	const T& at(const uint64_t idx) const {
		return this->t_.data()[idx * this->t_.strides()[0]];
	}

	Scalar<T> operator[](uint64_t idx) const {
		return Scalar<T>(this->t_[idx]);
	}
};

template <typename T>
class Matrix : public _tensorWrapper<Matrix<T>, T> {
	using Base = _tensorWrapper<Matrix<T>, T>;

  public:
	using Base::Base;

	Matrix(const Tensor<T>& t) : Base(t) {
		_CHECK(this->t_.dim() != 2, "not a Matrix in the TZ::Matrix");
	}

	Matrix(Tensor<T>&& t) : Base(std::move(t)) {
		_CHECK(this->t_.dim() != 2, "not a Matrix in the TZ::Matrix");
	}

	Matrix(const uint64_t rows, const uint64_t cols,
	       mem::Allocator& allocator = DEFAULT_ALLOCATOR) {
		set(rows, cols, allocator);
	}


	void set(const uint64_t rows, const uint64_t cols,
	         mem::Allocator& allocator = DEFAULT_ALLOCATOR) {
		std::array<uint64_t, 2> shape = {rows, cols};
		this->t_.set(2, shape.data(), allocator);
		_CHECK(this->t_.dim() != 2, "not a Matrix in the TZ::Matrix");
	}

	T& at(const uint64_t i, const uint64_t j) {
		return this->t_.data()[i * this->t_.strides()[0] + j * this->t_.strides()[1]];
	}

	const T& at(const uint64_t i, const uint64_t j) const {
		return this->t_.data()[i * this->t_.strides()[0] + j * this->t_.strides()[1]];
	}

	Vector<T> operator[](uint64_t idx) const {
		return Vector<T>(this->t_[idx]);
	}
};

} // namespace TZ