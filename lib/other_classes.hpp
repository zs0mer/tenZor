#pragma once

// TODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO

namespace TZ {

template <typename Derived, typename T>
class TensorWrapper {
  protected:
	Tensor<T> t_;

  public:
	//& seters ===========================================================================

	TensorWrapper();

	TensorWrapper(const Tensor<T>& t) : t_(t) {}

	TensorWrapper(Tensor<T>&& t) : t_(std::move(t)) {}

	TensorWrapper(const std::vector<uint64_t>& shape, const uint8_t alignment = DEFAULT_ALIGNMENT,
	              mem::Allocator& allocator = mem::Salloc::instance())
	    : t_(shape, alignment, allocator) {}


	TensorWrapper(const TensorWrapper&) = default;

	TensorWrapper& operator=(const TensorWrapper&) = default;

	TensorWrapper(TensorWrapper&&) = default;

	TensorWrapper& operator=(TensorWrapper&&) = default;

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

	Tensor<T> operator[](uint64_t idx) const {
		return t_[idx];
	}

	Tensor<T> clone() const {
		return t_.clone();
	}
};

template <typename T>
class Matrix : public TensorWrapper<Matrix<T>, T> {
	using Base = TensorWrapper<Matrix<T>, T>;

  public:
	using Base::Base;
};

template <typename T>
class Vector : public TensorWrapper<Vector<T>, T> {
	using Base = TensorWrapper<Vector<T>, T>;

  public:
	using Base::Base;
};

template <typename T>
class Scalar : public TensorWrapper<Scalar<T>, T> {
	using Base = TensorWrapper<Scalar<T>, T>;

  public:
	using Base::Base;
};

} // namespace TZ