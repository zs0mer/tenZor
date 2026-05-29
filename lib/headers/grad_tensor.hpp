#pragma once
#include <cstdint>
#include <functional>
#include <vector>

#include "allocator.hpp"
#include "math_classes.hpp"


namespace tz {
namespace grad {

template <class T, class DataType>
struct GradOpNode {
	uint32_t inEdges = 0;
	std::function<void(const DataType&)> backwardFn;
};

template <class T>
class GradBase {
  public:
	virtual Tensor<T> tensor() const = 0;
	virtual Tensor<T> grad() const = 0;
	virtual void zeroGrad() const = 0;
};

template <class Derived, class DataType, class T>
class GradWrapper : public GradBase<T> {
  private:
	DataType data_;
	mutable DataType grad_;
	// this class owns *fromOp_
	mutable GradOpNode<T, DataType>* fromOp_ = nullptr;

  public:
	GradWrapper() = default;

	~GradWrapper() {
		clean();
	}


	GradWrapper(const GradWrapper& other)
	    : data_(other.data_), grad_(createSame(data_)), fromOp_(nullptr) {
		grad_.setAll(0);
	}

	GradWrapper& operator=(const GradWrapper& other) {
		clean();

		data_ = other.data_;
		grad_ = DataType(createSame(data_));
		fromOp_ = nullptr;

		grad_.setAll(0);

		return *this;
	}

	GradWrapper(GradWrapper&& other)
	    : data_(std::move(other.data_)), grad_(std::move(other.grad_)), fromOp_(other.fromOp_) {
		other.fromOp_ = nullptr;
	}

	GradWrapper& operator=(GradWrapper&& other) {
		clean();

		data_ = std::move(other.data_);
		grad_ = std::move(other.grad_);
		fromOp_ = other.fromOp_;

		other.fromOp_ = nullptr;
		return *this;
	}


	GradWrapper(const DataType& data) : data_(data), grad_(createSame(data_)), fromOp_(nullptr) {
		grad_.setAll(0);
	}

	GradWrapper& operator=(const DataType& other) {
		clean();

		data_ = other;
		grad_ = DataType(createSame(data_));
		fromOp_ = nullptr;

		grad_.setAll(0);

		return *this;
	}

	GradWrapper(DataType&& other)
	    : data_(std::move(other)), grad_(createSame(data_)), fromOp_(nullptr) {
		grad_.setAll(0);
	}

	GradWrapper& operator=(DataType&& other) {
		clean();

		data_ = std::move(other);
		grad_ = DataType(createSame(data_));
		fromOp_ = nullptr;

		grad_.setAll(0);

		return *this;
	}

	// # essential functions ===============================================================

	void backward() {
		TZ_CHECK(fromOp_, "backward() called on a leaf tensor");
		grad_.setAll(T(1));
		fromOp_->backwardFn(grad_);
	}

	// # internal use ======================================================================

	DataType val() const {
		return data_;
	}

	Tensor<T> tensor() const override {
		return data_.toTensor();
	}

	Tensor<T> grad() const override {
		return grad_.toTensor();
	}

	void zeroGrad() const override {
		grad_.setAll(0);
	}

	static Derived fromOp(DataType&& data, std::function<void(const DataType&)> backwardFn) {
		void* raw = mem::defaultAllocator(CPU).allocate(sizeof(GradOpNode<T, DataType>),
		                                                alignof(GradOpNode<T, DataType>));
		GradOpNode<T, DataType>* node = new (raw) GradOpNode<T, DataType>();
		node->backwardFn = std::move(backwardFn);
		node->inEdges = 0;

		Derived out(std::move(data));
		out.fromOp_ = node;
		return out;
	}

	void propagate(const DataType& upstream) const {
		grad_ += upstream;

		if (!fromOp_)
			return;

		fromOp_->inEdges--;
		if (fromOp_->inEdges == 0)
			fromOp_->backwardFn(grad_);
	}

	void addConsumer() const {
		if (fromOp_)
			fromOp_->inEdges++;
	}

  private:
	void clean() {
		if (fromOp_)
			mem::defaultAllocator(CPU).deallocate(fromOp_, sizeof(GradOpNode<T, DataType>));
	}
};

template <class Derived, class DataType, class T>
std::ostream& operator<<(std::ostream& os, const GradWrapper<Derived, DataType, T>& t) {
	os << "data: " << t.val();
	return os;
}

// # ================================================================================

template <class T>
class GradTensor : public GradWrapper<GradTensor<T>, Tensor<T>, T> {
	using Base = GradWrapper<GradTensor<T>, Tensor<T>, T>;

  public:
	using Base::Base;
};

template <class T>
class GradScalar : public GradWrapper<GradScalar<T>, Scalar<T>, T> {
	using Base = GradWrapper<GradScalar<T>, Scalar<T>, T>;

  public:
	using Base::Base;
};

template <class T>
class GradVector : public GradWrapper<GradVector<T>, Vector<T>, T> {
	using Base = GradWrapper<GradVector<T>, Vector<T>, T>;

  public:
	using Base::Base;
};

template <class T>
class GradMatrix : public GradWrapper<GradMatrix<T>, Matrix<T>, T> {
	using Base = GradWrapper<GradMatrix<T>, Matrix<T>, T>;

  public:
	using Base::Base;
};

// # ================================================================================

// sochastic gradient descent optimizer
template <class T>
class SGD {
	std::vector<GradBase<T>*> params_;
	double learnRate_;

  public:
	SGD(std::vector<GradBase<T>*> params, double lr) : params_(params), learnRate_(lr) {}

	void step() {
		for (GradBase<T>* p : params_) {
			p->tensor() -= p->grad() * learnRate_;
			p->zeroGrad();
		}
	}
};


// sochastic gradient descent optimizer with momentum
template <class T>
class SGDmomentum {
	std::vector<GradBase<T>*> params_;
	double learnRate_;
	double momentumConstant_;
	std::vector<Tensor<T>> momentum_;

  public:
	SGDmomentum(std::vector<GradBase<T>*> params, double lr, double momentumConstant)
	    : params_(params), learnRate_(lr), momentumConstant_(momentumConstant),
	      momentum_(params_.size()) {
		for (uint64_t i = 0; i < params_.size(); i++) {
			momentum_[i] = Tensor<T>(createSame(params_[i]->tensor()));
			momentum_[i].setAll(0);
		}
	}

	void step() {
		for (uint64_t i = 0; i < params_.size(); i++) {
			GradBase<T>* p = params_[i];
			momentum_[i] = momentum_[i] * momentumConstant_ + p->grad();
			p->tensor() -= momentum_[i] * learnRate_;
			p->zeroGrad();
		}
	}
};


// RMSProp optimizer
template <class T>
class RMSProp {
	constexpr static double EPSILON = 1e-8;

	std::vector<GradBase<T>*> params_;
	double learnRate_;
	double decayRate_;
	std::vector<Tensor<T>> movingAverage_;

  public:
	RMSProp(std::vector<GradBase<T>*> params, double lr, double decayRate)
	    : params_(params), learnRate_(lr), decayRate_(decayRate), movingAverage_(params_.size()) {
		for (uint64_t i = 0; i < params_.size(); i++) {
			movingAverage_[i] = Tensor<T>(createSame(params_[i]->tensor()));
			movingAverage_[i].setAll(0);
		}
	}

	void step() {
		for (uint64_t i = 0; i < params_.size(); i++) {
			GradBase<T>* p = params_[i];
			movingAverage_[i] =
			    movingAverage_[i] * decayRate_ + p->grad() * p->grad() * (1 - decayRate_);
			p->tensor() -= p->grad() / (movingAverage_[i].pow(T(0.5)) + EPSILON) * learnRate_;
			p->zeroGrad();
		}
	}
};


} // namespace grad
} // namespace tz
