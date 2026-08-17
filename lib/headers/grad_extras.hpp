#pragma once

#include "grad_classes.hpp"
#include "grad_functions.hpp"
#include "math_classes.hpp"
#include "tensor.hpp"
#include "utils.hpp"
#include <cstdint>

namespace tz::grad {

// sochastic gradient descent optimizer
template <class T>
class SGD {
	std::vector<GradBase<T>*> params_;
	T learnRate_;

  public:
	SGD() = default;

	SGD(std::vector<GradBase<T>*> params, T lr) : params_(params), learnRate_(lr) {}

	void add(GradBase<T>& param) {
		params_.push_back(&param);
	}

	void add(const std::vector<GradBase<T>*>& params) {
		for (GradBase<T>* p : params)
			params_.push_back(p);
	}

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
	T learnRate_;
	T momentumConstant_;
	std::vector<Tensor<T>> momentum_;

  public:
	SGDmomentum() = default;

	SGDmomentum(std::vector<GradBase<T>*> params, T lr, T momentumConstant = 0.9)
	    : params_(params), learnRate_(lr), momentumConstant_(momentumConstant),
	      momentum_(params_.size()) {
		for (uint64_t i = 0; i < params_.size(); i++) {
			momentum_[i] = createSame(params_[i]->tensor());
			momentum_[i].setAll(0);
		}
	}

	void add(GradBase<T>& param) {
		params_.push_back(&param);

		momentum_.push_back(createSame(param.tensor()));
		momentum_.back().setAll(0);
	}

	void add(const std::vector<GradBase<T>*>& params) {
		for (GradBase<T>* p : params) {
			params_.push_back(p);

			momentum_.push_back(createSame(p->tensor()));
			momentum_.back().setAll(0);
		}
	}

	void step(uint64_t batchSize = 1) {
		for (uint64_t i = 0; i < params_.size(); i++) {
			GradBase<T>* p = params_[i];
			if (batchSize != 1)
				p->grad() /= batchSize;

			momentum_[i] = momentum_[i] * momentumConstant_ + p->grad();
			p->tensor() -= momentum_[i] * learnRate_;
			p->zeroGrad();
		}
	}
};

// RMSProp optimizer
template <class T>
class RMSProp {
	constexpr static T EPSILON = T(1e-8);

	std::vector<GradBase<T>*> params_;
	T learnRate_;
	T decayRate_;
	std::vector<Tensor<T>> movingAverage_;

  public:
	RMSProp() = default;

	RMSProp(std::vector<GradBase<T>*> params, T lr, T decayRate = 0.9)
	    : params_(params), learnRate_(lr), decayRate_(decayRate), movingAverage_(params_.size()) {
		for (uint64_t i = 0; i < params_.size(); i++) {
			movingAverage_[i] = createSame(params_[i]->tensor());
			movingAverage_[i].setAll(0);
		}
	}

	void add(GradBase<T>& param) {
		params_.push_back(&param);

		movingAverage_.push_back(createSame(param.tensor()));
		movingAverage_.back().setAll(0);
	}

	void add(const std::vector<GradBase<T>*>& params) {
		for (GradBase<T>* p : params) {
			params_.push_back(p);

			movingAverage_.push_back(createSame(p->tensor()));
			movingAverage_.back().setAll(0);
		}
	}

	void step(uint64_t batchSize = 1) {
		for (uint64_t i = 0; i < params_.size(); i++) {
			GradBase<T>* p = params_[i];
			if (batchSize != 1)
				p->grad() /= batchSize;

			movingAverage_[i] =
			    movingAverage_[i] * decayRate_ + p->grad() * p->grad() * (1 - decayRate_);
			p->tensor() -= p->grad() / (movingAverage_[i].pow(T(0.5)) + EPSILON) * learnRate_;
			p->zeroGrad();
		}
	}
};

// Adam optimizer
template <class T>
class Adam {
	constexpr static T EPSILON = T(1e-8);

	std::vector<GradBase<T>*> params_;
	T learnRate_;
	T decayRate_;
	T momentumConstant_;
	std::vector<Tensor<T>> movingAverage_;
	std::vector<Tensor<T>> momentum_;
	uint64_t t_ = 0;

  public:
	Adam() = default;

	Adam(std::vector<GradBase<T>*> params, T lr, T momentumConstant = 0.9, T decayRate = 0.999)
	    : params_(params), learnRate_(lr), momentumConstant_(momentumConstant),
	      decayRate_(decayRate), movingAverage_(params_.size()), momentum_(params_.size()) {
		for (uint64_t i = 0; i < params_.size(); i++) {
			movingAverage_[i] = createSame(params_[i]->tensor());
			movingAverage_[i].setAll(0);
			momentum_[i] = createSame(params_[i]->tensor());
			momentum_[i].setAll(0);
		}
	}

	void add(GradBase<T>& param) {
		params_.push_back(&param);

		momentum_.push_back(createSame(param.tensor()));
		momentum_.back().setAll(0);

		movingAverage_.push_back(createSame(param.tensor()));
		movingAverage_.back().setAll(0);
	}

	void add(const std::vector<GradBase<T>*>& params) {
		for (GradBase<T>* p : params) {
			params_.push_back(p);

			movingAverage_.push_back(createSame(p->tensor()));
			movingAverage_.back().setAll(0);

			momentum_.push_back(createSame(p->tensor()));
			momentum_.back().setAll(0);
		}
	}

	void step(uint64_t batchSize = 1) {
		t_++;
		for (uint64_t i = 0; i < params_.size(); i++) {
			GradBase<T>* p = params_[i];
			if (batchSize != 1)
				p->grad() /= batchSize;

			momentum_[i] = momentum_[i] * momentumConstant_ + p->grad() * (1 - momentumConstant_);
			movingAverage_[i] =
			    movingAverage_[i] * decayRate_ + p->grad() * p->grad() * (1 - decayRate_);

			Tensor<T> mHat = momentum_[i] / (1 - std::pow(momentumConstant_, t_));
			Tensor<T> vHat = movingAverage_[i] / (1 - std::pow(decayRate_, t_));

			p->tensor() -= mHat / (vHat.pow(T(0.5)) + EPSILON) * learnRate_;
			p->zeroGrad();
		}
	}
};

// # ================================================================================

template <class T>
class Linear {
  private:
	GradMatrix<T> weights_;
	GradVector<T> bias_;
	GradVector<T> tmp_;
	std::vector<GradBase<T>*> params_;

  public:
	Linear(uint64_t in, uint64_t out, Device device = CPU)
	    : weights_(Matrix<T>(out, in, device)), bias_(Vector<T>(out, device)),
	      params_({&weights_, &bias_}) {
		// He weight initialization (uniformly)
		T scale = std::sqrt(T(6.0) / in);

		bias_.val().setAll(T(0));
		weights_.val().apply(impl::RrandomUniform<T>(-scale, scale), impl::AnyDevice{});
	}

	const std::vector<GradBase<T>*>& parameters() {
		return params_;
	}

	GradVector<T> operator()(const GradVector<T>& input) {
		tmp_ = fn::matmul<T>(weights_, input);
		return fn::add<T>(tmp_, bias_);
	}
};

} // namespace tz::grad
