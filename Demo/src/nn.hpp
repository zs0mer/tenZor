#pragma once
#include <cstdint>
#include <vector>

#define TZ_APPLY_ERROR_IF_SHAPE_NOT_SAME 0

#include "TenZor.hpp"
#include "ExtraFunctions.hpp"

namespace zi {

template <class T>
class NeuralNet {

  private:
	tz::Vector<uint64_t> layerSize_;
	std::vector<tz::Matrix<T>> weights_;
	std::vector<tz::Vector<T>> biases_;

	std::vector<tz::Vector<T>> activation_;
	std::vector<tz::Vector<T>> preActivation_;

	// a : activation_
	// z : preActivation_
	std::vector<tz::Vector<T>> dC_da_;
	std::vector<tz::Vector<T>> dC_dz_;

	std::vector<tz::Matrix<T>> weightGrads_;
	std::vector<tz::Vector<T>> biasGrads_;

  public:
	template <class RandomFunc = RrandomUniform<T>>
	NeuralNet(std::initializer_list<uint64_t> layerSize, tz::Device d = tz::CPU,
	          RandomFunc random = {.min = -1, .max = 1})
	    : layerSize_(layerSize), weights_(layerSize.size()), biases_(layerSize.size()),
	      dC_da_(layerSize.size()), dC_dz_(layerSize.size()), activation_(layerSize.size()),
	      preActivation_(layerSize.size()), weightGrads_(layerSize.size()),
	      biasGrads_(layerSize.size()) {
		uint64_t L = layerSize_.size() - 1;

		for (uint64_t i = 0; i < L; i++) {
			weights_[i] = tz::Matrix<T>(layerSize_[i + 1], layerSize_[i], d);
			biases_[i] = tz::Vector<T>(layerSize_[i + 1], d);

			tz::impl::TensorIMPL<T>::apply(weights_[i].tensor_(), random, tz::impl::AnyDevice{});
			tz::impl::TensorIMPL<T>::apply(biases_[i].tensor_(), random, tz::impl::AnyDevice{});


			weightGrads_[i] = tz::Matrix<T>(layerSize_[i + 1], layerSize_[i], d);
			biasGrads_[i] = tz::Vector<T>(layerSize_[i + 1], d);

			weightGrads_[i].setAll(T(0.0));
			biasGrads_[i].setAll(T(0.0));
		}
	}


	template <class ActivationFunction = Sigmoid<T>>
	tz::Vector<T> forwardPass(const tz::Vector<T>& input, ActivationFunction activation = {}) {
		TZ_CHECK(input.size() == layerSize_[0],
		         "input size must be equal to the size of the first layer");

		activation_[0] = input.clone();
		uint64_t L = layerSize_.size() - 1;

		// calculating the next layer (i+1)
		for (uint64_t i = 0; i < L; i++) {
			preActivation_[i + 1] =
			    tz::Vector<T>(matmul(weights_[i], tz::Matrix<T>(activation_[i]))) + biases_[i];
			tz::impl::TensorIMPL<T>::apply(preActivation_[i + 1].tensor_(),
			                               activation_[i + 1].tensor_(), activation,
			                               tz::impl::AnyDevice{});
		}
		return activation_[layerSize_.size() - 1];
	}


	template <class ActivationFunctionDerivative = SigmoidDerivative<T>,
	          class CostFunctionDerivative = SquaredErrorDerivative<T>>
	void backwardPass(const tz::Vector<T>& input, const tz::Vector<T>& target,
	                  ActivationFunctionDerivative activationFunctionDerivative = {},
	                  CostFunctionDerivative costDerivative = {}) {
		TZ_CHECK(input.size() == layerSize_[0],
		         "input size must be equal to the size of the first layer");
		TZ_CHECK(target.size() == layerSize_[layerSize_.size() - 1],
		         "target size must be equal to the size of the last layer");

		uint64_t L = layerSize_.size() - 1;

		// dC/da = C'(a, y)
		tz::impl::TensorIMPL<T>::apply(activation_[layerSize_.size() - 1].tensor_(),
		                               target.tensor_(), dC_da_[layerSize_.size() - 1].tensor_(),
		                               costDerivative, tz::impl::AnyDevice{});

		// propagating back
		for (uint64_t i = L; i-- > 0;) {
			// dz/da = activationFunctionDerivative(z) -> dC/dz = dC/da * da/dz
			tz::impl::TensorIMPL<T>::apply(preActivation_[i + 1].tensor_(), dC_dz_[i + 1].tensor_(),
			                               activationFunctionDerivative, tz::impl::AnyDevice{});
			dC_dz_[i + 1] *= dC_da_[i + 1];

			// dz/db = 1 -> dC/db = dC/dz
			biasGrads_[i] += dC_dz_[i + 1];

			// dz/dw = a -> dC/dw = a * dC/dz
			weightGrads_[i] += matmul(tz::Matrix<T>(dC_dz_[i + 1]), activation_[i].transpose());

			// dz/da = w -> dC/da = w^T * dC/da
			dC_da_[i] = matmul(weights_[i].transpose(), tz::Matrix<T>(dC_dz_[i + 1]));
		}
	}


	void step(T learnRate, uint64_t batchSize) {
		uint64_t L = layerSize_.size() - 1;
		for (uint64_t i = 0; i < L; i++) {
			weightGrads_[i] /= batchSize;
			biasGrads_[i] /= batchSize;

			weights_[i] -= weightGrads_[i] * learnRate;
			biases_[i] -= biasGrads_[i] * learnRate;

			weightGrads_[i].setAll(T(0.0));
			biasGrads_[i].setAll(T(0.0));
		}
	}
};

} // namespace zi
