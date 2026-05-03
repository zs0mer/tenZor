#pragma once

#include "utils.hpp"
#include <cassert>
#include <math.h>

namespace zi {

template <class T>
struct RrandomUniform {
	T min, max;

	TZ_HOST_DEVICE void operator()(T& x) const {
#ifdef __CUDA_ARCH__
		uint64_t h = reinterpret_cast<uint64_t>(&x);
		h ^= h >> 35;
		h *= 2512347u;
		h ^= h >> 27;
		h *= 43543563u;
		h ^= h >> 3;
		h ^= h << 7;
		x = (T)(min + (max - min) * (double)(h & 0xFFFFFF) / (double)0xFFFFFF);
#else
		x = (T)(min + (max - min) * ((double)rand() / RAND_MAX));
#endif
	}
};

template <class T>
struct Sigmoid {
	TZ_HOST_DEVICE void operator()(const T& a, T& b) const {
		// x = 1 / (1 + e^(-x))
		b = T(1.0) / (T(1.0f) + expf(-a));
	}
};

template <class T>
struct SigmoidDerivative {
	TZ_HOST_DEVICE void operator()(const T& a, T& b) const {
		// x = sigmoid(x) * (1 - sigmoid(x))
		T sigmoid;
		Sigmoid<T>()(a, sigmoid);
		b = sigmoid * (T(1.0) - sigmoid);
	}
};

template <class T>
struct SquaredError {
	TZ_HOST_DEVICE void operator()(const T& a, const T& b, T& c) const {
		// c = (a - b)^2
		T diff = a - b;
		c = (diff * diff) / 2;
	}
};

template <class T>
struct SquaredErrorDerivative {
	TZ_HOST_DEVICE void operator()(const T& a, const T& b, T& c) const {
		// c = (a - b)
		c = (a - b);
	}
};

#define TZ_EXTRA_GPU_FUNCTIONS_INSTANTIATE(T)                                                      \
	INSTANTIATE_UNARY_APPLY(T, zi::RrandomUniform);                                                \
	INSTANTIATE_BINARY_APPLY(T, zi::Sigmoid);                                                      \
	INSTANTIATE_BINARY_APPLY(T, zi::SigmoidDerivative);                                            \
	INSTANTIATE_TERNARY_APPLY(T, zi::SquaredError);                                                \
	INSTANTIATE_TERNARY_APPLY(T, zi::SquaredErrorDerivative);

} // namespace zi
