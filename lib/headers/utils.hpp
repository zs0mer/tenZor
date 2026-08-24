#pragma once

#include <cmath>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cassert>
#include <sstream>

namespace tz::impl {

// # -------------------------
#ifdef TZ_ERRORS
// nothing
#else
#define TZ_ERRORS 1
#endif
// # -------------------------

#if TZ_ERRORS
#define TZ_CHECK(expr, error) tz::impl::check((expr), (error), __FILE__, __LINE__, __func__)
#define TZ_CHECK_(expr) tz::impl::check((expr), "unexpected", __FILE__, __LINE__, __func__)
#else
#define TZ_CHECK(expr, error) ((void)0)
#define TZ_CHECK_(expr) ((void)0)
#endif

#ifdef __CUDACC__
#define TZ_HOST_DEVICE __host__ __device__
#else
#define TZ_HOST_DEVICE
#endif

// a function to make error handleing easier
inline void check(const bool expr, const char* error, const char* file, int line,
                  const char* func) {

	if (!expr) {
		auto now = std::chrono::system_clock::now();
		std::time_t t_c = std::chrono::system_clock::to_time_t(now);

		std::ostringstream oss;
		oss << "Error: " << error << "\n"
		    << "File: " << file << "\n"
		    << "Line: " << line << "\n"
		    << "Function: " << func << "\n"
		    << "Time: " << std::put_time(std::localtime(&t_c), "%F %T");

		throw std::runtime_error(oss.str());
	}
}

// # --------------------------------------------------
// CUDA ONLY

#ifdef __CUDACC__

// # -------------------------
#ifdef SYNCGPU
// nothing
#else
#define SYNCGPU 1
#endif
// # -------------------------

void check_cuda(const char* file, int line, const char* func);

#if TZ_ERRORS
#define CHECK_CUDA check_cuda(__FILE__, __LINE__, __func__)
#else
#define CHECK_CUDA ((void)0)
#endif

template <class T>
__device__ inline void gpuAtomicAdd(T* address, T val) {
	atomicAdd(address, val);
}

template <>
__device__ inline void gpuAtomicAdd<uint64_t>(uint64_t* address, uint64_t val) {
	atomicAdd(reinterpret_cast<unsigned long long int*>(address),
	          static_cast<unsigned long long int>(val));
}

template <>
__device__ inline void gpuAtomicAdd<int64_t>(int64_t* address, int64_t val) {
	unsigned long long int* addressUll = (unsigned long long int*)address;
	unsigned long long int old = *addressUll;
	unsigned long long int assumed;

	// loop neded for to remove race conditions
	do {
		assumed = old;
		old = atomicCAS(addressUll, assumed, (unsigned long long int)((int64_t)assumed + val));
	} while (assumed != old);
}

#endif

// # --------------------------------------------------

template <class T>
struct Copy {
	TZ_HOST_DEVICE void operator()(const T& a, T& b) const {
		b = a;
	}
};

template <class T>
struct Add {
	TZ_HOST_DEVICE void operator()(const T& a, const T& b, T& c) const {
		c = a + b;
	}
};

template <class T>
struct Subtract {
	TZ_HOST_DEVICE void operator()(const T& a, const T& b, T& c) const {
		c = a - b;
	}
};

template <class T>
struct Multiply {
	TZ_HOST_DEVICE void operator()(const T& a, const T& b, T& c) const {
		c = a * b;
	}
};

template <class T>
struct Divide {
	TZ_HOST_DEVICE void operator()(const T& a, const T& b, T& c) const {
		c = a / b;
	}
};

template <class T>
struct Negate {
	TZ_HOST_DEVICE void operator()(const T& a, T& b) const {
		b = -a;
	}
};

template <class T>
struct Exp {
	TZ_HOST_DEVICE void operator()(const T& a, T& b) const {
		b = std::exp(double(a));
	}
};

template <class T>
struct Log {
	TZ_HOST_DEVICE void operator()(const T& a, T& b) const {
		b = std::log(double(a));
	}
};

template <class T>
struct Pow {
	T val;

	Pow(const T& s) : val(s) {}

	TZ_HOST_DEVICE void operator()(const T& a, T& b) const {
		b = std::pow(a, val);
	}
};

template <class T>
struct AddScalar {
	T val;

	AddScalar(const T& s) : val(s) {}

	TZ_HOST_DEVICE void operator()(const T& a, T& b) const {
		b = a + val;
	}
};

template <class T>
struct SubtractScalar {
	T val;

	SubtractScalar(const T& s) : val(s) {}

	TZ_HOST_DEVICE void operator()(const T& a, T& b) const {
		b = a - val;
	}
};

template <class T>
struct MultiplyScalar {
	T val;

	MultiplyScalar(const T& s) : val(s) {}

	TZ_HOST_DEVICE void operator()(const T& a, T& b) const {
		b = a * val;
	}
};

template <class T>
struct DivideScalar {
	T val;

	DivideScalar(const T& s) : val(s) {}

	TZ_HOST_DEVICE void operator()(const T& a, T& b) const {
		b = a / val;
	}
};

template <class T>
struct Set {
	T val;

	Set(const T& s) : val(s) {}

	TZ_HOST_DEVICE void operator()(T& a) const {
		a = val;
	}
};

template <class T>
struct Sum {
	T* val;

	Sum(T* s) : val(s) {}

	TZ_HOST_DEVICE void operator()(const T& a) {
#ifdef __CUDA_ARCH__
		gpuAtomicAdd(val, a);
#else
#pragma omp atomic
		*val += a;
#endif
	}
};

// # other functions --------------------------------------------------

template <class T>
struct Sigmoid {
	TZ_HOST_DEVICE void operator()(const T& a, T& b) const {
		b = T(1) / (T(1) + ::exp(-double(a)));
	}
};

template <class T>
struct SigmoidGrad {
	TZ_HOST_DEVICE void operator()(const T& upstream, const T& s, T& out) const {
		out = s * (T(1) - s);
		out *= upstream;
	}
};

template <class T>
struct Relu {
	TZ_HOST_DEVICE void operator()(const T& a, T& b) const {
		b = a > 0 ? a : 0;
	}
};

template <class T>
struct ReluGrad {
	TZ_HOST_DEVICE void operator()(const T& upstream, const T& a, T& out) const {
		out = a > T(0) ? upstream : T(0);
	}
};

template <class T>
struct Tanh {
	TZ_HOST_DEVICE void operator()(const T& a, T& b) const {
		b = ::tanh(double(a));
	}
};

template <class T>
struct TanhGrad {
	TZ_HOST_DEVICE void operator()(const T& upstream, const T& s, T& out) const {
		out = 1 - s * s;
		out *= upstream;
	}
};

template <class T>
struct RrandomUniform {
	T min_, max_;

	RrandomUniform(const T& mn, const T& mx) : min_(mn), max_(mx) {}

	TZ_HOST_DEVICE void operator()(T& x) const {
#ifdef __CUDA_ARCH__
		uint64_t h = reinterpret_cast<uint64_t>(&x);
		h ^= h >> 35;
		h *= 2512347u;
		h ^= h >> 27;
		h *= 43543563u;
		h ^= h >> 3;
		h ^= h << 7;
		x = (T)(min_ + (max_ - min_) * (double)(h & 0xFFFFFF) / (double)0xFFFFFF);
#else
		x = (T)(min_ + (max_ - min_) * ((double)rand() / RAND_MAX));
#endif
	}
};

} // namespace tz::impl
