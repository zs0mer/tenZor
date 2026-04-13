#pragma once

#include <iostream>
#include <chrono>
#include <iomanip>
#include <cassert>

namespace TZ::internal {

// # -------------------------
#ifdef TZ_ERRORS
// nothing
#else
#define TZ_ERRORS 1
#endif
// # -------------------------

#if TZ_ERRORS
#define TZ_CHECK(expr, error) TZ::internal::check((expr), (error), __FILE__, __LINE__, __func__)
#define TZ_CHECK_(expr) TZ::internal::check((expr), "unexpected", __FILE__, __LINE__, __func__)
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

	if (expr) {
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

class Timer {
  private:
	using Clock = std::chrono::high_resolution_clock;
	Clock::time_point start;
	std::string label;

  public:
	explicit Timer(std::string s = "timer") : label(std::move(s)), start(Clock::now()) {}

	~Timer() {
		auto end = Clock::now();
		std::chrono::duration<double> duration = end - start;

		std::cout << "> " << label << ": " << duration.count() << " seconds" << std::endl;
	}
};

// # --------------------------------------------------

template <typename T>
struct Copy {
	TZ_HOST_DEVICE void operator()(const T& a, T& b) const {
		b = a;
	}
};

template <typename T>
struct Add {
	TZ_HOST_DEVICE void operator()(const T& a, const T& b, T& c) const {
		c = a + b;
	}
};

template <typename T>
struct Subtract {
	TZ_HOST_DEVICE void operator()(const T& a, const T& b, T& c) const {
		c = a - b;
	}
};

template <typename T>
struct Negate {
	TZ_HOST_DEVICE void operator()(T& a) const {
		a = -a;
	}
};

template <typename T>
struct AddScalar {
	T val;

	AddScalar(const T& s) : val(s) {}

	TZ_HOST_DEVICE void operator()(const T& a, T& b) const {
		b = a + val;
	}
};

template <typename T>
struct SubtractScalar {
	T val;

	SubtractScalar(const T& s) : val(s) {}

	TZ_HOST_DEVICE void operator()(const T& a, T& b) const {
		b = a - val;
	}
};

template <typename T>
struct MultiplyScalar {
	T val;

	MultiplyScalar(const T& s) : val(s) {}

	TZ_HOST_DEVICE void operator()(const T& a, T& b) const {
		b = a * val;
	}
};

template <typename T>
struct Set {
	T val;

	Set(const T& s) : val(s) {}

	TZ_HOST_DEVICE void operator()(T& a) const {
		a = val;
	}
};

template <typename T>
struct Sum {
	T& val;

	Sum(T& s) : val(s) {}

	TZ_HOST_DEVICE void operator()(T& a) const {
#ifdef __CUDA_ARCH__
		assert(false);
#endif
		val += a;
	}
};


} // namespace TZ::internal
