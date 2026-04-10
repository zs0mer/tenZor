#pragma once

#include <iostream>
#include <chrono>
#include <iomanip>

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
	std::chrono::_V2::system_clock::time_point start;
	std::string s;

  public:
	Timer(std::string s_ = "timer") : s(s_) {
		start = std::chrono::high_resolution_clock::now();
	}
	~Timer() {
		std::chrono::_V2::system_clock::time_point end = std::chrono::high_resolution_clock::now();

		std::chrono::duration duration = std::chrono::duration<double>(end - start);
		std::cout << "> " << s << ": " << duration.count() << " seconds" << std::endl;
	}
};

// # --------------------------------------------------

template <typename T>
struct Copy {
#ifdef CUDACC
	__host__ __device__
#endif
	    void operator()(const T& a, T& b) const {
		b = a;
	}
};

template <typename T>
struct Add {
#ifdef CUDACC
	__host__ __device__
#endif
	    void operator()(const T& a, const T& b, T& c) const {
		c = a + b;
	}
};

template <typename T>
struct Subtract {
#ifdef CUDACC
	__host__ __device__
#endif
	    void operator()(const T& a, const T& b, T& c) const {
		c = a - b;
	}
};

template <typename T>
struct Negate {
#ifdef CUDACC
	__host__ __device__
#endif
	    void operator()(T& a) const {
		a = -a;
	}
};

template <typename T>
struct AddScalar {
	T val;

	AddScalar(const T& s) : val(s) {}

#ifdef CUDACC
	__host__ __device__
#endif
	    void operator()(const T& a, T& b) const {
		b = a + val;
	}
};

template <typename T>
struct SubtractScalar {
	T val;

	SubtractScalar(const T& s) : val(s) {}

#ifdef CUDACC
	__host__ __device__
#endif
	    void operator()(const T& a, T& b) const {
		b = a - val;
	}
};

template <typename T>
struct MultiplyScalar {
	T val;

	MultiplyScalar(const T& s) : val(s) {}

#ifdef CUDACC
	__host__ __device__
#endif
	    void operator()(const T& a, T& b) const {
		b = a * val;
	}
};

template <typename T>
struct Set {
	T val;

	Set(const T& s) : val(s) {}

#ifdef CUDACC
	__host__ __device__
#endif
	    void operator()(T& a) const {
		a = val;
	}
};

template <typename T>
struct Sum {
	T& val;

	Sum(T& s) : val(s) {}

#ifdef CUDACC
	__host__ __device__
#endif
	    void operator()(T& a) const {
		val += a;
	}
};


} // namespace TZ::internal
