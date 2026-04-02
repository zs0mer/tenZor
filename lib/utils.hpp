#pragma once

#include <iostream>
#include <chrono>
#include <iomanip>

namespace TZ::internal {

#define TZ_ERRORS 1

#if TZ_ERRORS
#define TZ_CHECK(expr, error) TZ::internal::check((expr), (error), __FILE__, __LINE__, __func__)
#define TZ_CHECK_(expr) TZ::internal::check((expr), "unexpected", __FILE__, __LINE__, __func__)

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
#else
#define TZ_CHECK(expr, error) ((void)0)
#define TZ_CHECK_(expr) ((void)0)
#endif

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

} // namespace TZ::internal
