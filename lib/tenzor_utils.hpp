#pragma once

namespace TZ {

#if ERRORS
#define _CHECK(expresson, error) _check((expresson), (error), __FILE__, __LINE__, __func__)
#define _CHECK_(expresson) _check((expresson), "unexpected", __FILE__, __LINE__, __func__)

// a function to make error handleing easier
inline void _check(const bool expresson, const char* error, const char* file, int line,
                   const char* func) {

	if (expresson) {
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
#define _CHECK(expresson, error)
#define _CHECK_(expresson)
#endif

} // namespace TZ