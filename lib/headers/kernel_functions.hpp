#pragma once

#include <cstdint>

namespace tz {
namespace impl {

// # -------------------------
#ifdef TZ_MAX_DIM
const constexpr std::uint8_t MAX_DIM = TZ_MAX_DIM;
#else
const constexpr std::uint8_t MAX_DIM = 4;
#endif
// # -------------------------

}; // namespace impl
namespace cuda {
constexpr uint32_t THREADS = 256;
constexpr uint32_t MAXBLOCKNUM = 64;

// # -------------------------

template <class T>
struct SimpleTensor {
	uint8_t dim;
	uint64_t shape[impl::MAX_DIM];
	uint64_t strides[impl::MAX_DIM];
	uint64_t offset;
	T* data;

	// cached
	uint64_t size;
	bool dense;
};

// # -------------------------

void* allocGPU(const uint64_t bytes, const uint8_t alignment = 64);

void freeGPU(void* ptr, const uint64_t bytes);

void copyToGPU(void* to, const void* from, const uint64_t bytes);

void copyToCPU(void* to, const void* from, const uint64_t bytes);

void memCopyOnGPU(void* to, const void* from, const uint64_t bytes);

// # -------------------------

template <class T, class Func>
void applyGPU(SimpleTensor<T> a, Func func);

template <class T, class Func>
void applyGPU(const SimpleTensor<T> a, SimpleTensor<T> b, Func func);

template <class T, class Func>
void applyGPU(const SimpleTensor<T> a, const SimpleTensor<T> b, SimpleTensor<T> c, Func func);

// # -------------------------

template <class T>
T dot(const SimpleTensor<T> a, const SimpleTensor<T> b);

template <class T>
void matmul(const SimpleTensor<T> a, const SimpleTensor<T> b, SimpleTensor<T> c);

}; // namespace cuda
}; // namespace tz
