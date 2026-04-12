#pragma once

#include <cstdint>

namespace TZ::cuda {

// # -------------------------
#ifdef SYNCGPU
// nothing
#else
#define SYNCGPU 1
#endif
// # -------------------------

constexpr uint32_t THREADS = 256;
constexpr uint32_t MAXBLOCKNUM = 64;

// # -------------------------

void* allocGPU(const uint64_t bytes, const uint8_t alignment = 64);

void freeGPU(void* ptr, const uint64_t bytes);

void copyToGPU(void* to, const void* from, const uint64_t bytes);

void copyToCPU(void* to, const void* from, const uint64_t bytes);

void memCopyGPU(void* to, const void* from, const uint64_t bytes);

// # -------------------------

template <class T>
struct SimpleTensor {
	uint8_t dim;
	uint64_t* shape;
	uint64_t* strides;
	uint64_t offset;
	T* data;

	// chashed
	uint64_t size;
	bool dense;
};

// # -------------------------

template <class T, class Func>
void applyGPU(SimpleTensor<T> a, Func func);

template <class T, class Func>
void applyGPU(const SimpleTensor<T> a, SimpleTensor<T> b, Func func);

template <class T, class Func>
void applyGPU(const SimpleTensor<T> a, const SimpleTensor<T> b, SimpleTensor<T> c, Func func);

}; // namespace TZ::cuda
