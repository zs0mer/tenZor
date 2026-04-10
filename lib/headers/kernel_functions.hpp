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

constexpr uint32_t GRIDKSIZE = 256;
constexpr uint32_t MAXBLOCKSIZE = 64;

// # -------------------------

void* allocGPU(const uint64_t bytes, const uint8_t alignment = 64);

void freeGPU(void* ptr, const uint64_t bytes);

void copyToGPU(void* to, void* from, const uint64_t bytes);

void copyToCPU(void* to, void* from, const uint64_t bytes);

void memCopyGPU(void* to, void* from, const uint64_t bytes);

// # -------------------------

template <class T>
struct SimpleTensor {
	uint8_t dim_;
	uint64_t* shape_;
	uint64_t* strides_;
	uint64_t offset_;
	T* data_;

	// chashed
	uint64_t size_;
	bool dense_;
};

// # -------------------------

template <class Func, class T>
void applyGPU(SimpleTensor<T> a, Func func);

template <class Func, class T>
void applyGPU(const SimpleTensor<T> a, SimpleTensor<T> b, Func func);

template <class Func, class T>
void applyGPU(const SimpleTensor<T> a, const SimpleTensor<T> b, SimpleTensor<T> c, Func func);

}; // namespace TZ::cuda
