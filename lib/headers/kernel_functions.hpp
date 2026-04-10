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

constexpr uint32_t BLOCKSIZE = 256;
constexpr uint32_t MAXGRIDSIZE = 64;


template <class T>
struct SimpleTensor {
	const uint8_t dim;
	const uint64_t* shape;
	const uint64_t* strides;
	const uint64_t offset;
	T* data;

	// chashed
	const uint64_t size;
	const bool dense;
};

// # ----------------

void* allocGPU(const uint64_t bytes, const uint8_t alignment);

void freeGPU(void* ptr, const uint64_t bytes);

void copyToGPU(void* to, void* from, const uint64_t bytes);

void copyToCPU(void* to, void* from, const uint64_t bytes);

void memCopyGPU(void* to, void* from, const uint64_t bytes);

// # ----------------

template <class Func, class T>
void applyGPU(SimpleTensor<T> a, Func func);

template <class Func, class T>
void applyGPU(const SimpleTensor<T> a, SimpleTensor<T> b, Func func);

template <class Func, class T>
void applyGPU(const SimpleTensor<T> a, const SimpleTensor<T> b, SimpleTensor<T> c, Func func);

}; // namespace TZ::cuda
