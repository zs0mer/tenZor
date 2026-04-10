#include <stdio.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cstdint>

#include "kernel_functions.hpp"
#include "utils.hpp"

namespace TZ::cuda {

// # ---------------------------------------------

#if TZ_ERRORS
#define CHECK_CUDA check_cuda(__FILE__, __LINE__, __func__)
#else
#define CHECK_CUDA ((void)0)
#endif

inline void check_cuda(const char* file, int line, const char* func) {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		auto now = std::chrono::system_clock::now();
		std::time_t t_c = std::chrono::system_clock::to_time_t(now);

		std::ostringstream oss;
		oss << "Error: " << cudaGetErrorString(err) << "\n"
		    << "File: " << file << "\n"
		    << "Line: " << line << "\n"
		    << "Function: " << func << "\n"
		    << "Time: " << std::put_time(std::localtime(&t_c), "%F %T");

		throw std::runtime_error(oss.str());
	}
}

void sync() {
#if SYNCGPU
	cudaDeviceSynchronize();
#endif
}


// # ---------------------------------------------

void* allocGPU(const uint64_t bytes, const uint8_t alignment) {
	// * cudaMalloc() already alings memory well (at least 64 byte)
	void* ptr = nullptr;
	cudaMalloc(&ptr, bytes);
	CHECK_CUDA;
	return ptr;
}

void freeGPU(void* ptr, const uint64_t bytes) {
	cudaFree(ptr);
	CHECK_CUDA;
}

void copyToGPU(void* to, void* from, const uint64_t bytes) {
	cudaMemcpy(to, from, bytes, cudaMemcpyHostToDevice);
}

void copyToCPU(void* to, void* from, const uint64_t bytes) {
	cudaMemcpy(to, from, bytes, cudaMemcpyDeviceToHost);
}

void memCopyGPU(void* to, void* from, const uint64_t bytes) {
	cudaMemcpy(to, from, bytes, cudaMemcpyDeviceToDevice);
}

// # ---------------------------------------------

template <typename T>
__device__ uint64_t computeLinearIdx(uint64_t flatIdx, const SimpleTensor<T>& a) {
	uint64_t idx = a.offset;

	for (uint8_t d = a.dim; d-- > 0;) {
		uint64_t coord = flatIdx % shape[d];
		flatIdx /= shape[d];
		idx += coord * strides[d];
	}

	return idx;
}

template <typename T, typename Func>
__global__ void applyKernel(SimpleTensor<T>& a, Func func) {
	const uint64_t stride = blockDim.x * gridDim.x;
	uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
	T* base = static_cast<T*>(a.data);

	while (i < a.size) {

		uint64_t idx = a.dense ? i + a.offset : computeLinearIdx(i, a);

		func(base[idx]);

		i += stride;
	}
}

template <typename T, typename Func>
__global__ void applyKernel(const SimpleTensor<T>& a, SimpleTensor<T>& b, Func func) {
	const uint64_t stride = blockDim.x * gridDim.x;
	uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
	const T* baseA = a.data;
	T* baseB = b.data;

	if (a.dense && b.dense) {
		while (i < a.size) {
			func(baseA[i + a.offset], baseB[i + b.offset]);

			i += stride;
		}
		return;
	}

	while (i < a.size) {

		uint64_t idxA = a.dense ? i + a.offset : computeLinearIdx(i, a);
		uint64_t idxB = b.dense ? i + b.offset : computeLinearIdx(i, b);

		func(baseA[idxA], baseB[idxB]);

		i += stride;
	}
}

template <typename T, typename Func>
__global__ void applyKernel(const SimpleTensor<T>& a, const SimpleTensor<T>& b, SimpleTensor<T>& c,
                            Func func) {
	const uint64_t stride = blockDim.x * gridDim.x;
	uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
	const T* baseA = a.data;
	const T* baseB = b.data;
	T* baseC = c.data;

	if (a.dense && b.dense && c.dense) {
		while (i < a.size) {
			func(baseA[i + a.offset], baseB[i + b.offset], baseC[i + c.offset]);

			i += stride;
		}
		return;
	}

	while (i < a.size) {

		uint64_t idxA = a.dense ? i + a.offset : computeLinearIdx(i, a);
		uint64_t idxB = b.dense ? i + b.offset : computeLinearIdx(i, b);
		uint64_t idxC = c.dense ? i + c.offset : computeLinearIdx(i, c);

		func(baseA[idxA], baseB[idxB], baseC[idxC]);

		i += stride;
	}
}

template <class Func, class T>
void applyGPU(SimpleTensor<T> a, Func func) {
	int gridSize = min((a.size + BLOCKSIZE - 1) / BLOCKSIZE, MAXGRIDSIZE);
	applyKernel<<<BLOCKSIZE, gridSize>>>(a, func);
	sync();
	CHECK_CUDA;
}

template <class Func, class T>
void applyGPU(const SimpleTensor<T> a, SimpleTensor<T> b, Func func) {
	int gridSize = min((a.size + BLOCKSIZE - 1) / BLOCKSIZE, MAXGRIDSIZE);
	applyKernel<<<BLOCKSIZE, gridSize>>>(a, func);
	sync();
	CHECK_CUDA;
}

template <class Func, class T>
void applyGPU(const SimpleTensor<T> a, const SimpleTensor<T> b, SimpleTensor<T> c, Func func) {
	int gridSize = min((a.size + BLOCKSIZE - 1) / BLOCKSIZE, MAXGRIDSIZE);
	applyKernel<<<BLOCKSIZE, gridSize>>>(a, func);
	sync();
	CHECK_CUDA;
}

} // namespace TZ::cuda
