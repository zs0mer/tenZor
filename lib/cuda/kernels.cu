#include <stdio.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <algorithm>
#include <cublas_v2.h>

#include "utils.hpp"
#include "kernel_functions.hpp"

namespace TZ::cuda {

void check_cuda(const char* file, int line, const char* func) {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		auto now = std::chrono::system_clock::now();
		std::time_t t_c = std::chrono::system_clock::to_time_t(now);

		std::ostringstream oss;
		oss << "CUDA Error: " << cudaGetErrorString(err) << "\n"
		    << "File: " << file << "\n"
		    << "Line: " << line << "\n"
		    << "Function: " << func << "\n"
		    << "Time: " << std::put_time(std::localtime(&t_c), "%F %T");

		throw std::runtime_error(oss.str());
	}
}

inline void sync() {
#if SYNCGPU
	cudaDeviceSynchronize();
#endif
}

cublasHandle_t getCublasHandle() {
	thread_local cublasHandle_t handle = nullptr;
	if (!handle) {
		cublasCreate(&handle);
	}
	return handle;
}

// # ---------------------------------------------

void* allocGPU(const uint64_t bytes, const uint8_t alignment) {
	// * cudaMalloc() already aligns memory well (at least 64 byte)
	void* ptr = nullptr;
	cudaMalloc(&ptr, bytes);
	CHECK_CUDA;
	return ptr;
}

void freeGPU(void* ptr, const uint64_t bytes) {
	cudaFree(ptr);
	CHECK_CUDA;
}

void copyToGPU(void* to, const void* from, const uint64_t bytes) {
	cudaMemcpy(to, from, bytes, cudaMemcpyHostToDevice);
	CHECK_CUDA;
}

void copyToCPU(void* to, const void* from, const uint64_t bytes) {
	cudaMemcpy(to, from, bytes, cudaMemcpyDeviceToHost);
	CHECK_CUDA;
}

void memCopyOnGPU(void* to, const void* from, const uint64_t bytes) {
	cudaMemcpy(to, from, bytes, cudaMemcpyDeviceToDevice);
	CHECK_CUDA;
}

// # ---------------------------------------------

template <class T>
__device__ uint64_t computeLinearIdx(uint64_t flatIdx, const SimpleTensor<T> a) {
	uint64_t idx = a.offset;

	for (uint8_t d = a.dim; d-- > 0;) {
		uint64_t coord = flatIdx % a.shape[d];
		flatIdx /= a.shape[d];
		idx += coord * a.strides[d];
	}

	return idx;
}

template <class T, class Func>
__global__ void applyKernel(SimpleTensor<T> a, Func func) {
	const uint64_t stride = blockDim.x * gridDim.x;
	uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
	T* base = static_cast<T*>(a.data);

	while (i < a.size) {

		uint64_t idx = a.dense ? i + a.offset : computeLinearIdx(i, a);

		func(base[idx]);

		i += stride;
	}
}

template <class T, class Func>
__global__ void applyKernel(const SimpleTensor<T> a, SimpleTensor<T> b, Func func) {
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

template <class T, class Func>
__global__ void applyKernel(const SimpleTensor<T> a, const SimpleTensor<T> b, SimpleTensor<T> c,
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

template <class T, class Func>
void applyGPU(SimpleTensor<T> a, Func func) {
	uint64_t blocks = (a.size + THREADS - 1) / THREADS;
	blocks = (blocks < MAXBLOCKNUM) ? blocks : MAXBLOCKNUM;
	applyKernel<<<blocks, THREADS>>>(a, func);
	sync();
	CHECK_CUDA;
}

template <class T, class Func>
void applyGPU(const SimpleTensor<T> a, SimpleTensor<T> b, Func func) {
	uint64_t blocks = (a.size + THREADS - 1) / THREADS;
	blocks = (blocks < MAXBLOCKNUM) ? blocks : MAXBLOCKNUM;
	applyKernel<<<blocks, THREADS>>>(a, b, func);
	sync();
	CHECK_CUDA;
}

template <class T, class Func>
void applyGPU(const SimpleTensor<T> a, const SimpleTensor<T> b, SimpleTensor<T> c, Func func) {
	uint64_t blocks = (a.size + THREADS - 1) / THREADS;
	blocks = (blocks < MAXBLOCKNUM) ? blocks : MAXBLOCKNUM;
	applyKernel<<<blocks, THREADS>>>(a, b, c, func);
	sync();
	CHECK_CUDA;
}

// # ---------------------------------------------

template <class T>
__global__ void dotKernel(const SimpleTensor<T> a, const SimpleTensor<T> b, T* out) {
	T sum = 0;
	const uint64_t stride = blockDim.x * gridDim.x;
	uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
	T* baseA = a.data;
	T* baseB = b.data;

	if (a.dense && b.dense) {
		while (i < a.size) {
			sum += baseA[i + a.offset] * baseB[i + b.offset];
			i += stride;
		}
	} else {
		while (i < a.size) {
			uint64_t idxA = a.dense ? i + a.offset : computeLinearIdx(i, a);
			uint64_t idxB = b.dense ? i + b.offset : computeLinearIdx(i, b);

			sum += baseA[idxA] * baseB[idxB];
			i += stride;
		}
	}

	gpuAtomicAdd(out, sum);
}

// cuBLAS specialization for float dot product
template <>
float dot<float>(const SimpleTensor<float> a, const SimpleTensor<float> b) {
	float result = 0.0f;
	int incx = a.dense ? 1 : a.strides[0];
	int incy = b.dense ? 1 : b.strides[0];

	cublasSdot(getCublasHandle(), a.size, a.data, incx, b.data, incy, &result);

	return result;
}

// cuBLAS specialization for double dot product
template <>
double dot<double>(const SimpleTensor<double> a, const SimpleTensor<double> b) {
	double result = 0.0;
	int incx = a.dense ? 1 : a.strides[0];
	int incy = b.dense ? 1 : b.strides[0];

	cublasDdot(getCublasHandle(), a.size, a.data, incx, b.data, incy, &result);

	return result;
}


template <class T>
T dot(const SimpleTensor<T> a, const SimpleTensor<T> b) {
	T p = 0;
	T* out = static_cast<T*>(allocGPU(sizeof(T)));
	copyToGPU(out, &p, sizeof(T));

	uint64_t blocks = (a.size + THREADS - 1) / THREADS;
	blocks = (blocks < MAXBLOCKNUM) ? blocks : MAXBLOCKNUM;
	dotKernel<<<blocks, THREADS>>>(a, b, out);

	sync();
	CHECK_CUDA;
	copyToCPU(&p, out, sizeof(T));
	freeGPU(out, sizeof(T));
	return p;
}

// # ---------------------------------------------

// cuBLAS specialization for float (fp32)
template <>
void matmul(const SimpleTensor<float> A, const SimpleTensor<float> B, SimpleTensor<float> C) {
	float alpha = 1.0f;
	float beta = 0.0f;

	int m = A.shape[0];
	int k = A.shape[1];
	int n = B.shape[1];

	int lda = A.strides[0];
	int ldb = B.strides[0];
	int ldc = C.strides[0];


	cublasSgemm(getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B.data, ldb, A.data,
	            lda, &beta, C.data, ldc);
}

// cuBLAS specialization for double (fp64)
template <>
void matmul(const SimpleTensor<double> A, const SimpleTensor<double> B, SimpleTensor<double> C) {
	double alpha = 1.0;
	double beta = 0.0;

	int m = A.shape[0];
	int k = A.shape[1];
	int n = B.shape[1];

	int lda = A.strides[0];
	int ldb = B.strides[0];
	int ldc = C.strides[0];


	cublasDgemm(getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B.data, ldb, A.data,
	            lda, &beta, C.data, ldc);
}

template <class T>
void matmul(const SimpleTensor<T> A, const SimpleTensor<T> B, SimpleTensor<T> C) {
	TZ_CHECK(false, "matmul is only implemented for float and double types");
}

// # ============================================================================================

using namespace TZ::impl;
#define INSTANTIATE_UNARY_APPLY(T, Func)                                                           \
	template void applyGPU<T, Func<T>>(SimpleTensor<T>, Func<T>)

#define INSTANTIATE_BINARY_APPLY(T, Func)                                                          \
	template void applyGPU<T, Func<T>>(const SimpleTensor<T>, SimpleTensor<T>, Func<T>)

#define INSTANTIATE_TERNARY_APPLY(T, Func)                                                         \
	template void applyGPU<T, Func<T>>(const SimpleTensor<T>, const SimpleTensor<T>,               \
	                                   SimpleTensor<T>, Func<T>)


#define INSTANTIATE_COMPUTE_LINEAR_IDX(T)                                                          \
	template __device__ uint64_t computeLinearIdx<T>(uint64_t, const SimpleTensor<T>)

#define INSTANTIATE_DOT(T) template T dot<T>(const SimpleTensor<T> a, const SimpleTensor<T> b)

#define INSTANTIATE_MATMUL(T)                                                                      \
	template void matmul<T>(const SimpleTensor<T> a, const SimpleTensor<T> b, SimpleTensor<T> out)


#define INSTANTIATE_ALL(T)                                                                         \
	INSTANTIATE_UNARY_APPLY(T, Negate);                                                            \
	INSTANTIATE_UNARY_APPLY(T, Set);                                                               \
	INSTANTIATE_UNARY_APPLY(T, Sum);                                                               \
                                                                                                   \
	INSTANTIATE_BINARY_APPLY(T, Copy);                                                             \
	INSTANTIATE_BINARY_APPLY(T, AddScalar);                                                        \
	INSTANTIATE_BINARY_APPLY(T, SubtractScalar);                                                   \
	INSTANTIATE_BINARY_APPLY(T, MultiplyScalar);                                                   \
                                                                                                   \
	INSTANTIATE_TERNARY_APPLY(T, Add);                                                             \
	INSTANTIATE_TERNARY_APPLY(T, Subtract);                                                        \
                                                                                                   \
	INSTANTIATE_COMPUTE_LINEAR_IDX(T);                                                             \
                                                                                                   \
	INSTANTIATE_DOT(T);                                                                            \
                                                                                                   \
	INSTANTIATE_MATMUL(T);

// # --------------------

INSTANTIATE_ALL(int32_t)

INSTANTIATE_ALL(int64_t)

INSTANTIATE_ALL(uint32_t)

INSTANTIATE_ALL(uint64_t)

INSTANTIATE_ALL(float)

INSTANTIATE_ALL(double)


} // namespace TZ::cuda
