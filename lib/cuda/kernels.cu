#include <stdio.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include "kernel_functions.hpp"

// # ---------------------------------------------

#define CHECK_CUDA check_cuda(__FILE__, __LINE__, __func__)

inline void check_cuda(const char* file, int line, const char* func) {

	if (cudaGetLastError() != cudaSuccess) {
		auto now = std::chrono::system_clock::now();
		std::time_t t_c = std::chrono::system_clock::to_time_t(now);

		std::ostringstream oss;
		oss << "Error: " << cudaGetErrorString(cudaGetLastError()) << "\n"
		    << "File: " << file << "\n"
		    << "Line: " << line << "\n"
		    << "Function: " << func << "\n"
		    << "Time: " << std::put_time(std::localtime(&t_c), "%F %T");

		throw std::runtime_error(oss.str());
	}
}

// # ------------------

#define CHECK(expr, error) check((expr), (error), __FILE__, __LINE__, __func__)
#define CHECK_(expr) check((expr), "unexpected", __FILE__, __LINE__, __func__)

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

// # ---------------------------------------------

void* GPUAlloc(const uint64_t bytes, const uint8_t alignment) {
	// * cudaMalloc() already alings memory well
	void* ptr = nullptr;
	cudaMalloc(&ptr, bytes);
	CHECK_CUDA;
	return ptr;
}

void GPUFree(void* ptr, const uint64_t bytes) {
	cudaFree(ptr);
	CHECK_CUDA;
	return;
}


/*
#define N 4 // small demo size (keep simple)

__global__ void matmul(int* A, int* B, int* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

void print_matrix(int* M, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%4d ", M[i * n + j]);
        }
        printf("\n");
    }
}

void calc() {
    int n = N;
    int size = n * n * sizeof(int);

    int A[N * N] = {1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 1, 1, 2, 2, 2, 2};

    int B[N * N] = {1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1};

    int C[N * N] = {0};

    int *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 threads(2, 2);
    dim3 blocks((n + 1) / 2, (n + 1) / 2);

    matmul<<<blocks, threads>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize(); // 🔥 WAIT for GPU to finish

    CHECK_CUDA;

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Matrix A:\n");
    print_matrix(A, n);

    printf("\nMatrix B:\n");
    print_matrix(B, n);

    printf("\nMatrix C = A x B:\n");
    print_matrix(C, n);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return;
}
*/
