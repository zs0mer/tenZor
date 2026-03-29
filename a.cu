#include <stdio.h>

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

int main() {
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

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
	}

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

	return 0;
}