#include <iostream>                    // Standard input/output library
#include <cstdlib>                     // For rand() and srand()
#include <ctime>                       // For time() to seed randomness
#include <cuda_runtime.h>              // CUDA runtime API

// CUDA kernel to perform matrix multiplication
__global__ void matrixMul(float *A, float *B, float *C, int N) {
    int row = threadIdx.y + blockIdx.y * blockDim.y; // Calculate row index
    int col = threadIdx.x + blockIdx.x * blockDim.x; // Calculate column index

    if (row < N && col < N) {
        float value = 0;
        for (int k = 0; k < N; k++) {
            value += A[row * N + k] * B[k * N + col]; // Dot product for row x column
        }
        C[row * N + col] = value; // Store result
    }
}

int main() {
    int N;
    std::cout << "Enter matrix size N (NxN): ";
    std::cin >> N;                            // Take matrix size from user

    size_t size = N * N * sizeof(float);      // Calculate total memory needed

    float *A = (float*)malloc(size);          // Host matrix A
    float *B = (float*)malloc(size);          // Host matrix B
    float *C = (float*)malloc(size);          // Host matrix C (result)

    float *d_A, *d_B, *d_C;                   // Device pointers
    cudaMalloc(&d_A, size);                   // Allocate device memory for A
    cudaMalloc(&d_B, size);                   // Allocate device memory for B
    cudaMalloc(&d_C, size);                   // Allocate device memory for C

    srand(time(0));                           // Seed random number generator

    // Fill A and B with random float values between 0 and 10

    for (int i = 0; i < N * N; i++) {
        A[i] = rand() % 1000;                  // Random integer in range [0, 999]
        B[i] = rand() % 1000;                  // Random integer in range [0, 999]
    }

    // for (int i = 0; i < N * N; i++) {
    //     A[i] = static_cast<float>(rand() % 1000) / 100.0f; // Random float: 0.00 - 9.99
    //     B[i] = static_cast<float>(rand() % 1000) / 100.0f;
    // }

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice); // Copy A to device
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice); // Copy B to device

    dim3 threadsPerBlock(16, 16);                      // CUDA block of 16x16 threads
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);  // Grid size to cover N x N matrix

    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N); // Launch kernel

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);  // Copy result C back to host

    // Display matrix A
    std::cout << "\nMatrix A:\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            std::cout << A[i * N + j] << " ";
        std::cout << "\n";
    }

    // Display matrix B
    std::cout << "\nMatrix B:\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            std::cout << B[i * N + j] << " ";
        std::cout << "\n";
    }

    // Display matrix C with calculations
    std::cout << "\nCalculations (C[i][j] = A[i][k] * B[k][j]):\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << "C[" << i << "][" << j << "] = ";
            for (int k = 0; k < N; k++) {
                std::cout << A[i * N + k] << "*" << B[k * N + j];
                if (k < N - 1) std::cout << " + ";
            }
            std::cout << " = " << C[i * N + j] << "\n";
        }
    }

    // Free all allocated memory
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
