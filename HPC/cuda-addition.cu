#include <iostream>              // For input/output operations
#include <cstdlib>               // For rand() and srand()
#include <ctime>                 // For seeding rand() with time
#include <cuda_runtime.h>        // CUDA runtime API for device memory and kernel launch

// CUDA kernel function to perform vector addition in parallel
__global__ void vectorAdd(int* A, int* B, int* C, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Calculate global thread index
    if (i < N) C[i] = A[i] + B[i];                 // Perform addition if within bounds
}

int main() {
    int N;
    std::cout << "Enter size of vectors: ";
    std::cin >> N;
    int size = N * sizeof(int);

    // Allocate memory on host (CPU) for vectors A, B, and C
    int *A = (int*)malloc(size);        // Vector A
    int *B = (int*)malloc(size);        // Vector B
    int *C = (int*)malloc(size);        // Vector to store results (A + B)

    // Device (GPU) pointers
    int *d_A, *d_B, *d_C;

    // Allocate memory on the device (GPU) for vectors A, B, and C
    cudaMalloc(&d_A, size);             // Device memory for A
    cudaMalloc(&d_B, size);             // Device memory for B
    cudaMalloc(&d_C, size);             // Device memory for result C

    // Initialize random seed
    srand(time(0));

    // Fill host vectors A and B with random integers between 0 and 99
    for (int i = 0; i < N; i++) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

    // Copy host vectors A and B to device memory
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);  // Copy A from host to device
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);  // Copy B from host to device

    // Launch CUDA kernel: number of blocks = ceil(N / 256), each block has 256 threads
    vectorAdd<<<(N + 255) / 256, 256>>>(d_A, d_B, d_C, N);

    // Copy result vector C from device to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Print vector A
    std::cout << "A:\t"; 
    for (int i = 0; i < N; i++) std::cout << A[i] << " ";

    // Print vector B
    std::cout << "\nB:\t"; 
    for (int i = 0; i < N; i++) std::cout << B[i] << " ";

    // Print vector C (A + B)
    std::cout << "\nA+B:\t"; 
    for (int i = 0; i < N; i++) std::cout << C[i] << " ";
    std::cout << "\n";

    // Free allocated memory on the device
    cudaFree(d_A); 
    cudaFree(d_B); 
    cudaFree(d_C);

    // Free allocated memory on the host
    free(A); 
    free(B); 
    free(C);

    return 0;  // End of program
}
