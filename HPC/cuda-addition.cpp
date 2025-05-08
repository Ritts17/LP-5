#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

using namespace std;

// Serial vector addition function
void vectorAddSerial(const int* A, const int* B, int* C, int N) {
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

// CUDA kernel for parallel vector addition
_global_ void vectorAddParallel(const int* A, const int* B, int* C, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N) {
        C[index] = A[index] + B[index];
    }
}

int main() {
    int N;
    cout << "Enter the size of the vectors: ";
    cin >> N;

    // Allocate host memory for vectors A, B, and C
    int* h_A = (int*)malloc(N * sizeof(int));
    int* h_B = (int*)malloc(N * sizeof(int));
    int* h_C_serial = (int*)malloc(N * sizeof(int));
    int* h_C_parallel = (int*)malloc(N * sizeof(int));

    // Initialize vectors A and B with random values
    srand(time(0));
    for (int i = 0; i < N; ++i) {
        h_A[i] = rand() % 1000;
        h_B[i] = rand() % 1000;
    }

    // Serial vector addition
    double start = clock();
    vectorAddSerial(h_A, h_B, h_C_serial, N);
    double end = clock();
    double serialTime = (end - start) / CLOCKS_PER_SEC;
    cout << "Serial Vector Addition Time: " << serialTime << " seconds\n";

    // Allocate device memory
    int* d_A;
    int* d_B;
    int* d_C;
    cudaMalloc((void**)&d_A, N * sizeof(int));
    cudaMalloc((void**)&d_B, N * sizeof(int));
    cudaMalloc((void**)&d_C, N * sizeof(int));

    // Copy vectors A and B from host to device
    cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice);

    // Define block size and grid size
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Parallel vector addition using CUDA
    start = clock();
    vectorAddParallel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();  // Ensure all threads have finished execution
    end = clock();
    double parallelTime = (end - start) / CLOCKS_PER_SEC;
    cout << "Parallel Vector Addition Time (CUDA): " << parallelTime << " seconds\n";

    // Copy the result from device to host
    cudaMemcpy(h_C_parallel, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Check if the results are the same (serial vs parallel)
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (h_C_serial[i] != h_C_parallel[i]) {
            correct = false;
            break;
        }
    }

    if (correct) {
        cout << "Results match between serial and parallel versions!" << endl;
    } else {
        cout << "Results do not match!" << endl;
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C_serial);
    free(h_C_parallel);

    return 0;
}