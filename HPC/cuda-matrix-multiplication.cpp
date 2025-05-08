#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

using namespace std;

// -------------------- Serial Matrix Multiplication --------------------
void serialMatrixMultiply(int *A, int *B, int *C, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            int sum = 0;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

// -------------------- CUDA Kernel for Matrix Multiplication --------------------
_global_ void matrixMulKernel(int *A, int *B, int *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// -------------------- Utility Functions --------------------
void initializeMatrix(int *matrix, int N) {
    for (int i = 0; i < N * N; ++i)
        matrix[i] = rand() % 10;
}

bool isEqual(int *a, int *b, int N) {
    for (int i = 0; i < N * N; ++i)
        if (a[i] != b[i])
            return false;
    return true;
}

// -------------------- Main Function --------------------
int main() {
    int N;
    cout << "Enter matrix size (N x N): ";
    cin >> N;

    size_t bytes = N * N * sizeof(int);

    int h_A = (int)malloc(bytes);
    int h_B = (int)malloc(bytes);
    int h_C_serial = (int)malloc(bytes);
    int h_C_cuda = (int)malloc(bytes);

    srand(time(0));
    initializeMatrix(h_A, N);
    initializeMatrix(h_B, N);

    // ---------- Serial Execution ----------
    clock_t start = clock();
    serialMatrixMultiply(h_A, h_B, h_C_serial, N);
    clock_t end = clock();
    double serialTime = double(end - start) / CLOCKS_PER_SEC;
    cout << "Serial Execution Time: " << serialTime << " seconds\n";

    // ---------- CUDA Execution ----------
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((N + 15) / 16, (N + 15) / 16);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent);
    matrixMulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stopEvent);

    cudaMemcpy(h_C_cuda, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    float cudaTime;
    cudaEventElapsedTime(&cudaTime, startEvent, stopEvent);

    cout << "CUDA Execution Time: " << cudaTime / 1000.0 << " seconds\n";

    // ---------- Verification ----------
    if (isEqual(h_C_serial, h_C_cuda, N))
        cout << "Result:  Matrices are equal!\n";
    else
        cout << "Result:  Mismatch in matrix results!\n";

    // ---------- Cleanup ----------
    free(h_A); free(h_B); free(h_C_serial); free(h_C_cuda);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}