#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <time.h>
#include <chrono>
#define N 512
#define M 512
using namespace std;


__global__ void addGPU(int *x, int *y, int *z) {
    int idx = blockIdx.x * M + blockIdx.y;
    z[idx] = x[idx] + y[idx];
}

void addCPU(int x[N][M], int y[N][M], int z[N][M]) {
    for(int i = 0; i < N; i++) for(int j = 0; j < M; j++) z[i][j] = x[i][j] + y[i][j];
}

int main() {
    srand(time(0));
    int h_array_1[N][M], h_array_2[N][M], h_array_3[N][M];
    int *d_array_1, *d_array_2, *d_array_3;

    cudaMalloc((void**)&d_array_1, N * M * sizeof(int));
    cudaMalloc((void**)&d_array_2, N * M * sizeof(int));
    cudaMalloc((void**)&d_array_3, N * M * sizeof(int));

    for(int i = 0; i < N; i++) for(int j = 0; j < M; j++) {
        h_array_1[i][j] = rand() % 1000;
        h_array_2[i][j] = rand() % 1000;
        h_array_3[i][j] = 0;
    }
    cudaMemcpy(d_array_1, h_array_1, N * M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_array_2, h_array_2, N * M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_array_3, h_array_3, N * M * sizeof(int), cudaMemcpyHostToDevice);

    auto start = chrono::steady_clock::now();
    addCPU(h_array_1, h_array_2, h_array_3);
    auto end = chrono::steady_clock::now();

    cout << "MATRIX ADDITION:\n\n";

    cout << "CPU STATISTICS:\n";
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds\n";
    
    dim3 grid(N,M);
    start = chrono::steady_clock::now();
    addGPU<<<grid, 1>>>(d_array_1, d_array_2, d_array_3);
    end = chrono::steady_clock::now();

    cout << "GPU STATISTICS:\n";
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds\n";

    return 0;
}