#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <time.h>
#include <chrono>
#define N1 512
#define M1 512
#define N2 512
#define M2 512
using namespace std;


__global__ void multiplyGPU(int *x, int *y, int *z) {
    int i = blockIdx.x;
    int j = blockIdx.y;
    for(int k = 0; k < M1; k++) z[M2*i + j] += x[M1*i + k] * y[M2*k + j];
}

void multiplyCPU(int x[N1][M1], int y[N2][M2], int z[N1][M2]) {
    for(int i = 0; i < N1; i++) for(int j = 0; j < M2; j++)
      for(int k = 0; k < M1; k++) z[i][j] += x[i][k] * y[k][j];
}

int main() {
    srand(time(0));
    int h_array_1[N1][M1], h_array_2[N2][M2], h_array_3[N1][M2];
    int *d_array_1, *d_array_2, *d_array_3;

    cudaMalloc((void**)&d_array_1, N1 * M1 * sizeof(int));
    cudaMalloc((void**)&d_array_2, N2 * M2 * sizeof(int));
    cudaMalloc((void**)&d_array_3, N1 * M2 * sizeof(int));

    for(int i = 0; i < N1; i++) for(int j = 0; j < M1; j++) h_array_1[i][j] = rand() % 1000;
    for(int i = 0; i < N2; i++) for(int j = 0; j < M2; j++) h_array_2[i][j] = rand() % 1000;
    for(int i = 0; i < N1; i++) for(int j = 0; j < M2; j++) h_array_3[i][j] = 0;

    cudaMemcpy(d_array_1, h_array_1, N1 * M1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_array_2, h_array_2, N2 * M2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_array_3, h_array_3, N1 * M2 * sizeof(int), cudaMemcpyHostToDevice);

    auto start = chrono::steady_clock::now();
    multiplyCPU(h_array_1, h_array_2, h_array_3);
    auto end = chrono::steady_clock::now();

    cout << "MATRIX MULTIPLICATION:\n\n";

    cout << "CPU STATISTICS:\n";
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds\n";
    
    dim3 grid(N1,M2);
    start = chrono::steady_clock::now();
    multiplyGPU<<<grid, 1>>>(d_array_1, d_array_2, d_array_3);
    end = chrono::steady_clock::now();

    cout << "GPU STATISTICS:\n";
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds\n";

    return 0;
}

