#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <time.h>
#include <chrono>
using namespace std;


__global__ void addGPU(int *x, int *y, int *z) {
    int idx = blockIdx.x;
    z[idx] = x[idx] + y[idx];
}

void addCPU(int *x, int *y, int *z, int n) {
    for(int i = 0; i < n; i++) z[i] = x[i] + y[i];
}

int main() {
    srand(time(0));
    int N = (1<<20);
    int *h_array_1, *h_array_2, *h_array_3;
    int *d_array_1, *d_array_2, *d_array_3;
    
    h_array_1 = new int[N];
    h_array_2 = new int[N];
    h_array_3 = new int[N];
    cudaMalloc((void**)&d_array_1, N * sizeof(int));
    cudaMalloc((void**)&d_array_2, N * sizeof(int));
    cudaMalloc((void**)&d_array_3, N * sizeof(int));

    for(int i = 0; i < N; i++) {
        h_array_1[i] = rand();
        h_array_2[i] = rand();
        h_array_3[i] = 0;
    }
    cudaMemcpy(d_array_1, h_array_1, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_array_2, h_array_2, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_array_3, h_array_3, N * sizeof(int), cudaMemcpyHostToDevice);

    auto start = chrono::steady_clock::now();
    addCPU(h_array_1, h_array_2, h_array_3, N);
    auto end = chrono::steady_clock::now();

    cout << "VECTOR ADDITION:\n\n";

    cout << "CPU STATISTICS:\n";
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds\n";

    start = chrono::steady_clock::now();
    addGPU<<<N, 1>>>(d_array_1, d_array_2, d_array_3);
    end = chrono::steady_clock::now();

    cout << "GPU STATISTICS:\n";
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds\n";

    return 0;
}
