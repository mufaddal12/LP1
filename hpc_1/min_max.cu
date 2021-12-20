#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <time.h>
#include <chrono>
using namespace std;


__global__ void getMinGPU(int *arr, int N, int mul, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = mul * idx, j = i + offset;
    if(j < N) if(arr[i] > arr[j]) arr[i] = arr[j];
}

void minimumGPU(int *arr, int n, int *ans) {
    int mul = 2, offset = 1, N = n;
    while(n > 1) {
        n = (n + 1) / 2;
        int blocks = (n-1) / 256 + 1;
        int threads = 256;
        getMinGPU<<<blocks, threads>>>(arr, N, mul, offset);
        mul *= 2;
        offset *= 2;
    }
    cudaMemcpy(ans, arr, sizeof(int), cudaMemcpyDeviceToHost);
}

void minimumCPU(int *arr, int n, int *ans) {
    *ans = INT_MAX;
    for(int i = 0; i < n; i++) {
        if(arr[i] < (*ans)) {
            (*ans) = arr[i];
        }
    }
}

__global__ void getMaxGPU(int *arr, int N, int mul, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = mul * idx, j = i + offset;
    if(j < N) if(arr[i] < arr[j]) arr[i] = arr[j];
}

void maximumGPU(int *arr, int n, int *ans) {
    int mul = 2, offset = 1, N = n;
    while(n > 1) {
        n = (n + 1) / 2;
        int blocks = (n-1) / 256 + 1;
        int threads = 256;
        getMaxGPU<<<blocks, threads>>>(arr, N, mul, offset);
        mul *= 2;
        offset *= 2;
    }
    cudaMemcpy(ans, arr, sizeof(int), cudaMemcpyDeviceToHost);
}

void maximumCPU(int *arr, int n, int *ans) {
    *ans = INT_MIN;
    for(int i = 0; i < n; i++) {
        if(arr[i] > (*ans)) {
            (*ans) = arr[i];
        }
    }
}

int main() {
    srand(time(0));
    int N = (1<<20);
    int *h_array, *d_array;
    
    h_array = new int[N];
    cudaMalloc((void**)&d_array, N * sizeof(int));

    for(int i = 0; i < N; i++) h_array[i] = rand();
    cudaMemcpy(d_array, h_array, N * sizeof(int), cudaMemcpyHostToDevice);

    int *ans_CPU, *ans_GPU;
    ans_CPU = new int;
    ans_GPU = new int;

    auto start = chrono::steady_clock::now();
    minimumCPU(h_array, N, ans_CPU);
    auto end = chrono::steady_clock::now();

    cout << "CPU STATISTICS:\n";
    cout << "Minimum value: " << *ans_CPU << '\n';
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds\n";

    start = chrono::steady_clock::now();
    maximumCPU(h_array, N, ans_CPU);
    end = chrono::steady_clock::now();

    cout << "Maximum value: " << *ans_CPU << '\n';
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds\n\n";

    start = chrono::steady_clock::now();
    minimumGPU(d_array, N, ans_GPU);
    end = chrono::steady_clock::now();

    cout << "GPU STATISTICS:\n";
    cout << "Minimum value: " << *ans_GPU << '\n';
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds\n";

    cudaMemcpy(d_array, h_array, N * sizeof(int), cudaMemcpyHostToDevice);

    start = chrono::steady_clock::now();
    maximumGPU(d_array, N, ans_GPU);
    end = chrono::steady_clock::now();

    cout << "Maximum value: " << *ans_GPU << '\n';
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds\n";

    return 0;
}

