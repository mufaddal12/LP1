#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <time.h>
#include <chrono>
#include <math.h>
using namespace std;


__global__ void getSumGPU(int *arr, int N, int mul, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = mul * idx, j = i + offset;
    if(j < N) arr[i] += arr[j];
}

__global__ void getStdDevGPU(int *arr, int N, int mean, int mul, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = mul * idx, j = i + offset;
    if(j < N) arr[i] = (arr[i]-mean)*(arr[i]-mean) + (arr[j]-mean)*(arr[j]-mean);
    else if(i < N) arr[i] = (arr[i]-mean)*(arr[i]-mean);
}

void sumGPU(int *arr, int n, int *ans) {
    int mul = 2, offset = 1, N = n;
    while(n > 1) {
        n = (n + 1) / 2;
        int blocks = (n-1) / 256 + 1;
        int threads = 256;
        getSumGPU<<<blocks, threads>>>(arr, N, mul, offset);
        mul *= 2;
        offset *= 2;
    }
    cudaMemcpy(ans, arr, sizeof(int), cudaMemcpyDeviceToHost);
}

void stdDevGPU(int *arr, int n, int mean, int *ans) {
    int mul = 2, offset = 1, N = n;
    bool first = true;
    while(n > 1) {
        n = (n + 1) / 2;
        int blocks = (n-1) / 256 + 1;
        int threads = 256;
        if(first) {
            getStdDevGPU<<<blocks, threads>>>(arr, N, mean, mul, offset);
            first =  false;
        }
        else getSumGPU<<<blocks, threads>>>(arr, N, mul, offset);
        mul *= 2;
        offset *= 2;
    }
    cudaMemcpy(ans, arr, sizeof(int), cudaMemcpyDeviceToHost);
    (*ans) = sqrt((*ans) / N);
}

void sumCPU(int *arr, int n, int *ans) {
    *ans = 0;
    for(int i = 0; i < n; i++) {
        (*ans) += arr[i];
    }
}

void stdDevCPU(int *arr, int n, int mean, int *ans) {
    *ans = 0;
    for(int i = 0; i < n; i++) (*ans) += (arr[i]-mean) * (arr[i]-mean);
    (*ans) = sqrt((*ans) / n);
}

int main() {
    srand(time(0));
    int N = (1<<19);
    int *h_array, *d_array;
    
    h_array = new int[N];
    cudaMalloc((void**)&d_array, N * sizeof(int));

    int maxVal = 1e3;
    for(int i = 0; i < N; i++) h_array[i] = rand() % maxVal;
    cudaMemcpy(d_array, h_array, N * sizeof(int), cudaMemcpyHostToDevice);

    int *ans_CPU, *ans_GPU, meanCPU, meanGPU;
    ans_CPU = new int;
    ans_GPU = new int;

    auto start = chrono::steady_clock::now();
    sumCPU(h_array, N, ans_CPU);
    meanCPU = (*ans_CPU) / N;
    auto end = chrono::steady_clock::now();

    cout << "CPU STATISTICS:\n";
    cout << "Sum value: " << *ans_CPU << '\n';
    cout << "Average value: " << meanCPU << '\n';
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds\n";

    start = chrono::steady_clock::now();
    stdDevCPU(h_array, N, meanCPU, ans_CPU);
    end = chrono::steady_clock::now();

    cout << "Standard deviation value: " << *ans_CPU << '\n';
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds\n\n";

    start = chrono::steady_clock::now();
    sumGPU(d_array, N, ans_GPU);
    meanGPU = (*ans_GPU) / N;
    end = chrono::steady_clock::now();

    cout << "GPU STATISTICS:\n";
    cout << "Sum value: " << *ans_GPU << '\n';
    cout << "Average value: " << meanGPU << '\n';
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds\n";

    cudaMemcpy(d_array, h_array, N * sizeof(int), cudaMemcpyHostToDevice);

    start = chrono::steady_clock::now();
    stdDevGPU(d_array, N, meanGPU, ans_GPU);
    end = chrono::steady_clock::now();

    cout << "Standard deviation value: " << *ans_GPU << '\n';
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds\n";

    return 0;
}
