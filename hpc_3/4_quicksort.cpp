#include <bits/stdc++.h>
#include <omp.h>
#include <chrono>
#define N 100000
using namespace std;

int partition(vector<int> &arr, int low, int high) {
    int len = high - low + 1;
    int pivot = rand() % len + low;
    swap(arr[pivot], arr[low]);
    int x = low + 1;
    for(int i = low + 1; i <= high; i++) {
        if(arr[i] < arr[low]) {
            swap(arr[i], arr[x]);
            x++;
        }
    }
    swap(arr[low], arr[x-1]);
    return x-1;
}

void quicksortSingleThread(vector<int> &arr, int low, int high) {
    int pivot = partition(arr, low, high);
    if(pivot > low + 1) quicksortSingleThread(arr, low, pivot - 1);
    if(pivot < high - 1) quicksortSingleThread(arr, pivot + 1, high);
}

void quicksortMultiThread(vector<int> &arr, int low, int high) {
    int pivot = partition(arr, low, high);
    #pragma omp parallel sections 
    {
        #pragma omp section 
        {
            if(pivot > low + 1) quicksortMultiThread(arr, low, pivot - 1);
        }
        #pragma omp section 
        {
            if(pivot < high - 1) quicksortMultiThread(arr, pivot + 1, high);
        }
    }    
}

int main() {
    srand(time(0));
    vector<int> arr_1(N), arr_2(N);
    for(int i = 0; i < N; i++) {
        int x = rand() % 10000;
        arr_1[i] = arr_2[i] = x;
    }

    auto start = chrono::steady_clock::now();
    quicksortSingleThread(arr_1, 0, N-1);
    auto end = chrono::steady_clock::now();
    cout << "SINGLE THREAD STATISTICS:\n";
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds\n";

    start = chrono::steady_clock::now();
    quicksortMultiThread(arr_2, 0, N-1);
    end = chrono::steady_clock::now();
    cout << "MULTI THREAD STATISTICS:\n";
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds\n";
}