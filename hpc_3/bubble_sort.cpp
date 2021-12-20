#include <bits/stdc++.h>
#include <omp.h>
#include <chrono>
#define N 10000
using namespace std;

void bubbleSortSingleThread(vector<int> &arr) {
    int n = arr.size();
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n-i-1; j++) {
            if(arr[j] > arr[j+1]) swap(arr[j], arr[j+1]);
        }
    }
}

void bubbleSortMultiThread(vector<int> &arr) {
    for(int i = 0; i < N-1; i++) {
        int first = i%2;
        #pragma omp parallel for
        for(int j = first; j < N-1; j += 2) {
            if(arr[j] > arr[j+1]) swap(arr[j], arr[j+1]);
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
    cout << "BUBBLE SORT:" << endl;
    auto start = chrono::steady_clock::now();
    bubbleSortSingleThread(arr_1);
    auto end = chrono::steady_clock::now();
    cout << "SINGLE THREAD STATISTICS:" << endl;
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds" << endl;

    start = chrono::steady_clock::now();
    bubbleSortMultiThread(arr_2);
    end = chrono::steady_clock::now();
    cout << "MULTI THREAD STATISTICS:" << endl;
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds" << endl;
}