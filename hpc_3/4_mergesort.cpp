#include <bits/stdc++.h>
#include <omp.h>
#include <chrono>
#define N 100000
using namespace std;

void merge(vector<int> &arr, int low, int mid, int high) {
    vector<int> a(arr.begin()+low, arr.begin()+mid+1);
    vector<int> b(arr.begin()+mid+1, arr.begin()+high+1);
    vector<int> res;
    int i = 0, j = 0;
    while(i < a.size() && j < b.size()) {
        if(a[i] < b[j]) {
            res.push_back(a[i]);
            i++;
        }
        else {
            res.push_back(b[j]);
            j++;
        }
    }
    for(; i < a.size(); i++) res.push_back(a[i]);
    for(; j < b.size(); j++) res.push_back(b[j]);
    for(i = 0; i < res.size(); i++) arr[low + i] = res[i];
}

void mergeSortSingleThread(vector<int> &arr, int low, int high) {
    if(low < high) {
        int mid = (low + high) / 2;
        mergeSortSingleThread(arr, low, mid);
        mergeSortSingleThread(arr, mid + 1, high);
        merge(arr, low, mid, high);
    }
}

void mergeSortMultiThread(vector<int> &arr, int low, int high) {
    if(low < high) {
        int mid = (low + high) / 2;
        #pragma omp parallel sections 
        {
            #pragma omp section 
            {
                mergeSortSingleThread(arr, low, mid);
            }
            #pragma omp section 
            {
                mergeSortSingleThread(arr, mid + 1, high);
            }
        }
        merge(arr, low, mid, high);
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
    mergeSortSingleThread(arr_1, 0, N-1);
    auto end = chrono::steady_clock::now();
    cout << "SINGLE THREAD STATISTICS:" << endl;
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds" << endl;

    start = chrono::steady_clock::now();
    mergeSortMultiThread(arr_2, 0, N-1);
    end = chrono::steady_clock::now();
    cout << "MULTI THREAD STATISTICS:" << endl;
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds" << endl;
}