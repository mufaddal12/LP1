#include <bits/stdc++.h>
#include <chrono>
#define N 10000
#define MAX_VAL 1000000
#define IDX 9912

using namespace std;

int master_array[N];
int element_to_search;

void generate_array() {
	srand(time(0));
	for(int i = 0; i < N; i++) master_array[i] = rand() % MAX_VAL;
	sort(master_array, master_array + N);
	element_to_search = master_array[IDX];
}

int binary_search(int arr[], int low, int high, int x) {
	while(low <= high) {
		int mid = (low + high) / 2;
		if(arr[mid] == x) return mid;
		else if(arr[mid] < x) low = mid + 1;
		else high = mid - 1;
	}
	return -1;
}

int main() {
	generate_array();
	auto start = chrono::steady_clock::now();
	int idx = binary_search(master_array, 0, N - 1, element_to_search);
	auto end = chrono::steady_clock::now();
	if(idx != -1) cout << "Element found at root process at: " << idx << "." << endl;
	cout << "MPI STATISTICS:" << endl;
	cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds" << endl;
}