#include <bits/stdc++.h>
#include <chrono>
#include <mpi.h>
#define N 10000
#define MAX_VAL 1000000
#define IDX 9912
#define SEND_TAG 2001
#define RECEIVE_TAG 2002
#define ROOT_PROCESS 0
using namespace std;

int master_array[N];
int slave_array[N];
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

void send_data(int id, int num_procs) {
	int start = N / num_procs * id;
	int length = (id < num_procs-1) ? N / num_procs : N - start;
	MPI_Send(&length, 1, MPI_INT, id, SEND_TAG, MPI_COMM_WORLD);
	MPI_Send(&master_array[start], length, MPI_INT, id, SEND_TAG, MPI_COMM_WORLD);
	MPI_Send(&element_to_search, 1, MPI_INT, id, SEND_TAG, MPI_COMM_WORLD);
}

void receive_data(int id, int num_procs, MPI_Status status) {
	int length, search_element;
	MPI_Recv(&length, 1, MPI_INT, ROOT_PROCESS, SEND_TAG, MPI_COMM_WORLD, &status);
	MPI_Recv(&slave_array, length, MPI_INT, ROOT_PROCESS, SEND_TAG, MPI_COMM_WORLD, &status);
	MPI_Recv(&search_element, 1, MPI_INT, ROOT_PROCESS, SEND_TAG, MPI_COMM_WORLD, &status);
	int idx = binary_search(slave_array, 0, length, search_element);
	if(idx != -1) {
		int i = N / num_procs * id + idx;
		cout << "Element found at process " << id << " at: " << i << "." << endl;
	}
}

int main(int argc, char** argv) {
	MPI_Status status;

	int id, num_procs;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	if(id == ROOT_PROCESS) {
		generate_array();
		auto start = chrono::steady_clock::now();
		for(int i = 1; i < num_procs; i++) {
			send_data(i, num_procs);
		}
		int idx = binary_search(master_array, 0, N / num_procs - 1, element_to_search);
    auto end = chrono::steady_clock::now();
		if(idx != -1) cout << "Element found at root process at: " << idx << "." << endl;
		cout << "MPI STATISTICS:" << endl;
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds" << endl;
	}
	else {
		receive_data(id, num_procs, status);
	}
	MPI_Finalize();
}