#include <bits/stdc++.h>
#include <chrono>
#include <mpi.h>
#define N 127
#define TREES 4
#define SEND_TAG 2001
#define RECEIVE_TAG 2002
#define ROOT_PROCESS 0
using namespace std;

int master_array[N];
int slave_array[N];

void generate_tree(int low, int high, int idx, int offset) {
	if(low <= high && idx < N) {
		int mid = (low + high) / 2;
		master_array[idx] = mid + offset;
		generate_tree(low, mid-1, 2*idx+1, offset);
		generate_tree(mid+1, high, 2*idx+2, offset);
	}
}

void inorder(int *array, int low, int high) {
	if(low <= high) {
		inorder(array, low*2+1, high);
		cout << array[low] << ' ';
		inorder(array, low*2+2, high);
	}
}

int main(int argc, char** argv) {
	MPI_Status status;

	int id, num_procs;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	if(id == ROOT_PROCESS) {
		auto start = chrono::steady_clock::now();
		for(int i = 1; i < num_procs; i++) {
			generate_tree(0, N-1, 0, i*100);
			MPI_Send(&master_array[0], N, MPI_INT, i, SEND_TAG, MPI_COMM_WORLD);
		}
		cout << "Root process: ";
		generate_tree(0, N-1, 0, 0);
		inorder(master_array, 0, N-1);
		cout << endl;
    	auto end = chrono::steady_clock::now();
		cout << "MPI STATISTICS:" << endl;
    	cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds" << endl;
	}
	else {
		MPI_Recv(&slave_array, N, MPI_INT, ROOT_PROCESS, SEND_TAG, MPI_COMM_WORLD, &status);
		cout << "Process " << id << ": ";
		inorder(slave_array, 0, N-1);
		cout << endl;
	}
	MPI_Finalize();
}
