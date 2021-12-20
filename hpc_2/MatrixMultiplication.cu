%%cu

#include <iostream>
#include <time.h>
#include <stdexcept>
#include <vector> 
#include <cstdlib>
#include <chrono>

#define r1 512
#define c1 512
#define r2 512
#define c2 512
using namespace std;


__global__ void matmul(int* a,int* b, int* c){
		
	int x = blockIdx.x;
	int y = blockIdx.y;

	c[c2 * y + x] = 0;
	for(int k = 0; k < c1; k++)
		c[c2 * y + x] += (a[c1 * y + k] * b[c2 * k + x]);

}
int main(void){

	srand(time(0));
	int a[r1][c1];
	int b[r2][c2];
	int c[r1][c2];
	

	for(int i = 0; i < r1; i++)
		for(int j = 0; j < c1; j++)
			a[i][j] = rand();

	for(int i = 0; i < r2; i++)
		for(int j = 0; j < c2; j++)
			b[i][j] = rand();

	int *p ,*q, *r;

	cudaMalloc((void**)&p,r1 * c1 * sizeof(int));
	cudaMalloc((void**)&q,r2 * c2 * sizeof(int));
	cudaMalloc((void**)&r,r1 * c2 * sizeof(int));

	cudaMemcpy(p,a,r1 * c1 * sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(q,b,r1 * c1 * sizeof(int),cudaMemcpyHostToDevice);

	dim3 grid(c2,r1); // cols * rows

  auto start = chrono::steady_clock::now();
  matmul<<<grid,1>>>(p,q,r);
  auto end = chrono::steady_clock::now();

	cudaMemcpy(c,r, r1 * c2 * sizeof(int),cudaMemcpyDeviceToHost);

  cout << "GPU STATISTICS:\n";
  cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds\n";

	cudaFree(p);
	cudaFree(q);
	cudaFree(r);
	return 0;
}
