// #include "LinearSolver.hh"
// #include "table.hh"

#include <vector>
#include <iostream>
#include <cmath> 

using namespace std; 

#define THREADS_PER_BLOCK 1024

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void gpu(int * a, int * b, int * dot) {

	__shared__ int product[THREADS_PER_BLOCK]; 

	int tid = blockIdx.x * blockDim.x + threadIdx.x; 

	product[threadIdx.x] = a[tid] * b[tid]; 

	__syncthreads(); 

	if (threadIdx.x == 0) {

		int sum = 0; 

		for (int i=0; i<THREADS_PER_BLOCK; i++) {

			sum += product[i]; 

		}

		atomicAdd(dot, sum); 

	}

}

int main(int argc, char * argv[]) {

	int N = atoi(argv[1]); 

	// int threadsPerBlock = pow(2, 2); 
	// threadsPerBlock = min(N, threadsPerBlock); 
	// int nBlock = max(1, (N + threadsPerBlock - 1)/threadsPerBlock);  

	// cout << "threads/block = " << threadsPerBlock << " number of blocks = " << nBlock << endl; 

	int *a, *b; 
	int *d_a, *d_b; 
	int dot, *d_dot; 

	size_t size = N*sizeof(int); 

	cudaMalloc(&d_a, size); 
	cudaMalloc(&d_b, size); 
	cudaMalloc(&d_dot, sizeof(int)); 

	a = new int[N]; 
	b = new int[N]; 

	dot = 0; 

	for (int i=0; i<N; i++) {

		a[i] = 1; 
		b[i] = 1; 

	}

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice); 
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dot, &dot, sizeof(int), cudaMemcpyHostToDevice); 

	gpu<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_dot);

	cudaMemcpy(&dot, d_dot, sizeof(int), cudaMemcpyDeviceToHost);  

	cudaFree(d_a); cudaFree(d_b); cudaFree(d_dot);
	delete a, b;  

}

// int main() {

// 	int tend = 1; 
// 	int Nt = 10; 
// 	int y0 = 1; 

// 	int dt = tend/Nt; 

// 	auto f = [] (int y) {return -1*pow(y, 3); }; 

// 	vector<int> t = linspace(0, tend, Nt+1); 

// 	vector<int> y(Nt+1); 
// 	y[0] = y0; 

// 	vector<int> y_ex(Nt+1); 
// 	y_ex[0] = y0; 

// 	int tol = 1e-6; 
// 	int maxiter = 100; 

// 	for (int i=1; i<=Nt; i++) {

// 		int yi = y[i-1];
// 		int ylag = yi; 

// 		for (int j=0; j<maxiter; j++) {

// 			y[i] = yi + dt * f(.5*(ylag + yi)); 

// 			if (abs(ylag - y[i]) < tol) break; 

// 			if (j == maxiter - 1) cout << "max iterations reached" << endl; 

// 			ylag = y[i]; 

// 		} 

// 		y_ex[i] = y0*pow(1/(1 + 2*pow(y0,2)*t[i]), .5); 

// 	}

// 	Table table("out"); 

// 	table.addColumn(t); 
// 	table.addColumn(y); 
// 	table.addColumn(y_ex); 

// 	table.write(); 

// }