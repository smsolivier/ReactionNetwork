// #include "LinearSolver.hh"
// #include "helper.hh"
// #include "table.hh"

#include <vector>
#include <iostream>
#include <cmath> 

using namespace std; 

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void gpu(vector<double> &y) {


}

int main() {

	int Nt = 10; 

	vector<double> y(Nt+1); 

	gpu<<<10, 4>>>(&y[0]); 

}

// int main() {

// 	double tend = 1; 
// 	int Nt = 10; 
// 	double y0 = 1; 

// 	double dt = tend/Nt; 

// 	auto f = [] (double y) {return -1*pow(y, 3); }; 

// 	vector<double> t = linspace(0, tend, Nt+1); 

// 	vector<double> y(Nt+1); 
// 	y[0] = y0; 

// 	vector<double> y_ex(Nt+1); 
// 	y_ex[0] = y0; 

// 	double tol = 1e-6; 
// 	int maxiter = 100; 

// 	for (int i=1; i<=Nt; i++) {

// 		double yi = y[i-1];
// 		double ylag = yi; 

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