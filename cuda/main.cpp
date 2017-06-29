#include "LinearSolver.hh"
#include "helper.hh"
#include "table.hh"
#include "LinearAlgebra.hh"

#include <vector>
#include <iostream>
#include <cmath> 

using namespace std; 

vector<double> getFunction(vector<double> &y) {

	vector<double> f(2); 

	f[0] = sin(y[0]); 

	f[1] = cos(y[1]); 

}

vector<vector<double>> getJacobian(vector<double> &y) {

	vector<vector<double>> J(2, vector<double>(2)); 
	J[0][0] = cos(y[0]); 
	J[0][1] = 0; 
	J[1][0] = 0; 
	J[1][1] = -sin(y[1]); 

	return J; 

}

int main() {

	int N = 1000; 
	double tend = 1; 

	vector<double> t = linspace(0, tend, N+1); 

	vector<vector<double>> y(N+1, vector<double>(2)); 

	vector<vector<double>> I = Identity(2); 

	for (int i=1; i<N+1; i++) {

		double dt = t[i] - t[i-1]; 

		vector<vector<double>> J = getJacobian(y[i-1]); 
		J = J*dt; 
		vector<double> f = getFunction(y[i-1]); 
		
		// lhs 
		vector<vector<double>> A = I - J; 

		// rhs 
		vector<double> b = y[i-1] + f*dt - J*y[i-1]; 

	}

}