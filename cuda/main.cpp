#include "LinearSolver.hh"
#include "helper.hh"
#include "table.hh"
#include "LinearAlgebra.hh"

#include <vector>
#include <iostream>
#include <fstream>
#include <cmath> 

using namespace std; 

vector<double> getFunction(vector<double> &y) {

	vector<double> f(y.size()); 

	for (int i=0; i<y.size(); i++) {

		f[i] = -(i+1) * pow(y[i], 2); 

	}

	return f; 
}

vector<vector<double>> getJacobian(vector<double> &y) {

	vector<vector<double>> J(y.size(), vector<double>(y.size()));

	for (int i=0; i<y.size(); i++) {

		J[i][i] = -2*(i+1)*y[i]; 

	} 

	return J; 

}

int main() {

	int N = 100; 
	double tend = 10; 
	int Neq = 50; 

	vector<double> t = linspace(0, tend, N+1); 

	vector<vector<double>> y(N+1, vector<double>(Neq)); 
	for (int i=0; i<Neq; i++) {

		y[0][i] = (double) (i+1)/Neq;  
		// y[0][i] = 1; 

	}

	vector<vector<double>> I = Identity(Neq); 

	for (int i=1; i<N+1; i++) {

		double dt = t[i] - t[i-1]; 

		vector<vector<double>> J = getJacobian(y[i-1]); 
		J = J*dt; 
		vector<double> f = getFunction(y[i-1]); 
		
		// lhs 
		vector<vector<double>> A = I - J; 

		// rhs 
		vector<double> b = y[i-1] + f*dt - J*y[i-1]; 

		int status = gauss_elim(Neq, A, y[i], b); 

		if (status != 0) cout << "linear solver error" << endl; 

	}

	ofstream file; 
	file.open("out"); 

	for (int i=0; i<N+1; i++) {

		file << t[i] << " "; 

		for (int j=0; j<Neq; j++) {

			file << y[i][j]; 

			if (j != Neq - 1) file << " "; 

		}

		file << endl; 

	}

	file.close(); 

}