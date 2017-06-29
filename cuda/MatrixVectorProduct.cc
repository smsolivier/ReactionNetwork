#include "LinearAlgebra.hh"

// matrix vector product 
void multiply(vector<vector<double>> &A, vector<double> &x, vector<double> &b) {

	int N = x.size(); 

	for (int i=0; i<N; i++) {

		for (int j=0; j<N; j++) {

			b[i] += A[i][j] * x[j]; 

		}

	}

}

// matrix scalar product 
void multiply(vector<vector<double>> &A, double x) {

	for (int i=0; i<A.size(); i++) {

		for (int j=0; j<A.size(); j++) {

			A[i][j] *= x; 

		}

	}

}

// matrix matrix addition 
void add(vector<vector<double>> &A, vector<vector<double>> &B) {

	int N = A.size(); 

	for (int i=0; i<N; i++) {

		for (int j=0; j<N; j++) {

			A[i][j] = A[i][j] + B[i][j]; 

		}

	}

}

// vector vector addition 
void add(vector<double> &A, vector<double> &B) {

	for (int i=0; i<A.size(); i++) {

		A[i] += B[i]; 

	}

}

