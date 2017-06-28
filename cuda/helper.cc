#include "helper.hh"

#include <iostream>
#include <fstream>

vector<double> linspace(double start, double end, int N) {
	/* generate N evenly spaced points between [start, end] */ 

	vector<double> x(N); 

	double h = (end - start)/(N-1); 

	for (int i=0; i<N; i++) {

		x[i] = h*i + start; 

	}

	return x; 

}

void printVector(vector<double> x) {

    for (int i=0; i<x.size(); i++) {

        cout << x[i] << " "; 
    }

    cout << endl; 
    cout << endl; 

}

void printVector(vector<vector<double>> x) {

    for (int i=0; i<x.size(); i++) {

        for (int j=0; j<x[i].size(); j++) {

            cout << x[i][j] << " "; 
        }

        cout << endl; 
    }

}

void writeTable(string name, vector<double> &x, vector<double> &y) {

    ofstream file; 
    file.open(name); 

    int N = x.size(); 

    for (int i=0; i<N; i++) {

        file << x[i] << " " << y[i] << endl; 

    }

    file.close(); 
    
}

vector<double> matVectorProduct(vector<vector<double>> &A, vector<double> &x) {

    int N = x.size(); // dimensions 

    vector<double> sol(N, 0); 

    for (int i=0; i<N; i++) {

        for (int j=0; j<N; j++) {

            sol[i] += A[i][j] * x[j]; 

        }

    }

    return sol; 

}