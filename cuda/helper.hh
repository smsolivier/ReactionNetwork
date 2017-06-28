#ifndef helper_hh
#define helper_hh

#include <vector>
#include <string> 

using namespace std; 

vector<double> linspace(double start, double end, int N); // generate evenly spaced points 

void printVector(vector<double> x); // print 1D vector 

void printVector(vector<vector<double>> x); // print 2D vector 

void writeTable(string name, vector<double> &x, vector<double> &y); 

vector<double> matVectorProduct(vector<vector<double>> &A, vector<double> &x); 

#endif 