#include <iostream>

using namespace std; 

int main(int argc, char* argv[]) {

	int N = atoi(argv[1]); 

	int *a, *b; 
	int dot; 

	a = new int[N]; 
	b = new int[N]; 

	for (int i=0; i<N; i++) {

		a[i] = 1; 
		b[i] = 1; 

	}

	dot = 0; 

	for (int i=0; i<N; i++) {

		dot += a[i] * b[i]; 

	}

	delete a, b; 

}