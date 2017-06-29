#include "Timer.hh"

#include <iostream>

Timer::Timer() {} 

void Timer::start() {

	begin = clock(); 

}

void Timer::stop() {

	end = clock(); 

	double elapsed = double(end - begin) / CLOCKS_PER_SEC; 

	cout << "Elapsed Time = " << elapsed << " seconds" << endl; 

}