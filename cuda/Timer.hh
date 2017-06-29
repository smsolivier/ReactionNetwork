#ifndef __TIMER_HH__
#define __TIMER_HH__

#include <ctime> 

using namespace std; 

class Timer {

public:

	Timer(); 

	void start(); 
	void stop(); 

private:

	clock_t begin; 
	clock_t end; 

}; 

#endif 