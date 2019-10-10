#include "util.h"

TimeWatcher::TimeWatcher(){
	QueryPerformanceFrequency(&Frequency);
}

TimeWatcher::~TimeWatcher(){

}

void TimeWatcher::startWatch() {
	QueryPerformanceCounter(&start);
}

float TimeWatcher::stopWatch() {
	QueryPerformanceCounter(&end);
	RunTime = (end.QuadPart - start.QuadPart) * 1000.0f / Frequency.QuadPart;
	std::cout << "RunTime: " << RunTime<< std::endl;
	return RunTime;
}

//定义静态成员变量
float TimeWatcher::RunTime = 0.0;
LARGE_INTEGER TimeWatcher::start;
LARGE_INTEGER TimeWatcher::end;
LARGE_INTEGER TimeWatcher::Frequency;