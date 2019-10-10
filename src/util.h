#ifndef _UTIL_H_
#define _UTIL_H_
#include <windows.h>
#include <iostream>


class TimeWatcher {
public:
	TimeWatcher();

	~TimeWatcher();

	static void startWatch();

	static float stopWatch();

	static float RunTime ;

	static LARGE_INTEGER start, end, Frequency;
};
#endif
