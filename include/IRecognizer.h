#ifndef _IRECOGNIZER_H_

#define _IRECOGNIZER_H_

#include <string>
#include "IDetector.h"

/*
	����ʶ��ӿ���
*/
class RecognizerInterface {
public:
	
	virtual void init(DetectorInterface *detector, std::string fn_DB, std::string fn_model) = 0;

	virtual void train() = 0;

	virtual int predict() = 0;

	virtual void showResult() = 0;
	
	
	DetectorInterface *detector;
};

#endif
