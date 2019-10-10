#ifndef _MONOCHROME_H
#define _MONOCHROME_H

#include <opencv2/opencv.hpp>
#include <string>

//define inetrface

class Monochrome{
public:	

	Monochrome() {
	}

	~Monochrome() {
	}

	void Init();

	bool SetPath(const wchar_t*);

	void Configure(float scale, int quality, int sizemin, int sizemax);

	bool SetImage(unsigned char* pImgData, int w, int h, int bpp);

	int ProcessFace();

	void ShowImage(cv::Mat& image);

	bool GetFace(int index, int& left, int& bottom, int& width, int& height);
	std::vector<cv::Rect> GetFace();

private:
	wchar_t models_path;
	unsigned char* pImgData;
	int * pResults;
	unsigned char * pBuffer;
	float scale = 1.2f;
	int quality = 2;
	int weight;
	int height;
	int bpp;
	int sizemin;
	int sizemax;
};

#endif // _MONOCHROME_H