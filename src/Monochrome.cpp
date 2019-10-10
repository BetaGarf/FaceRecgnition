#include "Monochrome.h"
#include "facedetect-dll.h"
#include <opencv2/opencv.hpp>
#include <string>

//define the buffer size. Do not change the size!
#define DETECT_BUFFER_SIZE 0x20000

Monochrome::Monochrome() {

}

Monochrome::~Monochrome() {
	free(pBuffer);
}

void Monochrome::Init() {
	pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
	if (!pBuffer) {
		std::cerr << "Can not alloc buffer" << std::endl;
	}
}

bool Monochrome::SetPath(const wchar_t* path) {
	if (path == NULL) {
		return false;
	}
	models_path = *path;
	return true;
}

void Monochrome::Configure(float scale, int quality, int sizemin, int sizemax) {
	Monochrome::sizemin = sizemin;
	Monochrome::sizemax = sizemax;
}

bool Monochrome::SetImage(unsigned char* pImgData, int w, int h, int bpp) {
	if (pImgData == NULL) {
		return false;
	}
	Monochrome::pImgData = pImgData;
	Monochrome::weight = w;
	Monochrome::height = h;
	Monochrome::bpp = bpp;
	return true;
}
int Monochrome::ProcessFace() {
	int step = weight * (bpp / 8);//单通道
	pResults = facedetect_frontal_surveillance(pBuffer, pImgData, weight, height, step,
		scale, quality, sizemin, sizemax, 1);
	return *pResults;
}
std::vector<cv::Rect> Monochrome::GetFace() {
	std::vector<cv::Rect> faces;
	for (int index = 0; index < (pResults ? *pResults : 0); index++) {
		short * p = ((short*)(pResults + 1)) + 142 * index;
		int left = p[0];
		int bottom = p[1];
		int width = p[2];
		int height = p[3];
		faces.push_back(cv::Rect(left, bottom, width, height));
	}	
	return faces;
}

bool Monochrome::GetFace(int index, int& left, int& bottom, int& width, int& height) {
	if (pResults == NULL) {
		return false;
	}
	short * p = ((short*)(pResults + 1)) + 142 * index;
	left = p[0];
	bottom = p[1];
	width = p[2];
	height = p[3];
	return true;
}

void Monochrome::ShowImage(cv::Mat& image) {
	for (int i = 0; i < (pResults ? *pResults : 0); i++)
	{
		short * p = ((short*)(pResults + 1)) + 142 * i;
		int x = p[0];
		int y = p[1];
		int w = p[2];
		int h = p[3];
		cv::rectangle(image, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);
	}
	cv::imshow("FaceRegister", image);
}

void Monochrome::ShowImage(cv::Mat& image, float time, std::string& name) {
	std::string Ref = "No face can be detected:  ProcessTime: " + std::to_string(time) + "ms";
	for (int i = 0; i < (pResults ? *pResults : 0); i++)
	{
		short * p = ((short*)(pResults + 1)) + 142 * i;
		int x = p[0];
		int y = p[1];
		int w = p[2];
		int h = p[3];
		cv::rectangle(image, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);
		cv::putText(image, name, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
		Ref = std::to_string(weight) + "X" + std::to_string(height) + " ProcessTime: " + std::to_string(time) + "ms";
	}
	cv::putText(image, Ref, cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
	cv::imshow("FaceRecognition", image);
}