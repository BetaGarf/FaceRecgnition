#ifndef _DLIBRECOGNITION_H
#define _DLIBRECOGNITION_H

#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <sstream>
#include <io.h>
#include <opencv2/opencv.hpp>


class DlibRecognition {
public:
	DlibRecognition(){}

	~DlibRecognition(){}

	void SetThreshold(float threshold);
	
	bool SetImage(cv::Mat& image, std::vector<cv::Rect>& faces);

	//人脸注册时调用
	void RecognizeFace(std::vector<dlib::matrix<float, 0, 1>>&);

	std::string RecognizeFace(const std::vector<dlib::matrix<float, 0, 1>>&, const std::vector<std::string>&);

	//人脸比对阈值，范围在0~1之间，可以根据需要适当调整threshold的大小，threshold越大，误识别率越高。
	//通过大量实验统计阈值等于0.5时识别效果最好。
	float threshold = 0.5;

private:
	dlib::matrix<dlib::rgb_pixel> img;

	std::vector<dlib::rectangle> dets;
};
void load_model();
#endif // _DLIBRECOGNITION_H

