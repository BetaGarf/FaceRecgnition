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

	//����ע��ʱ����
	void RecognizeFace(std::vector<dlib::matrix<float, 0, 1>>&);

	std::string RecognizeFace(const std::vector<dlib::matrix<float, 0, 1>>&, const std::vector<std::string>&);

	//�����ȶ���ֵ����Χ��0~1֮�䣬���Ը�����Ҫ�ʵ�����threshold�Ĵ�С��thresholdԽ����ʶ����Խ�ߡ�
	//ͨ������ʵ��ͳ����ֵ����0.5ʱʶ��Ч����á�
	float threshold = 0.5;

private:
	dlib::matrix<dlib::rgb_pixel> img;

	std::vector<dlib::rectangle> dets;
};
void load_model();
#endif // _DLIBRECOGNITION_H

