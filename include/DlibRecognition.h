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
	
	void SetPath(const wchar_t* model_path);

	void Load_Models();

	bool SetImage(cv::Mat& image, std::vector<cv::Rect>& faces);

	int RecognizeFace();

private:
	wchar_t* landmarks_model_path;
	wchar_t* resnet_model_path;
	dlib::matrix<dlib::rgb_pixel> img;
	std::vector<dlib::rectangle> dets;
};
void load_model();
int face_recognizer(std::vector<dlib::rectangle>& dets, cv::Mat& image);
#endif // _DLIBRECOGNITION_H

