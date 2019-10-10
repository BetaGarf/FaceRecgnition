#ifndef _IDETECTOR_H_
#define _IDETECTOR_H_

#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect.hpp>

// 接口类
class DetectorInterface {
public:
	cv::Mat image;
	cv::Mat gray;
	// 识别出来的矩形框
	std::vector<cv::Rect> Faces;

	// 是否 Face Landmark 用于人脸对齐
	int doLandmark = 0;
	// 68_face_landmarks
	std::vector<cv::Point> Points;
	
	std::string WindowName;

	DetectorInterface() {
	}

	~DetectorInterface() {
	}

	virtual void init() = 0;

	virtual void setTarget(cv::Mat &img, const std::string& = "Face Detection example") = 0;

	virtual void runDetection() = 0;

	virtual void showDetection(int index) = 0;

	virtual void stopDetection() = 0;


};


#endif