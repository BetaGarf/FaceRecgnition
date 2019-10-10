#ifndef _FACEDETECTION_H
#define _FACEDETECTION_H
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#define DETECT_BUFFER_SIZE 0x20000
class FaceDetection {
public:
	FaceDetection();
	void face_detected();
	void show_face();
	std::vector<cv::Rect> get_RectFace();

private:
	std::vector<cv::Rect> faces;
};
#endif // _FACEDETECTION_H

