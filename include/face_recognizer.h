#pragma once
#include <vector>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/image_loader/load_image.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>

int face_recognizer(std::vector<dlib::rectangle>& dets, cv::Mat& img);
void load_model();
