#ifndef _DLIB_RECOGNIZER_H_

#define _DLIB_RECOGNIZER_H_

#include <stdio.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

#include "IDetector.h"
#include "IRecognizer.h"


#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include "dlib/image_io.h"
#include "dlib/opencv.h"


#include <util.h>

using namespace dlib;
using namespace std;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, dlib::relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = dlib::relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
	alevel0<
	alevel1<
	alevel2<
	alevel3<
	alevel4<
	max_pool<3, 3, 2, 2, dlib::relu<affine<con<32, 7, 7, 2, 2,
	input_rgb_image_sized<150>
	>>>>>>>>>>>>;


class dlibRecognizer :public RecognizerInterface {
public:
	void initDB(std::string path);

	void initModel(std::string fn_model);

	void init(DetectorInterface *detector, std::string fn_DB, std::string fn_model);

	void train();

	int predict();

	void showResult();

	// 预测用的数据库
	std::vector<Mat> images;
	std::vector<unsigned long> labels;					// 
	std::map<int, string> labelsInfo;
	string test_data = "./data/at.txt";
	//string test_data = "C:/Users/Administrator/Downloads/att_faces/orl_faces/at.txt"
	std::vector<matrix<rgb_pixel>> faces;				// 输入脸数据
	std::vector<matrix<float, 0, 1>> face_descriptors;  // 128-d 描述子

	// 


};
#endif
