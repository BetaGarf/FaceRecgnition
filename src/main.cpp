#include <iostream>
#include <opencv2/opencv.hpp>

#include "Monochrome.h"
#include "DlibRecognition.h"
#include "FaceDB.h"
#include "util.h"


int main(int argc, char* argv[])
{
	//opencv捕获摄像头
	cv::VideoCapture capture(cv::CAP_DSHOW);
	if (!capture.isOpened()) {
		std::cerr << "Can not open video from camera!" << std::endl;
		return -1;
	}

	//设置摄像头分辨率、帧率
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
	capture.set(CV_CAP_PROP_FPS, 30);

	//计时接口
	TimeWatcher* timewatcher = new TimeWatcher();

	//单色接口，并设置设置人脸大小范围
	Monochrome* monochrome = new Monochrome();
	monochrome->Configure(1.2f, 2, 30, 240);

	//识别接口，设置人脸比对阈值，加载模型文件
	DlibRecognition *dlibrecognition = new DlibRecognition();	
	dlibrecognition->SetThreshold(0.5);
	load_model();

	//人脸数据库接口，设置文件路径并导入内存
	FaceDB* facedb = new FaceDB();
	std::string image_path = "../facedata/data/";
	std::string face_desc_path = "../facedata/face_data.txt";
	std::string face_label_path = "../facedata/face_lable.txt"; \
	std::cout << image_path << "  " << face_desc_path << "  " << face_label_path << std::endl;
	facedb->SetFilePath(image_path, face_desc_path, face_label_path);
	//首次初始化人脸库，需要调用，以后可以注释掉
	//facedb->InitDB(*monochrome, *dlibrecognition);
	facedb->FacedescToVector();
	facedb->FacelabelToVector();

	//初始化人名
	std::string name = "";
	int KeyValue = 0;
	try {
		do {
			cv::Mat image;
			capture.read(image);
			if (image.rows <= 0 || image.cols <= 0) {
				std::cerr << "Camera did not grap  the image!" << std::endl;
				continue;
			}
			std::cout << "Resolution is: " << image.cols << "x"
				<< image.rows << std::endl;
			cv::Mat gray;
			cvtColor(image, gray, cv::COLOR_BGR2GRAY);

			//开始计时
			timewatcher->startWatch();

			//人脸检测
			monochrome->Init();
			monochrome->SetImage(gray.ptr(0), gray.cols, gray.rows, 8);
			int face_num = monochrome->ProcessFace();
			std::vector<cv::Rect> faces = monochrome->GetFace();

			//人脸识别
			if(dlibrecognition->SetImage(image, faces)) {
				if (KeyValue == int('r')) {//按键'r'进入人脸注册
					if (faces.size() != 1) {						
						std::cout << "请看摄像头并调整姿势，确保镜头只能有一张人脸" << std::endl;
					}
					else {
						facedb->FaceRegister(*dlibrecognition, image);
					}
				}
				else {//人脸识别
					name = dlibrecognition->RecognizeFace(facedb->face_descriptors, facedb->face_label);
				}
			}
			
			//结束计时并显示图像
			float total_time = timewatcher->stopWatch();
			monochrome->ShowImage(image, total_time, name);

			KeyValue = cvWaitKey(10);
			if (KeyValue == int('q')) {//按键'q'可以退出程序
				break;
			}

		} while (1);
	}

	catch (std::exception& e)
	{
		std::cout << "\nexception thrown!" << std::endl;
		std::cout << e.what() << std::endl;
	}
	return 0;
}




