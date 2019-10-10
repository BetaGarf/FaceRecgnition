#include <iostream>
#include <opencv2/opencv.hpp>

#include "Monochrome.h"
#include "DlibRecognition.h"
#include "FaceDB.h"
#include "util.h"


int main(int argc, char* argv[])
{
	//opencv��������ͷ
	cv::VideoCapture capture(cv::CAP_DSHOW);
	if (!capture.isOpened()) {
		std::cerr << "Can not open video from camera!" << std::endl;
		return -1;
	}

	//��������ͷ�ֱ��ʡ�֡��
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
	capture.set(CV_CAP_PROP_FPS, 30);

	//��ʱ�ӿ�
	TimeWatcher* timewatcher = new TimeWatcher();

	//��ɫ�ӿڣ�����������������С��Χ
	Monochrome* monochrome = new Monochrome();
	monochrome->Configure(1.2f, 2, 30, 240);

	//ʶ��ӿڣ����������ȶ���ֵ������ģ���ļ�
	DlibRecognition *dlibrecognition = new DlibRecognition();	
	dlibrecognition->SetThreshold(0.5);
	load_model();

	//�������ݿ�ӿڣ������ļ�·���������ڴ�
	FaceDB* facedb = new FaceDB();
	std::string image_path = "../facedata/data/";
	std::string face_desc_path = "../facedata/face_data.txt";
	std::string face_label_path = "../facedata/face_lable.txt"; \
	std::cout << image_path << "  " << face_desc_path << "  " << face_label_path << std::endl;
	facedb->SetFilePath(image_path, face_desc_path, face_label_path);
	//�״γ�ʼ�������⣬��Ҫ���ã��Ժ����ע�͵�
	//facedb->InitDB(*monochrome, *dlibrecognition);
	facedb->FacedescToVector();
	facedb->FacelabelToVector();

	//��ʼ������
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

			//��ʼ��ʱ
			timewatcher->startWatch();

			//�������
			monochrome->Init();
			monochrome->SetImage(gray.ptr(0), gray.cols, gray.rows, 8);
			int face_num = monochrome->ProcessFace();
			std::vector<cv::Rect> faces = monochrome->GetFace();

			//����ʶ��
			if(dlibrecognition->SetImage(image, faces)) {
				if (KeyValue == int('r')) {//����'r'��������ע��
					if (faces.size() != 1) {						
						std::cout << "�뿴����ͷ���������ƣ�ȷ����ͷֻ����һ������" << std::endl;
					}
					else {
						facedb->FaceRegister(*dlibrecognition, image);
					}
				}
				else {//����ʶ��
					name = dlibrecognition->RecognizeFace(facedb->face_descriptors, facedb->face_label);
				}
			}
			
			//������ʱ����ʾͼ��
			float total_time = timewatcher->stopWatch();
			monochrome->ShowImage(image, total_time, name);

			KeyValue = cvWaitKey(10);
			if (KeyValue == int('q')) {//����'q'�����˳�����
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




