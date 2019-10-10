#include "FaceDB.h"

bool FaceDB::SetFilePath(std::string image_path, std::string face_desc_path, std::string face_label_path) {
	if (image_path == "" || face_desc_path == "" || face_label_path == "") {
		return false;
	}
	FaceDB::image_path = image_path;
	FaceDB::face_desc_path = face_desc_path;
	FaceDB::face_label_path = face_label_path;
	return true;
}

bool FaceDB::InitDB(Monochrome& monochrome, DlibRecognition& dlibrecognition) {
	std::ofstream fout_desc(face_desc_path, std::ios::out);
	std::ofstream fout_label(face_label_path, std::ios::out);
	assert(fout_desc);
	assert(fout_label);
	std::vector<cv::String> file_vec;
	cv::glob(image_path + "*.jpg", file_vec, false);
	if (file_vec.size() <= 0) {
		std::cout << "文件读取失败" << std::endl;
		return false;
	}
	for (int i = 0; i < file_vec.size(); i++) {
		std::string name = file_vec[i];	
		int first = name.find_first_of("\\");
		first = first + 1;
		int last = name.find_last_of(".");
		name = name.substr(first, last - first);

		cv::Mat image = cv::imread(image_path + name + ".jpg");
		cv::Mat gray;
		cvtColor(image, gray, cv::COLOR_BGR2GRAY);
		//初始化
      	monochrome.Init();

		//传递一个单色图像
		monochrome.SetImage(gray.ptr(0), gray.cols, gray.rows, 8);

		//人脸检测
		int face_num = monochrome.ProcessFace();

		//返回所有人脸坐标和长宽
		std::vector<cv::Rect> faces = monochrome.GetFace();

		std::vector<dlib::matrix<float, 0, 1>> face_descriptors;
		//人脸识别
		if (faces.size() == 1) {//只检测一张人脸
			//传递图像和人脸
			if (dlibrecognition.SetImage(image, faces)) {
				//人脸识别
				dlibrecognition.RecognizeFace(face_descriptors);
			}
			fout_label << name << " ";
			fout_label.flush();
			fout_desc << dlib::trans(face_descriptors[0]);
		}
	}
	fout_label.close();
	fout_desc.close();
	return true;
}

bool FaceDB::FaceRegister(DlibRecognition& dlibrecognition, cv::Mat& image) {
	std::ofstream fout_desc(face_desc_path, std::ios::app);
	std::ofstream fout_label(face_label_path, std::ios::app);
	assert(fout_desc);
	assert(fout_label);

	std::cout << "请输入你的姓名：" << std::endl;
	std::string name = "";
	scanf("%s", &name);

	std::vector<dlib::matrix<float, 0, 1>> face_descriptors;
		
	//人脸识别
	dlibrecognition.RecognizeFace(face_descriptors);
	for (dlib::matrix<float, 0, 1> face_descriptor : face_descriptors) {
		fout_desc << dlib::trans(face_descriptor);;
	}

	fout_label << name.c_str() << " ";
	fout_label.close();
	fout_desc.close();
	cv::imwrite(image_path + name.c_str() + ".jpg", image);
	return true;
}

void FaceDB::FacedescToVector() {
	std::ifstream fin(face_desc_path);
	assert(fin);
	float items = 0.0;
	float num[128] = {0.128};
	while (! fin.eof()) {
		dlib::matrix<float, 0, 1> vec_desc(128, 1);
		for (int index = 0; index < 128; index++) {
			fin >> items;
			vec_desc(index, 0) = items;
		}
		face_descriptors.push_back(vec_desc);
	}
}

void FaceDB::FacelabelToVector() {
	std::ifstream fin(face_label_path);
	assert(fin);
	std::string str = "";
	while (!fin.eof()) {
		fin >> str;
		int first = str.find_first_of("\"");
		int last = str.find_last_of("\"");
		first = first + 1;
		last = last - first;
		str = str.substr(first, last);
		face_label.push_back(str);
	}
}