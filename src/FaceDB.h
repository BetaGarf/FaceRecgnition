#ifndef _FACEDB_H
#define _FACEDB_H

#include <fstream>
#include <assert.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <dlib/matrix/matrix.h>
#include "Monochrome.h"
#include "DlibRecognition.h"

class FaceDB {
public:
	FaceDB() {};

	~FaceDB() {};

	bool SetFilePath(std::string, std::string, std::string);

	//初始化，从本地人脸库提取描述子并保存到文本文件；该函数只需在第一次运行时执行初始化，后续可以注释调用
	bool InitDB(Monochrome&, DlibRecognition&);

	//人脸注册
	bool FaceRegister(DlibRecognition&, cv::Mat&);

	//从文本文件导入人脸描述子
	void FacedescToVector();

	//从文本文件导入人脸标签
	void FacelabelToVector();

	std::vector<dlib::matrix<float, 0, 1>> face_descriptors;

	std::vector<std::string> face_label; 

private:
	//人脸数据库路径
	std::string image_path;

	//人脸描述子路径
	std::string face_desc_path;

	//人脸标签路径
	std::string face_label_path;
};

#endif // _FACEDB_H
