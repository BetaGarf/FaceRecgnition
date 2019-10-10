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

	//��ʼ�����ӱ�����������ȡ�����Ӳ����浽�ı��ļ����ú���ֻ���ڵ�һ������ʱִ�г�ʼ������������ע�͵���
	bool InitDB(Monochrome&, DlibRecognition&);

	//����ע��
	bool FaceRegister(DlibRecognition&, cv::Mat&);

	//���ı��ļ���������������
	void FacedescToVector();

	//���ı��ļ�����������ǩ
	void FacelabelToVector();

	std::vector<dlib::matrix<float, 0, 1>> face_descriptors;

	std::vector<std::string> face_label; 

private:
	//�������ݿ�·��
	std::string image_path;

	//����������·��
	std::string face_desc_path;

	//������ǩ·��
	std::string face_label_path;
};

#endif // _FACEDB_H
