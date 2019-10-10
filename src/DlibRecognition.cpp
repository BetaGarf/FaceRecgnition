#include "DlibRecognition.h"

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

// The next bit of code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the introductory
// dlib DNN examples to learn what all this stuff means.
//
// Also, the dnn_metric_learning_on_images_ex.cpp example shows how to train this network.
// The dlib_face_recognition_resnet_model_v1 model used by this example was trained using
// essentially the code shown in dnn_metric_learning_on_images_ex.cpp except the
// mini-batches were made larger (35x15 instead of 5x5), the iterations without progress
// was set to 10000, and the training dataset consisted of about 3 million images instead of
// 55.  Also, the input layer was locked to images of size 150.
template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

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
	max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
	input_rgb_image_sized<150>
	>>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

shape_predictor sp;
anet_type net;

//模型加载
void load_model() {
	deserialize("../models/shape_predictor_68_face_landmarks.dat") >> sp;
	deserialize("../models/dlib_face_recognition_resnet_model_v1.dat") >> net;
}

void DlibRecognition::SetThreshold(float threshold) {
	DlibRecognition::threshold = threshold;
}

bool DlibRecognition::SetImage(cv::Mat& image, std::vector<cv::Rect>& faces) {
	dlib::assign_image(img, cv_image<rgb_pixel>(image));
	if (faces.size() <= 0) {
		return false;
	}
	dets.clear();
	for (int i = 0; i < faces.size(); i++) {
		dlib::rectangle dlibRect((long)faces[i].tl().x, (long)faces[i].tl().y, (long)faces[i].br().x - 1, (long)faces[0].br().y - 1);
		dets.push_back(dlibRect);
	}
	return true;
}

void DlibRecognition::RecognizeFace(std::vector<dlib::matrix<float, 0, 1>>& face_desc) {
	try
	{
		std::vector<dlib::matrix<rgb_pixel>> faces;
		for (auto face : dets)
		{
			auto shape = sp(img, face);
			dlib::matrix<rgb_pixel> face_chip;
			extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
			faces.push_back(move(face_chip));
			// Also put some boxes on the faces so we can see that the detector is finding
			// them.
		}
		if (faces.size() == 0){
			cout << "No faces found in image!" << endl;
		}
		else {
			// DNN to generate 128D vector.
			std::vector<dlib::matrix<float, 0, 1>> face_descriptors;
			face_descriptors = net(faces);
			for (dlib::matrix<float, 0, 1> face_descriptor : face_descriptors) {
				face_desc.push_back(face_descriptor);
			}
		}
	}
	catch (std::exception& e)
	{
		cout << e.what() << endl;
	}
}

std::string DlibRecognition::RecognizeFace(const std::vector<dlib::matrix<float, 0, 1>>& face_desc, const std::vector<std::string>& face_label) {
	try
	{
		std::vector<dlib::matrix<rgb_pixel>> faces;
		for (auto face : dets){
			auto shape = sp(img, face);
			dlib::matrix<rgb_pixel> face_chip;
			extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
			faces.push_back(move(face_chip));
			// Also put some boxes on the faces so we can see that the detector is finding
			// them.
		}
		if (faces.size() == 0){
			cout << "No faces found in image!" << endl;
			return " ";
		}

		// DNN to generate 128D vector.
		std::vector<dlib::matrix<float, 0, 1>> face_descriptors;
		face_descriptors = net(faces);
		float min_distance = 1024;
		int label_index = 0;
		for (int index = 0; index < face_desc.size();index++ ) {
			float temp_distance = length((face_desc[index] - face_descriptors[0]));
			if (temp_distance < min_distance) {
				min_distance = temp_distance;
				label_index = index;
			}
		}
		if (min_distance < DlibRecognition::threshold) {
			std::cout << "已识别" << std::endl;
			return face_label[label_index];
		}
		else{
			std::cout << "未识别" << std::endl;
			return "others";
		}
	}
	catch (std::exception& e)
	{
		cout << e.what() << endl;
	}
	return "others";
}
// ----------------------------------------------------------------------------------------