一、人脸识别系统简介
人脸识别技术作为计算机视觉和生物识别领域最为热门的研究主题之一，已经广泛应用到公共安全、人事考勤、金融交易、互联网等领域。基于深度学习框架实现了一套人脸识别系统，该系统由人脸检测、人脸对齐、特征提取和人脸匹配四个模块组成。测试结果表明，当输入分辨率为640X480的视频时，人脸识别速度达到50fps，准确率达到90%以上。


性能测试
人脸识别系统性能	速度	精度
	50FPS	90%
备注：640 X 480输入、5000张人脸的数据库、i58600、GTX 1080

二、环境配置
1、FaceRecognition压缩包包含了测试demo、源代码和说明文档。
 
2、Demo中的FaceRecognition.exe双击直接运行。该文件夹包含了必要的人脸数据和模型文件。
 
3、FaceRecognition中的FaceRecognition.sln双击可以编译源码。
【注意】如果源码编译遇到问题，可以参照下面的工程配置：
1、	头文件路径：..\include
2、	库文件路径：..\lib
3、依赖项：
opencv_world342.lib
libfacedetect-x64.lib
dlib19.17.0_release_64bit_msvc1916.lib
cublas.lib
cuda.lib
cudadevrt.lib
cudart.lib
cudart_static.lib
OpenCL.lib
cudnn.lib
curand.lib
cusolver.lib
4、	预处理器：DLIB_USE_CUDA
5、	从Demo文件夹中拷贝libfacedetect-x64.dll 、cudart64_100.dll、cublas64_100.dll、cudnn64_7.dll到工程目录下。
4、开源库介绍
人脸检测	于仕褀libfacedetection老版本
人脸识别	dlib19.7
Opencv	3.4.2
GPU	cuda-10.0、cudnn-10.0
备注：VS2017
