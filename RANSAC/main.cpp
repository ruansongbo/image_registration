#include <iostream>
#include "opencv2/opencv.hpp"
#include <windows.h>
#define DEBUG
#define Resize
using namespace cv;
using namespace std;
inline void imresize(Mat &src, int height) {
	double ratio = src.rows * 1.0 / height;
	int width = static_cast<int>(src.cols * 1.0 / ratio);
	resize(src, src, Size(width, height));
}
int main(int argc, char** argv)
{
	int 
	long start_time = GetTickCount();
	//Mat obj = imread("../Data/4.jpg");//载入目标图像
	//Mat scene = imread("../Data/3.jpg");//载入场景图像
	Mat obj = imread("C:/Users/Administrator/Desktop/MindVision/1.BMP");//载入目标图像
	Mat scene = imread("C:/Users/Administrator/Desktop/MindVision/16.BMP");//载入场景图像
#ifdef Resize
	imresize(obj, 640);
	imresize(scene, 640);
#endif // Resize

	
	if (obj.empty() || scene.empty())
	{
		cout << "Can't open the picture!\n";
		return 0;
	}
	vector<KeyPoint> obj_keypoints, scene_keypoints;
	Mat obj_descriptors, scene_descriptors;
	Ptr<ORB> detector = ORB::create(10000);
	detector->setFastThreshold(0);
	Mat mask;
	Rect r1(133, 165, 580, 380);
	mask = Mat::zeros(scene.size(), CV_8UC1);
	mask(r1).setTo(255);
	detector->detectAndCompute(obj, mask, obj_keypoints, obj_descriptors);
	detector->detectAndCompute(scene, Mat(), scene_keypoints, scene_descriptors);
	BFMatcher matcher(NORM_HAMMING, true); //汉明距离做为相似度度量
	vector<DMatch> matches;
	matcher.match(obj_descriptors, scene_descriptors, matches);
#ifdef DEBUG
	Mat match_img;
	drawMatches(obj, obj_keypoints, scene, scene_keypoints, matches, match_img);
	imshow("滤除误匹配前", match_img);
	imwrite("match_img.jpg", match_img);
#endif // DEBUG
	//保存匹配对序号zhouj
	vector<int> queryIdxs(matches.size()), trainIdxs(matches.size());
	for (size_t i = 0; i < matches.size(); i++)
	{
		queryIdxs[i] = matches[i].queryIdx;
		trainIdxs[i] = matches[i].trainIdx;
	}
	Mat H12;   //变换矩阵
	vector<Point2f> points1; KeyPoint::convert(obj_keypoints, points1, queryIdxs);
	vector<Point2f> points2; KeyPoint::convert(scene_keypoints, points2, trainIdxs);
	int ransacReprojThreshold = 3;  //拒绝阈值
	H12 = findHomography(Mat(points2),Mat(points1) , CV_RANSAC, ransacReprojThreshold);
	cout << GetTickCount() - start_time << endl;

#ifdef DEBUG

	vector<char> matchesMask(matches.size(), 0);
	Mat points1t;
	perspectiveTransform(Mat(points1), points1t, H12);
	for (size_t i1 = 0; i1 < points1.size(); i1++)  //保存‘内点’
	{
		if (norm(points2[i1] - points1t.at<Point2f>((int)i1, 0)) <= ransacReprojThreshold) //给内点做标记
		{
			matchesMask[i1] = 1;
		}
	}
	Mat match_img2;   //滤除‘外点’后
	drawMatches(obj, obj_keypoints, scene, scene_keypoints, matches, match_img2, Scalar(0, 0, 255), Scalar::all(-1), matchesMask);

	//画出目标位置
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(obj.cols, 0);
	obj_corners[2] = cvPoint(obj.cols, obj.rows); obj_corners[3] = cvPoint(0, obj.rows);
	std::vector<Point2f> scene_corners(4);
	perspectiveTransform(obj_corners, scene_corners, H12);
	line(match_img2, scene_corners[0] + Point2f(static_cast<float>(obj.cols), 0),
		scene_corners[1] + Point2f(static_cast<float>(obj.cols), 0), Scalar(0, 0, 255), 2);
	line(match_img2, scene_corners[1] + Point2f(static_cast<float>(obj.cols), 0),
		scene_corners[2] + Point2f(static_cast<float>(obj.cols), 0), Scalar(0, 0, 255), 2);
	line(match_img2, scene_corners[2] + Point2f(static_cast<float>(obj.cols), 0),
		scene_corners[3] + Point2f(static_cast<float>(obj.cols), 0), Scalar(0, 0, 255), 2);
	line(match_img2, scene_corners[3] + Point2f(static_cast<float>(obj.cols), 0),
		scene_corners[0] + Point2f(static_cast<float>(obj.cols), 0), Scalar(0, 0, 255), 2);

	imshow("滤除误匹配后", match_img2);
	imwrite("match_img2.jpg", match_img2);
#endif // DEBUG	
	Mat img_tran;
	img_tran = Mat::zeros(obj.rows, obj.cols, obj.type());
	warpPerspective(scene, img_tran, H12, img_tran.size());
	imshow("img_tran", img_tran);

	//显示转换后图像
	Mat Template_rgb = imread("C:/Users/Administrator/Desktop/MindVision/1.BMP");
	//Mat Template_rgb = imread("../Data/3.jpg");
#ifdef Resize
	imresize(Template_rgb, 640);
#endif // Resize

	
	Mat Template, img_tran_gray;
	cvtColor(Template_rgb, Template, CV_BGR2GRAY);
	cvtColor(img_tran, img_tran_gray, CV_BGR2GRAY);
	Mat diff, BW_image;
	absdiff(Template, img_tran_gray, diff);
	//imshow("ss", diff);
	imwrite("C:/Users/Administrator/Desktop/diff_160.jpg", diff);




	waitKey(0);

	return 0;
}