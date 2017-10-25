#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <fstream>
using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
	Mat obj = imread("C:/Users/Administrator/Desktop/5_1.jpg");   //载入目标图像
	Mat scene = imread("C:/Users/Administrator/Desktop/5.jpg"); //载入场景图像
	ofstream  pointfile;
	pointfile.open("C:/Users/Administrator/Desktop/point.txt");
	if (obj.empty() || scene.empty())
	{
		cout << "Can't open the picture!\n";
		return 0;
	}
	vector<KeyPoint> obj_keypoints, scene_keypoints;
	Mat obj_descriptors, scene_descriptors;
	ORB detector(10000);     //采用ORB算法提取特征点
	detector.detect(obj, obj_keypoints);
	detector.detect(scene, scene_keypoints);
	detector.compute(obj, obj_keypoints, obj_descriptors);
	detector.compute(scene, scene_keypoints, scene_descriptors);
	BFMatcher matcher(NORM_HAMMING, true); //汉明距离做为相似度度量
	vector<DMatch> matches;
	matcher.match(obj_descriptors, scene_descriptors, matches);
	Mat match_img;
	drawMatches(obj, obj_keypoints, scene, scene_keypoints, matches, match_img);
	imshow("滤除误匹配前", match_img);

	imwrite("match_img.jpg", match_img);
	//保存匹配对序号
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


	H12 = findHomography(Mat(points1),Mat(points2) , CV_RANSAC, ransacReprojThreshold);
	pointfile << "H12"<<H12 << endl;
	vector<char> matchesMask(matches.size(), 0);
	Mat points1t;
	perspectiveTransform(Mat(points1), points1t, H12);
	for (size_t i1 = 0; i1 < points1.size(); i1++)  //保存‘内点’
	{
		if (norm(points2[i1] - points1t.at<Point2f>((int)i1, 0)) <= ransacReprojThreshold) //给内点做标记
		{
			matchesMask[i1] = 1;
			pointfile << points1[i1] << "," << points2[i1] << ";" << endl;
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
	Mat img_tran;
	img_tran = Mat::zeros(obj.rows, obj.cols, obj.type());
	warpPerspective(obj, img_tran, H12, img_tran.size());
	imshow("img_tran", img_tran);
	
	//显示转换后图像
	/*Mat Template_rgb = imread("C:/Users/Administrator/Desktop/5.jpg");
	Mat Template, img_tran_gray;
	cvtColor(Template_rgb, Template, CV_BGR2GRAY);
	cvtColor(img_tran, img_tran_gray, CV_BGR2GRAY);
	Mat diff, BW_image;
	float num = (393 - 131 + 1)*(531 - 113 + 1);
	float sum = 0;
	for (int i = 131; i <= 393; i++)
	{
		for (int j = 113; j <= 531; j++)
		{
			float temp = Template.at<uchar>(i, j) - img_tran_gray.at<uchar>(i, j);
			sum += temp*temp;
		}
	}
	float mean = sum / num;
	printf("pixel:%f\n", mean);
	absdiff(Template, img_tran_gray, diff);
	imshow("ss", diff);
	imwrite("original.jpg", diff);
	float ssum = 0.;
	uchar max = 0, min = 0;
	for (int i = 0; i < diff.rows; i++)
	{
		for (int j = 0; j < diff.cols; j++)
		{
			if (diff.at<uchar>(i, j)<250)
			{
				ssum += diff.at<uchar>(i, j);
				if (diff.at<uchar>(i, j)>max)
					max = diff.at<uchar>(i, j);
				if (diff.at<uchar>(i, j) < min)
					min = diff.at<uchar>(i, j);
			}
			
		}
	}
	printf("pixel:%f\n", ssum / (diff.rows*diff.cols));
	printf("max:%d\n", max);
	printf("min:%d\n", min);
	printf("white:%d\n", diff.at<uchar>(3, 3));
	printf("black:%d\n", img_tran.at<uchar>(0, 0));*/
	pointfile.close();
	waitKey(0);

	return 0;
}