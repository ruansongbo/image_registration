// GridMatch.cpp : Defines the entry point for the console application.

//#define USE_GPU 

#include "Header.h"
#include "gms_matcher.h"
#include <Eigen/Dense> 
#include <time.h> 
//#define USE_GPU
#define Resize
using namespace Eigen;
using namespace cv::cuda;

void GmsMatch(Mat &img1, Mat &img2);

void runImagePair(){
	//Mat img1 = imread("C:/Users/Administrator/Desktop/Template.png");
	//Mat img2 = imread("C:/Users/Administrator/Desktop/result_0.png");
	Mat img1 = imread("../Data/3.jpg");
	Mat img2 = imread("../Data/4.jpg");
#ifdef Resize
	imresize(img1, 640);
	imresize(img2, 640);
#endif // Resize

	

	GmsMatch(img1, img2);
}


int main()
{
	clock_t start, finish;
	start = clock();
#ifdef USE_GPU
	int flag = cuda::getCudaEnabledDeviceCount();
	if (flag != 0){ cuda::setDevice(0); }
#endif // USE_GPU
	runImagePair();
	finish = clock();
	cout << finish - start << endl;
   	return 0;
}


void GmsMatch(Mat &img1, Mat &img2){
	vector<KeyPoint> kp1, kp2;
	Mat d1, d2;
	vector<DMatch> matches_all, matches_gms; 

	Ptr<ORB> orb = ORB::create(10000);
	orb->setFastThreshold(0);
	orb->detectAndCompute(img1, Mat(), kp1, d1);
	orb->detectAndCompute(img2, Mat(), kp2, d2);

#ifdef USE_GPU
	GpuMat gd1(d1), gd2(d2);
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(NORM_HAMMING);
	matcher->match(gd1, gd2, matches_all);
#else
	BFMatcher matcher(NORM_HAMMING);
	matcher.match(d1, d2, matches_all);
#endif

	// GMS filter
	int num_inliers = 0;
	std::vector<bool> vbInliers;
	gms_matcher gms(kp1,img1.size(), kp2,img2.size(), matches_all);
	num_inliers = gms.GetInlierMask(vbInliers, false, false);
	
	
	for (size_t i = 0; i < vbInliers.size(); ++i)
	{
		if (vbInliers[i] == true)
		{
			matches_gms.push_back(matches_all[i]);
		}
	}
	//RANSAC
	//std::vector<Point2f> obj;
	//std::vector<Point2f> scene;
	//for (size_t i = 0; i < num_inliers; i++)
	//{
	//	obj.push_back(kp1[matches_gms[i].queryIdx].pt);
	//	scene.push_back(kp2[matches_gms[i].trainIdx].pt);
	//}
	//int ransacReprojThreshold = 3;  //¾Ü¾øãÐÖµ
	//Mat H12 = findHomography(scene, obj, CV_RANSAC, ransacReprojThreshold);
	//Mat img_tran;
	//img_tran = Mat::zeros(img2.rows, img2.cols, img2.type());
	//warpPerspective(img2, img_tran, H12, img_tran.size());
	//imshow("img_tran", img_tran);							//ÏÔÊ¾×ª»»ºóÍ¼Ïñ

	//GMS
	MatrixXd U = MatrixXd::Ones(matches_gms.size(),3);
	MatrixXd U_1 = MatrixXd::Ones(matches_gms.size(), 3);
	for (int i = 0; i < matches_gms.size(); i++)
	{
		U(i, 0) = kp1[matches_gms[i].queryIdx].pt.x;
		U(i, 1) = kp1[matches_gms[i].queryIdx].pt.y;
		U_1(i, 0) = kp2[matches_gms[i].trainIdx].pt.x;
		U_1(i, 1) = kp2[matches_gms[i].trainIdx].pt.y;
	}

	MatrixXd HH = U_1.jacobiSvd(ComputeThinU | ComputeThinV).solve(U);
	Mat H12(3, 3, CV_32FC1);
	H12.at<float>(0, 0) = HH(0, 0);
	H12.at<float>(0, 1) = HH(0, 1);
	H12.at<float>(0, 2) = HH(0, 2);
	H12.at<float>(1, 0) = HH(1, 0);
	H12.at<float>(1, 1) = HH(1, 1);
	H12.at<float>(1, 2) = HH(1, 2);
	H12.at<float>(2, 0) = HH(2, 0);
	H12.at<float>(2, 1) = HH(2, 1);
	H12.at<float>(2, 2) = HH(2, 2);
	Mat img_tran;
	img_tran = Mat::zeros(img2.rows, img2.cols, img2.type());
	warpPerspective(img2, img_tran, H12.t(), img_tran.size());
	imshow("img_tran", img_tran);							//ÏÔÊ¾×ª»»ºóÍ¼Ïñ



	Mat Template_rgb = imread("../Data/3.jpg");
#ifdef Resize
	imresize(Template_rgb, 640);
#endif // Resize

	
	Mat Template, img_tran_gray;
	cvtColor(Template_rgb, Template, CV_BGR2GRAY);
	cvtColor(img_tran, img_tran_gray, CV_BGR2GRAY);
	Mat diff, BW_image;
	absdiff(Template, img_tran_gray, diff);
	imshow("ss", diff); 
	imwrite("C:/Users/Administrator/Desktop/GMS_0.jpg", diff);
	waitKey(0);
}

cv::Mat SVDfun(const vector<Point2f> &points1, const vector<Point2f> &points2)
{
	cv::Mat A_Mat(points1.size() * 2, 9, CV_32FC1);
	for (int i = 0; i < points1.size(); i++)
	{
		A_Mat.at<float>(i * 2, 0) = points1[i].x;
		A_Mat.at<float>(i * 2, 1) = points1[i].y;
		A_Mat.at<float>(i * 2, 2) = 1;
		A_Mat.at<float>(i * 2, 3) = 0;
		A_Mat.at<float>(i * 2, 4) = 0;
		A_Mat.at<float>(i * 2, 5) = 0;
		A_Mat.at<float>(i * 2, 6) = -points1[i].x * points2[i].x;
		A_Mat.at<float>(i * 2, 7) = -points1[i].y * points2[i].x;
		A_Mat.at<float>(i * 2, 8) = -points2[i].x;
		A_Mat.at<float>(i * 2 + 1, 0) = 0;
		A_Mat.at<float>(i * 2 + 1, 1) = 0;
		A_Mat.at<float>(i * 2 + 1, 2) = 0;
		A_Mat.at<float>(i * 2 + 1, 3) = points1[i].x;
		A_Mat.at<float>(i * 2 + 1, 4) = points1[i].y;
		A_Mat.at<float>(i * 2 + 1, 5) = 1;
		A_Mat.at<float>(i * 2 + 1, 6) = -points1[i].x * points2[i].y;
		A_Mat.at<float>(i * 2 + 1, 7) = -points1[i].y * points2[i].y;
		A_Mat.at<float>(i * 2 + 1, 8) = -points2[i].y;
	}
	cout << A_Mat << endl;
	cv::SVD svd(A_Mat, SVD::FULL_UV);
	cout << "end" << endl;
	Mat H12(3, 3, CV_32FC1);
	H12.at<float>(0, 0) = svd.vt.at<float>(8, 0) / svd.vt.at<float>(8, 8);
	H12.at<float>(0, 1) = svd.vt.at<float>(8, 1) / svd.vt.at<float>(8, 8);
	H12.at<float>(0, 2) = svd.vt.at<float>(8, 2) / svd.vt.at<float>(8, 8);
	H12.at<float>(1, 0) = svd.vt.at<float>(8, 3) / svd.vt.at<float>(8, 8);
	H12.at<float>(1, 1) = svd.vt.at<float>(8, 4) / svd.vt.at<float>(8, 8);
	H12.at<float>(1, 2) = svd.vt.at<float>(8, 5) / svd.vt.at<float>(8, 8);
	H12.at<float>(2, 0) = svd.vt.at<float>(8, 6) / svd.vt.at<float>(8, 8);
	H12.at<float>(2, 1) = svd.vt.at<float>(8, 7) / svd.vt.at<float>(8, 8);
	H12.at<float>(2, 2) = 1;
	return H12;
}
