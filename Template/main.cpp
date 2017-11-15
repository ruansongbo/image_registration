#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

int main()
{
	cv::Mat image = imread("../data/2.jpg", cv::IMREAD_COLOR);
	cv::Mat templateImage = imread("../data/1.jpg", cv::IMREAD_COLOR);

	int result_cols = image.cols - templateImage.cols + 1;
	int result_rows = image.rows - templateImage.rows + 1;

	cv::Mat result = cv::Mat(result_cols, result_rows, CV_32FC1);
	cv::matchTemplate(image, templateImage, result, CV_TM_SQDIFF);

	double minVal, maxVal;
	cv::Point minLoc, maxLoc, matchLoc;
	cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	matchLoc = minLoc;

	cv::rectangle(image, cv::Rect(matchLoc, cv::Size(templateImage.cols, templateImage.rows)), Scalar(0, 0, 255), 2, 8, 0);

	imshow("", image);

	return 0;
}