#pragma once
#include "Header.h"

#define THRESH_FACTOR 6

// 8 possible rotation and each one is 3 X 3 
const int mRotationPatterns[8][9] = {
	1,2,3,
	4,5,6,
	7,8,9,

	4,1,2,
	7,5,3,
	8,9,6,

	7,4,1,
	8,5,2,
	9,6,3,

	8,7,4,
	9,5,1,
	6,3,2,

	9,8,7,
	6,5,4,
	3,2,1,

	6,9,8,
	3,5,7,
	2,1,4,

	3,6,9,
	2,5,8,
	1,4,7,

	2,3,6,
	1,5,9,
	4,7,8
};

// 5 level scales
const double mScaleRatios[5] = { 1.0, 1.0 / 2, 1.0 / sqrt(2.0), sqrt(2.0), 2.0 };


class gms_matcher
{
public:
	// Normalized Points
	vector<Point2f> mvP1, mvP2;

	// Matches
	vector<pair<int, int> > mvMatches;
	// OpenCV Keypoints & Correspond Image Size & Nearest Neighbor Matches 
	gms_matcher(const vector<KeyPoint> &vkp1, const Size size1, const vector<KeyPoint> &vkp2, const Size size2, const vector<DMatch> &vDMatches) 
	{
		// Input initialize
		NormalizePoints(vkp1, size1, mvP1);
		NormalizePoints(vkp2, size2, mvP2);
		mNumberMatches = vDMatches.size();//匹配点对数目
		ConvertMatches(vDMatches, mvMatches);

		// Grid initialize
		mGridSizeLeft = Size(20, 20);//网格尺寸
		mGridNumberLeft = mGridSizeLeft.width * mGridSizeLeft.height;//网格数量

		// Initialize the neihbor of left grid 
		mGridNeighborLeft = Mat::zeros(mGridNumberLeft, 9, CV_32SC1);//网格邻域的网格索引矩阵
		InitalizeNiehbors(mGridNeighborLeft, mGridSizeLeft);////返回所有网格邻域的网格索引矩阵
	};
	~gms_matcher() {};


private:



	// Number of Matches
	size_t mNumberMatches;

	// Grid Size
	Size mGridSizeLeft, mGridSizeRight;
	int mGridNumberLeft;
	int mGridNumberRight;


	// x	  : left grid idx
	// y      :  right grid idx
	// value  : how many matches from idx_left to idx_right
	Mat mMotionStatistics;

	// 
	vector<int> mNumberPointsInPerCellLeft;

	// Inldex  : grid_idx_left
	// Value   : grid_idx_right
	vector<int> mCellPairs;

	// Every Matches has a cell-pair 
	// first  : grid_idx_left
	// second : grid_idx_right
	vector<pair<int, int> > mvMatchPairs;

	// Inlier Mask for output
	vector<bool> mvbInlierMask;

	//
	Mat mGridNeighborLeft;
	Mat mGridNeighborRight;


public:

	// Get Inlier Mask
	// Return number of inliers 
	int GetInlierMask(vector<bool> &vbInliers, bool WithScale = false, bool WithRotation = false);

private:

	// Normalize Key Points to Range(0 - 1) 规定化，将像素坐标转换到[0,1]范围内。
	void NormalizePoints(const vector<KeyPoint> &kp, const Size &size, vector<Point2f> &npts) {
		const size_t numP = kp.size();
		const int width   = size.width;
		const int height  = size.height;
		npts.resize(numP);

		for (size_t i = 0; i < numP; i++)
		{
			npts[i].x = kp[i].pt.x / width;
			npts[i].y = kp[i].pt.y / height;
		}
	}

	// Convert OpenCV DMatch to Match (pair<int, int>)将DMatch转换到Match

	//* DMatch主要用来储存匹配信息的结构体，query是要匹配的描述子，train是被匹配的描述子，在Opencv中进行匹配时
	//* void DescriptorMatcher::match(const Mat& queryDescriptors, const Mat& trainDescriptors, vector<DMatch>& matches, const Mat& mask) const
	//* match函数的参数中位置在前面的为query descriptor，后面的是 train descriptor
	//* 例如：query descriptor的数目为20，train descriptor数目为30，则DescriptorMatcher::match后的vector<DMatch>的size为20
	//* 若反过来，则vector<DMatch>的size为30
	void ConvertMatches(const vector<DMatch> &vDMatches, vector<pair<int, int> > &vMatches) {
		vMatches.resize(mNumberMatches);
		for (size_t i = 0; i < mNumberMatches; i++)
		{
			vMatches[i] = pair<int, int>(vDMatches[i].queryIdx, vDMatches[i].trainIdx);
		}
	}

	int GetGridIndexLeft(const Point2f &pt, int type) //获取匹配点所在网格索引值
	{
		int x = 0, y = 0;

		if (type == 1) {
			x = floor(pt.x * mGridSizeLeft.width);
			y = floor(pt.y * mGridSizeLeft.height);
		}

		if (type == 2) {
			x = floor(pt.x * mGridSizeLeft.width + 0.5);
			y = floor(pt.y * mGridSizeLeft.height);
		}

		if (type == 3) {
			x = floor(pt.x * mGridSizeLeft.width);
			y = floor(pt.y * mGridSizeLeft.height + 0.5);
		}

		if (type == 4) {
			x = floor(pt.x * mGridSizeLeft.width + 0.5);
			y = floor(pt.y * mGridSizeLeft.height + 0.5);
		}


		if (x >= mGridSizeLeft.width || y >= mGridSizeLeft.height)
		{
			return -1;
		}

		return x + y * mGridSizeLeft.width;
	}

	int GetGridIndexRight(const Point2f &pt) //获取待匹配点所在网格索引值
	{
		int x = floor(pt.x * mGridSizeRight.width);
		int y = floor(pt.y * mGridSizeRight.height);

		return x + y * mGridSizeRight.width;
	}

	// Assign Matches to Cell Pairs 
	void AssignMatchPairs(int GridType);

	// Verify Cell Pairs
	void VerifyCellPairs(int RotationType);

	// Get Neighbor 9
	vector<int> GetNB9(const int idx, const Size& GridSize) {
		vector<int> NB9(9, -1);

		int idx_x = idx % GridSize.width;
		int idx_y = idx / GridSize.width;//确定当前网格所在位置

		for (int yi = -1; yi <= 1; yi++)
		{
			for (int xi = -1; xi <= 1; xi++)
			{	
				int idx_xx = idx_x + xi;
				int idx_yy = idx_y + yi;

				if (idx_xx < 0 || idx_xx >= GridSize.width || idx_yy < 0 || idx_yy >= GridSize.height)
					continue;

				NB9[xi + 4 + yi * 3] = idx_xx + idx_yy * GridSize.width;
			}
		}
		return NB9;
	}

	//
	void InitalizeNiehbors(Mat &neighbor, const Size& GridSize) {
		for (int i = 0; i < neighbor.rows; i++)
		{
			vector<int> NB9 = GetNB9(i, GridSize);//返回当前网格邻域的网格索引矩阵
			int *data = neighbor.ptr<int>(i);
			memcpy(data, &NB9[0], sizeof(int) * 9);
		}
	}

	void SetScale(int Scale) //通过更改网格数量，改变尺度
	{
		// Set Scale
		mGridSizeRight.width = mGridSizeLeft.width  * mScaleRatios[Scale];
		mGridSizeRight.height = mGridSizeLeft.height * mScaleRatios[Scale];
		mGridNumberRight = mGridSizeRight.width * mGridSizeRight.height;//网格数量

		// Initialize the neihbor of right grid 
		mGridNeighborRight = Mat::zeros(mGridNumberRight, 9, CV_32SC1);//网格邻域矩阵
		InitalizeNiehbors(mGridNeighborRight, mGridSizeRight);
	}


	// Run 
	int run(int RotationType);
};


int gms_matcher::GetInlierMask(vector<bool> &vbInliers, bool WithScale, bool WithRotation) {

	int max_inlier = 0;

	if (!WithScale && !WithRotation)
	{
		SetScale(0);
		max_inlier = run(1);
		vbInliers = mvbInlierMask;
		return max_inlier;
	}

	if (WithRotation && WithScale)
	{
		for (int Scale = 0; Scale < 5; Scale++)
		{
			SetScale(Scale);
			for (int RotationType = 1; RotationType <= 8; RotationType++)
			{
				int num_inlier = run(RotationType);

				if (num_inlier > max_inlier)
				{
					vbInliers = mvbInlierMask;
					max_inlier = num_inlier;
				}
			}
		}
		return max_inlier;
	}

	if (WithRotation && !WithScale)
	{
		for (int RotationType = 1; RotationType <= 8; RotationType++)
		{
			int num_inlier = run(RotationType);

			if (num_inlier > max_inlier)
			{
				vbInliers = mvbInlierMask;
				max_inlier = num_inlier;
			}
		}
		return max_inlier;
	}

	if (!WithRotation && WithScale)
	{
		for (int Scale = 0; Scale < 5; Scale++)
		{
			SetScale(Scale);

			int num_inlier = run(1);

			if (num_inlier > max_inlier)
			{
				vbInliers = mvbInlierMask;
				max_inlier = num_inlier;
			}
			
		}
		return max_inlier;
	}

	return max_inlier;
}



void gms_matcher::AssignMatchPairs(int GridType) {

	for (size_t i = 0; i < mNumberMatches; i++)
	{
		Point2f &lp = mvP1[mvMatches[i].first];
		Point2f &rp = mvP2[mvMatches[i].second];

		int lgidx = mvMatchPairs[i].first = GetGridIndexLeft(lp, GridType);//获得匹配点所在网格的索引值
		int rgidx = -1;

		if (GridType == 1)
		{
			rgidx = mvMatchPairs[i].second = GetGridIndexRight(rp);//获得待匹配点所在网格的索引值
		}
		else
		{
			rgidx = mvMatchPairs[i].second;
		}

		if (lgidx < 0 || rgidx < 0)	continue;

		mMotionStatistics.at<int>(lgidx, rgidx)++;//网格对里有的匹配点对数量
		mNumberPointsInPerCellLeft[lgidx]++;//匹配网格内匹配点数量
	}

}


void gms_matcher::VerifyCellPairs(int RotationType) {

	const int *CurrentRP = mRotationPatterns[RotationType - 1];

	for (int i = 0; i < mGridNumberLeft; i++)
	{
		if (sum(mMotionStatistics.row(i))[0] == 0)//第i个网格内的匹配点对数量
		{
			mCellPairs[i] = -1;
			continue;
		}

		int max_number = 0;
		for (int j = 0; j < mGridNumberRight; j++)//第i个左网格中匹配点对数量最多的右网格
		{
			int *value = mMotionStatistics.ptr<int>(i);
			if (value[j] > max_number)
			{
				mCellPairs[i] = j;
				max_number = value[j];
			}
		}

		int idx_grid_rt = mCellPairs[i];

		const int *NB9_lt = mGridNeighborLeft.ptr<int>(i);
		const int *NB9_rt = mGridNeighborRight.ptr<int>(idx_grid_rt); 

		int score = 0;
		double thresh = 0;
		int numpair = 0;

		for (size_t j = 0; j < 9; j++)
		{
			int ll = NB9_lt[j];
			int rr = NB9_rt[CurrentRP[j] - 1];
			if (ll == -1 || rr == -1)	continue;

			score += mMotionStatistics.at<int>(ll, rr);
			thresh += mNumberPointsInPerCellLeft[ll];
			numpair++;
		}

		thresh = THRESH_FACTOR * sqrt(thresh / numpair);

		if (score < thresh)
			mCellPairs[i] = -2;
	}

}

int gms_matcher::run(int RotationType) {

	mvbInlierMask.assign(mNumberMatches, false);//匹配点对选择状态初始化

	// Initialize Motion Statisctics
	mMotionStatistics = Mat::zeros(mGridNumberLeft, mGridNumberRight, CV_32SC1);//匹配点对所对应的网格索引值矩阵
	mvMatchPairs.assign(mNumberMatches, pair<int, int>(0, 0));//匹配点对初始化

	for (int GridType = 1; GridType <= 4; GridType++) 
	{
		// initialize
		mMotionStatistics.setTo(0);
		mCellPairs.assign(mGridNumberLeft, -1);//匹配网格匹配上的网格索引
		mNumberPointsInPerCellLeft.assign(mGridNumberLeft, 0);//网格内匹配点对数量矩阵
		
		AssignMatchPairs(GridType);//确定每组点对所在的网格对位置
		VerifyCellPairs(RotationType);

		// Mark inliers
		for (size_t i = 0; i < mNumberMatches; i++)
		{
			if (mCellPairs[mvMatchPairs[i].first] == mvMatchPairs[i].second)
			{
				mvbInlierMask[i] = true;
			}
		}
	}
	int num_inlier = sum(mvbInlierMask)[0];
	return num_inlier;

}

// utility
inline Mat DrawInlier(Mat &src1, Mat &src2, vector<KeyPoint> &kpt1, vector<KeyPoint> &kpt2, vector<DMatch> &inlier, int type) {
	const int height = max(src1.rows, src2.rows);
	const int width = src1.cols + src2.cols;
	Mat output(height, width, CV_8UC3, Scalar(0, 0, 0));
	src1.copyTo(output(Rect(0, 0, src1.cols, src1.rows)));
	src2.copyTo(output(Rect(src1.cols, 0, src2.cols, src2.rows)));

	if (type == 1)
	{
		for (size_t i = 0; i < inlier.size(); i++)
		{
			Point2f left = kpt1[inlier[i].queryIdx].pt;
			Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
			line(output, left, right, Scalar(0, 255, 255));
		}
	}
	else if (type == 2)
	{
		for (size_t i = 0; i < inlier.size(); i++)
		{
			Point2f left = kpt1[inlier[i].queryIdx].pt;
			Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
			line(output, left, right, Scalar(255, 0, 0));
		}

		for (size_t i = 0; i < inlier.size(); i++)
		{
			Point2f left = kpt1[inlier[i].queryIdx].pt;
			Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
			circle(output, left, 1, Scalar(0, 255, 255), 2);
			circle(output, right, 1, Scalar(0, 255, 0), 2);
		}
	}

	return output;
}

inline void imresize(Mat &src, int height) {
	double ratio = src.rows * 1.0 / height;
	int width = static_cast<int>(src.cols * 1.0 / ratio);
	resize(src, src, Size(width, height));
}
