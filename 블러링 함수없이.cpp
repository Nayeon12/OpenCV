#include "opencv2/opencv.hpp"
#include <iostream>
using namespace std;
using namespace cv;

void filter_blur();

int main() {

	filter_blur();

}

void filter_blur() {

	Mat src = imread("rose.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat dst = imread("rose.bmp", IMREAD_GRAYSCALE);
	Mat mask = Mat::ones(5, 5, CV_32SC1); //int
	//Mat mask(Size(5, 5), CV_32FC1, 1/25);
	//int mask[5][5] = { { 1,1,1,1,1 },
	//					{ 1,1,1,1,1 },
	//					{ 1,1,1,1,1 },
	//					{ 1,1,1,1,1 },
	//					{ 1,1,1,1,1 } };

	int sum;
	for (int y = 0; y < src.rows; y++) { //행 결과영상
		for (int x = 0; x < src.cols; x++) { //열
			sum = 0;
			for (int i = -2; i <= 2; i++) { // -2 -1 0 1 2
				for (int j = -2; j <= 2; j++) {
					int ny = y + i; 
					int nx = x + j;

					if (ny < 0) // 가장자리 픽셀
						ny = 0;
					else if (ny > src.rows - 1) 
						ny = src.rows - 1;

					if (nx < 0) 
						nx = 0;
					else if (nx > src.cols - 1) 
						nx = src.cols - 1;

					//sum += src.at<uchar>(ny, nx) * mask[2 + i][2 + j];
					sum += src.at<int>(ny, nx) * mask.at<int>(2 + i, 2 + j);
	
				}
			}
			sum = sum / 25;

			if (sum > 255) // 포화연산
				sum = 255; 
			if (sum < 0) 
				sum = 0;

			dst.at<int>(y, x) = sum; 
		}
	}

	imshow("src", src);
	imshow("dst", dst);
	waitKey();
	destroyAllWindows();
}