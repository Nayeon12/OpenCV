#include "opencv2/opencv.hpp"
#include <iostream>
using namespace std;
using namespace cv;

int main() {
	cout << CV_VERSION << endl;

	Mat img = imread("luzzi.jpg"); // 이미지 리드 // 파일 이름

	if (img.empty()) {
		cerr << "File open failed!" << endl;
		return -1;
	}

	namedWindow("image"); // 이미지라는 창을 화면에 띄움
	imshow("image", img); // 이미지 쇼, 이미지(Mat타입)를 이미지라는 칭에 보여줌

	waitKey(); // 키를 누를 때까지 화면 유지

	return 0;
}