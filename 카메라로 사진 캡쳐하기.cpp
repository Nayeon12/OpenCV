#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

using namespace cv;
using namespace std;

int main() {
	Mat frame;

	VideoCapture cap(0); // 첫번째 카메라와 연결

	if (!cap.isOpened()) {
		cerr << "카메라를 열 수 없습니다." << endl;
		return -1;
	}

	cap.read(frame);
	if (frame.empty()) {
		cerr << "캡쳐 실패" << endl;
		return -1;
	}

	imshow("Live", frame);
	waitKey(0);

	return 0;
}