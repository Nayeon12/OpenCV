#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

using namespace cv;
using namespace std;

int main() {
	Mat frame;

	VideoCapture cap(0); // ù��° ī�޶�� ����

	if (!cap.isOpened()) {
		cerr << "ī�޶� �� �� �����ϴ�." << endl;
		return -1;
	}

	cap.read(frame);
	if (frame.empty()) {
		cerr << "ĸ�� ����" << endl;
		return -1;
	}

	imshow("Live", frame);
	waitKey(0);

	return 0;
}