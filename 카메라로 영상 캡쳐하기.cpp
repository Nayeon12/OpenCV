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

	int fps = cap.get(CAP_PROP_FPS); // �ʴ� ������ ��

	while (1) {

		cap.read(frame);
		if (frame.empty()) {
			cerr << "ĸ�� ����" << endl;
			break;
		}

		imshow("Live", frame);

		int wait = int(1.0 / fps * 1000); // fps�� ���� �̹����� �������� ���� ��� �ð�
		if (waitKey(wait) >= 0)
			break;
	}
	return 0;
}