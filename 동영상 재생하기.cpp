#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;

int main() {
	Mat frame;

	VideoCapture cap("output.avi");// ù��° ī�޶�� ����

	if (!cap.isOpened()) {
		cerr << "ī�޶� �� �� �����ϴ�." << endl;
		return -1;
	}

	int fps = cap.get(CAP_PROP_FPS); // �ʴ� ������ ��
	//int width = cap.get(CAP_PROP_FRAME_WIDTH);
	//int height = cap.get(CAP_PROP_FRAME_HEIGHT);
	//int fourcc = VideoWriter::fourcc('M', 'J', 'P', 'G'); 
	// ���� ĸ�� ��ü
	// ī�޶󿡼� ĸ�ĵǴ� �̹��� ũ��, �ڵ� ����

	//VideoWriter outputVideo;
	//outputVideo.open("output.avi", fourcc, fps, Size(width, height), true);
	//����϶��� fourcc �� -1�� ����

	//if (!outputVideo.isOpened()) {
	//	cerr << "������ ������ ���� �ʱ�ȭ�� ����" << endl;
	//	return -1;
	//}

	while (1) {

		cap.read(frame);
		if (frame.empty()) {
			cerr << "ĸ�� ����" << endl;
			break;
		}

		//cvtColor(frame, frame, COLOR_BGR2GRAY);
		imshow("Live", frame);

		//outputVideo.write(frame);

		int wait = int(1.0 / fps * 1000); // fps�� ���� �̹����� �������� ���� ��� �ð�
		if (waitKey(wait) >= 0)
			break;
	}

	return 0;
}