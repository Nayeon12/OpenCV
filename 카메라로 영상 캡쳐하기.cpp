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

	int fps = cap.get(CAP_PROP_FPS); // 초당 프레임 수

	while (1) {

		cap.read(frame);
		if (frame.empty()) {
			cerr << "캡쳐 실패" << endl;
			break;
		}

		imshow("Live", frame);

		int wait = int(1.0 / fps * 1000); // fps을 통해 이미지를 가져오는 사이 대기 시간
		if (waitKey(wait) >= 0)
			break;
	}
	return 0;
}