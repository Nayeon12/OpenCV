#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;

int main() {
	Mat frame;

	VideoCapture cap("output.avi");// 첫번째 카메라와 연결

	if (!cap.isOpened()) {
		cerr << "카메라를 열 수 없습니다." << endl;
		return -1;
	}

	int fps = cap.get(CAP_PROP_FPS); // 초당 프레임 수
	//int width = cap.get(CAP_PROP_FRAME_WIDTH);
	//int height = cap.get(CAP_PROP_FRAME_HEIGHT);
	//int fourcc = VideoWriter::fourcc('M', 'J', 'P', 'G'); 
	// 비디오 캡쳐 개체
	// 카메라에서 캡쳐되는 이미지 크기, 코덱 지정

	//VideoWriter outputVideo;
	//outputVideo.open("output.avi", fourcc, fps, Size(width, height), true);
	//흑백일때는 fourcc 를 -1로 지정

	//if (!outputVideo.isOpened()) {
	//	cerr << "동영상 저장을 위한 초기화중 에러" << endl;
	//	return -1;
	//}

	while (1) {

		cap.read(frame);
		if (frame.empty()) {
			cerr << "캡쳐 실패" << endl;
			break;
		}

		//cvtColor(frame, frame, COLOR_BGR2GRAY);
		imshow("Live", frame);

		//outputVideo.write(frame);

		int wait = int(1.0 / fps * 1000); // fps을 통해 이미지를 가져오는 사이 대기 시간
		if (waitKey(wait) >= 0)
			break;
	}

	return 0;
}