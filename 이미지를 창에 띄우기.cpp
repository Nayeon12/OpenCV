#include "opencv2/opencv.hpp"
#include <iostream>
using namespace std;
using namespace cv;

int main() {
	cout << CV_VERSION << endl;

	Mat img = imread("luzzi.jpg"); // �̹��� ���� // ���� �̸�

	if (img.empty()) {
		cerr << "File open failed!" << endl;
		return -1;
	}

	namedWindow("image"); // �̹������ â�� ȭ�鿡 ���
	imshow("image", img); // �̹��� ��, �̹���(MatŸ��)�� �̹������ Ī�� ������

	waitKey(); // Ű�� ���� ������ ȭ�� ����

	return 0;
}