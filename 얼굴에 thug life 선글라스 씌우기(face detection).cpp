#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;


void detectAndDraw(Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade,
	double scale, bool tryflip, Mat glasses);
void overlayImage(const Mat &background, const Mat &foreground,
	Mat &output, Point2i location);

string cascadeName;
string nestedCascadeName;

int main(int argc, const char** argv)
{
	VideoCapture capture;
	Mat frame, image, glasses;
	string inputName; // = "youknowyunho.png"; // �������� ����
	string glassesImage = "sunglasses.png"; // ���۶� �̹����� ������
	bool tryflip = false;
	CascadeClassifier cascade, nestedCascade;
	double scale;

	scale = 1;

	glasses = imread(glassesImage, IMREAD_UNCHANGED);
	if (glasses.empty()) {

		cout << "Could not read image - " << glassesImage << endl;
		return -1;
	}


	if (!nestedCascade.load(samples::findFileOrKeep("haarcascade_eye_tree_eyeglasses.xml")))
		cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
	if (!cascade.load(samples::findFile("haarcascade_frontalface_alt.xml")))
	{
		cerr << "ERROR: Could not load classifier cascade" << endl;
		return -1;
	}
	if (inputName.empty() || (isdigit(inputName[0]) && inputName.size() == 1))
	{
		int camera = inputName.empty() ? 0 : inputName[0] - '0';
		if (!capture.open(camera))
		{
			cout << "Capture from camera #" << camera << " didn't work" << endl;
			return 1;
		}
	}
	else if (!inputName.empty()) // �̹���
	{
		image = imread(samples::findFileOrKeep(inputName), IMREAD_COLOR);
		if (image.empty())
		{
			if (!capture.open(samples::findFileOrKeep(inputName)))
			{
				cout << "Could not read " << inputName << endl;
				return 1;
			}
		}

		detectAndDraw(image, cascade, nestedCascade, scale, tryflip, glasses);

		waitKey(0);
	}


	if (capture.isOpened()) // ����
	{
		cout << "Video capturing has been started ..." << endl;

		for (;;)
		{
			capture >> frame;
			if (frame.empty())
				break;

			Mat frame1 = frame.clone();
			detectAndDraw(frame1, cascade, nestedCascade, scale, tryflip, glasses);

			char c = (char)waitKey(10);
			if (c == 27 || c == 'q' || c == 'Q')
				break;
		}
	}

	return 0;
}

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade,
	double scale, bool tryflip, Mat glasses) // �󱼰���
{

	Mat output2;
	img.copyTo(output2); // �����̹��� ����

	double t = 0;
	vector<Rect> faces, faces2;
	const static Scalar colors[] =
	{
		Scalar(255,0,0),
		Scalar(255,128,0),
		Scalar(255,255,0),
		Scalar(0,255,0),
		Scalar(0,128,255),
		Scalar(0,255,255),
		Scalar(0,0,255),
		Scalar(255,0,255)
	};
	Mat gray, smallImg;

	cvtColor(img, gray, COLOR_BGR2GRAY);
	double fx = 1 / scale;
	resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR_EXACT);
	equalizeHist(smallImg, smallImg);

	t = (double)getTickCount();
	cascade.detectMultiScale(smallImg, faces,
		1.1, 2, 0
		//|CASCADE_FIND_BIGGEST_OBJECT
		//|CASCADE_DO_ROUGH_SEARCH
		| CASCADE_SCALE_IMAGE,
		Size(30, 30)); // ����ġ ����

	t = (double)getTickCount() - t;

	Mat result;

	printf("detection time = %g ms\n", t * 1000 / getTickFrequency());
	for (size_t i = 0; i < faces.size(); i++)
	{
		Rect r = faces[i];
		Mat smallImgROI;
		vector<Rect> nestedObjects;
		Point center;
		Scalar color = colors[i % 8];
		int radius;



		double aspect_ratio = (double)r.width / r.height;

		if (0.75 < aspect_ratio && aspect_ratio < 1.3)
		{
			center.x = cvRound((r.x + r.width*0.5)*scale);
			center.y = cvRound((r.y + r.height*0.5)*scale);
			radius = cvRound((r.width + r.height)*0.25*scale);
			circle(img, center, radius, color, 3, 8, 0); // ����ġ�� ���� �׷���
		}
		else
			rectangle(img, Point(cvRound(r.x*scale), cvRound(r.y*scale)),
				Point(cvRound((r.x + r.width - 1)*scale), cvRound((r.y + r.height - 1)*scale)),
				color, 3, 8, 0);
		if (nestedCascade.empty()) {
			cout << "nestedCascade.empty()" << endl;
			continue;
		}

		smallImgROI = smallImg(r);
		nestedCascade.detectMultiScale(smallImgROI, nestedObjects,
			1.1, 2, 0
			//|CASCADE_FIND_BIGGEST_OBJECT
			//|CASCADE_DO_ROUGH_SEARCH
			//|CASCADE_DO_CANNY_PRUNING
			| CASCADE_SCALE_IMAGE,
			Size(20, 20)); // ����ġ ����


		cout << nestedObjects.size() << endl;

		vector<Point> points;

		for (size_t j = 0; j < nestedObjects.size(); j++)
		{
			Rect nr = nestedObjects[j];
			center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
			center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
			radius = cvRound((nr.width + nr.height)*0.25*scale);
			circle(img, center, radius, color, 3, 8, 0);

			Point p(center.x, center.y);
			points.push_back(p);
		} // ����ġ�� �� �׸�


		if (points.size() == 2) {

			Point center1 = points[0];
			Point center2 = points[1];

			if (center1.x > center2.x) {
				Point temp;
				temp = center1;
				center1 = center2;
				center2 = temp;
			} // �� ��ġ�� 2���� ����� ��� x ��ǥ �������� ����


			int width = abs(center2.x - center1.x); // ����ġ�� �ƴ� ��츦 
			int height = abs(center2.y - center1.y); // ���ο� ���� ���̷� ����

			if (width > height) {

				float imgScale = width / 600.0; // ������ ���ݰ� �Ȱ�� ���� ���� ���� ��� 

				int w, h;
				w = glasses.cols * imgScale;
				h = glasses.rows * imgScale;

				int offsetX = 600 * imgScale;
				int offsetY = 450 * imgScale; // ������ �Ȱ�� �߽���ǥ�� �Ȱ���ġ ����

				Mat resized_glasses;
				resize(glasses, resized_glasses, cv::Size(w, h), 0, 0); // �Ȱ�ũ�� ����

				overlayImage(output2, resized_glasses, result, Point(center1.x - offsetX, center1.y - offsetY)); // ���̹����� �Ȱ��̹��� ������
				output2 = result;
			}
		}
	}

	if (result.empty())
		imshow("result", img);
	else
		imshow("result", result); // �Ȱ��� ������ ���ߴٸ� ���� ����� ������

}


void overlayImage(const Mat &background, const Mat &foreground,
	Mat &output, Point2i location)
{
	background.copyTo(output);


	// start at the row indicated by location, or at row 0 if location.y is negative.
	for (int y = std::max(location.y, 0); y < background.rows; ++y)
	{
		int fY = y - location.y; // because of the translation

		// we are done of we have processed all rows of the foreground image.
		if (fY >= foreground.rows) {
			break;
		}

		// start at the column indicated by location, 

		// or at column 0 if location.x is negative.
		for (int x = std::max(location.x, 0); x < background.cols; ++x)
		{
			int fX = x - location.x; // because of the translation.

									 // we are done with this row if the column is outside of the foreground image.
			if (fX >= foreground.cols) {
				break;
			}

			// determine the opacity of the foregrond pixel, using its fourth (alpha) channel.
			double opacity =
				((double)foreground.data[fY * foreground.step + fX * foreground.channels() + 3])

				/ 255.;


			// and now combine the background and foreground pixel, using the opacity, 

			// but only if opacity > 0.
			for (int c = 0; opacity > 0 && c < output.channels(); ++c)
			{
				unsigned char foregroundPx =
					foreground.data[fY * foreground.step + fX * foreground.channels() + c];
				unsigned char backgroundPx =
					background.data[y * background.step + x * background.channels() + c];
				output.data[y*output.step + output.channels()*x + c] =
					backgroundPx * (1. - opacity) + foregroundPx * opacity;
			}
		}
	}
}