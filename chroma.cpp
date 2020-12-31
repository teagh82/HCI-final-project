#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;
using namespace cv::ml;

int main() {
	// 1��
	Mat back = imread("back.jpg", IMREAD_COLOR);
	//resize(back, back, Size(400, 400));
	Mat clone = back.clone();

	VideoCapture cap(0); // real-time camera capture  
	if (!cap.isOpened()) { cout << "file not found" << endl; }

	Mat kernel = (Mat_<int>(3, 3) <<
		1, 1, 1,
		1, 1, 1,
		1, 1, 1);

	Rect bbox(287, 23, 86, 320);

	while (1)
	{
		Mat imgHSV;
		Mat frame;
		cap >> frame;	// �������� �������� ����

		resize(frame, frame, Size(400, 400));

		cvtColor(frame, imgHSV, COLOR_BGR2HSV);		// BGR ���� ������ HSV ���� �������� ��ȯ

		Mat blueScreen = imgHSV.clone();

		inRange(imgHSV, Scalar(100, 100, 100), Scalar(120, 255, 255), blueScreen);	// HSV ������ blue �� ������ ����ȭ
		morphologyEx(blueScreen, blueScreen, MORPH_OPEN, kernel);	// OPEN ���� �̿��� ��濡 ��Ÿ���� ���� ����

		/*
		Mat dst, dst1, inverted;
		bitwise_not(blueScreen, inverted);
		bitwise_and(frame, frame, dst, inverted);
		bitwise_or(dst, back, dst1, blueScreen);
		bitwise_or(dst, dst1, dst1);
		*/

		int minx = 500, miny = 500, maxx = -1, maxy = -1;

		for (int y = 0; y < blueScreen.rows; ++y) {
			for (int x = 0; x < blueScreen.cols; ++x) {
				if (blueScreen.at<uchar>(y, x) == 255) { // ����ũ �̹����� ��� �ȼ��̸� 
					if (minx > x)
						minx = x;
					if (miny > y)
						miny = y;
					if (maxx < x)
						maxx = x;
					if (maxy < y)
						maxy = y;

				}
			}
		}

		if (maxx > 50 && maxy > 80) {
			Mat roi(frame, Rect(minx, miny, maxx - minx, maxy - miny));
			resize(clone, back, Size(maxx - minx, maxy - miny));
			back.copyTo(roi);
		}

		rectangle(frame, Rect(minx, miny, maxx - minx, maxy - miny), Scalar(0, 0, 255), 2);

		imshow("blue", blueScreen);
		imshow("dst", frame);

		if (waitKey(30) >= 0) break;

	}

	waitKey(0);
	return 0;
}