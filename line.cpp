#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;
using namespace cv::ml;


// 3번
Mat src, dst, cdst;
vector<Vec4i> lines;
int line_threshold = 15, minLineLength = 10, maxLineGap = 20;

int main() {	
	// 3번
	VideoCapture vcap("highway.mp4");

	if (!vcap.isOpened()) {
		printf("Can't open the video");
		return -1;
	}

	while (vcap.isOpened()) {
		Mat vframe, vgframe;
		vcap >> vframe;

		// 비디오 끝나면 종료
		if (vframe.empty()) {
			cout << "Video END" << std::endl;
			break;
		}
		// esx키 누르면 종료
		if (waitKey(10) == 27) {
			break;
		}

		cvtColor(vframe, vgframe, COLOR_BGR2GRAY);

		Mat vdst;
		Canny(vgframe, vdst, 100, 200); 		// edge detection
		//cvtColor(vframe, vframe, COLOR_GRAY2BGR);

		HoughLinesP(vdst, lines, 1, CV_PI / 180, line_threshold, minLineLength, maxLineGap);
		for (size_t i = 0; i < lines.size(); i++) {
			Vec4i l = lines[i];
			line(vframe, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0), 1, LINE_AA);
		}  //칼라 영상 위에 라인 그리기
		
		imshow("edge Video", vdst);
		imshow("line Video", vframe);
		
		waitKey(10);
	}
	destroyWindow("edge Video");
	destroyWindow("line Video");

	waitKey(0);
	return 0;
}