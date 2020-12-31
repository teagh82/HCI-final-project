#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;
using namespace cv::ml;

// knn 학습 위한 변수
Mat train_features(900, 400, CV_32FC1);
Mat labels(900, 1, CV_32FC1);
string fashion[10] = { "T-shirt", "pants", "shirt", "dress", "jacket", "shoes", "blouse", "sneakers", "bag", "boots" };

Ptr<ml::KNearest> knn_func(Mat img) {
	// 각 영상을 row vector로 만들어서 train_features에 저장한다. 
	for (int r = 0; r < 30; r++) {     // 세로로 30개
		for (int c = 0; c < 30; c++) { //가로로 30개
			int i = 0;
			for (int y = 0; y < 20; y++) { // 20x20
				for (int x = 0; x < 20; x++) {
					train_features.at<float>(r * 30 + c, i++)
						= img.at<uchar>(r * 20 + y, c * 20 + x);
				}
			}

		}
	}

	// 각 패션 영상에 대한 레이블을 저장한다. 
	for (int i = 0; i < 900; i++) {
		labels.at<float>(i, 0) = (i / 90);
	}

	// 학습 단계
	Ptr<ml::KNearest> knn = ml::KNearest::create();
	Ptr<ml::TrainData> trainData = ml::TrainData::create(train_features, ROW_SAMPLE, labels);
	knn->train(trainData);

	return knn;
}

// 2-1번
void n2(Mat img) {
	Ptr<ml::KNearest> knn = knn_func(img);

	// 맞은 개수 확인
	float cnt = 0;

	// 테스트 단계
	Mat predictedLabels;
	for (int i = 0; i < 900; i++) { // 900개에 대해서
		Mat test = train_features.row(i); //train 데이터를 test 데이터로 사용
		knn->findNearest(test, 3, predictedLabels);  // k=3
		float prediction = predictedLabels.at<float>(0);
		cout << "테스트 샘플" << i << "의 라벨 = " << fashion[(int)prediction] << '\n';

		// 맞은 개수
		if ((int)prediction == i / 90) {
			cnt+=1;
			//cout << "correct" << endl;
			//cout << cnt;
		}
	}

	// 정답률 구하기 (정답률 = 맞힌 개수 / 총 개수 x 100)
	float correct = cnt / 900 * 100;
	cout << "정답률 : " << correct << "%\n";
}

// 이미지 전처리
Mat pro_img(Mat test) {
	Mat gray_test;
	cvtColor(test, gray_test, COLOR_BGR2GRAY);

	Mat blur, th;

	Size size = Size(5, 5);
	GaussianBlur(gray_test, blur, size, 0);  // Blurring
	threshold(blur, th, 0, 255, THRESH_BINARY | THRESH_OTSU);

	Mat trans;
	bitwise_or(th, gray_test, trans);
	bitwise_not(trans, trans, trans);

	/*
	Mat gray_test;
	cvtColor(test, gray_test, COLOR_BGR2HSV);

	Mat kernel = (Mat_<int>(3, 3) <<
		1, 1, 1,
		1, 1, 1,
		1, 1, 1);

	Mat trans = gray_test.clone();
	
	inRange(gray_test, Scalar(0, 0, 0), Scalar(255, 100, 70), trans);	// HSV 영상의 black 색 구간을 이진화
	morphologyEx(trans, trans, MORPH_OPEN, kernel);	// OPEN 연산 이용해 배경에 나타나는 잡음 제거
*/
	return trans;
}

// test 위해
void test_f(Mat test_features, Mat trans) {
	// 각 숫자 영상을 row vector로 만들어서 test_features에 저장한다. 
	for (int r = 0; r < 1; r++) {     // 숫자 세로로 30개
		for (int c = 0; c < 1; c++) { //숫자 가로로 30개
			int i = 0;
			for (int y = 0; y < 20; y++) { // 20x20
				for (int x = 0; x < 20; x++) {
					test_features.at<float>(r * 1 + c, i++)
						= trans.at<uchar>(r * 20 + y, c * 20 + x);
				}
			}

		}
	}
}

// 2-2번
void n3(Mat img) {
	Mat test, test2, test3, test4;
	test = imread("test.jpg");	//컬러로 읽기
	resize(test, test, Size(400, 400));

	test2 = imread("test2.jpg");	//컬러로 읽기
	resize(test2, test2, Size(400, 400));

	test3 = imread("test3.jpg");	//컬러로 읽기
	resize(test3, test3, Size(400, 400));

	test4 = imread("test4.jpg");	//컬러로 읽기
	resize(test4, test4, Size(400, 400));

	imshow("color test1 img", test);
	imshow("color test2 img", test2);
	imshow("color test3 img", test3);
	imshow("color test4 img", test4);

	Mat trans = pro_img(test);
	imshow("trans1", trans);
	resize(trans, trans, Size(20, 20));

	Mat trans2 = pro_img(test2);
	imshow("trans2", trans2);
	resize(trans2, trans2, Size(20, 20));

	Mat trans3 = pro_img(test3);
	imshow("trans3", trans3);
	resize(trans3, trans3, Size(20, 20));

	Mat trans4 = pro_img(test4);
	imshow("trans4", trans4);
	resize(trans4, trans4, Size(20, 20));


	Mat test_features(1, 400, CV_32FC1);
	test_f(test_features, trans);

	Mat test2_features(1, 400, CV_32FC1);
	test_f(test2_features, trans2);
	

	Mat test3_features(1, 400, CV_32FC1);
	test_f(test3_features, trans3);

	Mat test4_features(1, 400, CV_32FC1);
	test_f(test4_features, trans4);
	
	// 훈련
	Ptr<ml::KNearest> knn = knn_func(img);

	// 테스트 단계
	Mat predictedLabels;
	for (int i = 0; i < 1; i++) { // 900개에 대해서
		Mat test1 = test_features.row(i); // test1
		knn->findNearest(test1, 3, predictedLabels);  // k=3
		float prediction = predictedLabels.at<float>(0);
		cout << "테스트 1의 라벨 = " << fashion[(int)prediction] << '\n';
	}

	for (int i = 0; i < 1; i++) { // 900개에 대해서
		Mat test2 = test2_features.row(i); // test2
		knn->findNearest(test2, 3, predictedLabels);  // k=3
		float prediction = predictedLabels.at<float>(0);
		cout << "테스트 2의 라벨 = " << fashion[(int)prediction] << '\n';

	}

	for (int i = 0; i < 1; i++) { // 900개에 대해서
		Mat test3 = test3_features.row(i); // test3
		knn->findNearest(test3, 3, predictedLabels);  // k=3
		float prediction = predictedLabels.at<float>(0);
		cout << "테스트 3의 라벨 = " << fashion[(int)prediction] << '\n';

	}

	for (int i = 0; i < 1; i++) { // 900개에 대해서
		Mat test4 = test4_features.row(i); // test4
		knn->findNearest(test4, 3, predictedLabels);  // k=3
		float prediction = predictedLabels.at<float>(0);
		cout << "테스트 4의 라벨 = " << fashion[(int)prediction] << '\n';

	}
}

void n4(Mat img) {
	threshold(img, img, 127, 255, cv::THRESH_BINARY);

	Mat skel(img.size(), CV_8UC1, Scalar(0)); // skeleton = 0
	Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));
	Mat temp, eroded;

	do
	{
		erode(img, eroded, element);
		dilate(eroded, temp, element);
		subtract(img, temp, temp);  // 빼기 : outline 돌출 부분
		bitwise_or(skel, temp, skel); //OR : 기존의 skeleto에 합침.
		eroded.copyTo(img);
	} while ((countNonZero(img) != 0));

	imshow("Skeletonization", skel);

	string new_fashion[10] = { "top", "pants", "top", "dress", "top", "shoes", "top", "sneakers", "bag", "boots" };

	
	Ptr<ml::KNearest> knn = knn_func(skel);

	// 맞은 개수 확인
	float cnt = 0;

	// 테스트 단계
	Mat predictedLabels;
	for (int i = 0; i < 900; i++) { // 900개에 대해서
		Mat test = train_features.row(i); //train 데이터를 test 데이터로 사용
		knn->findNearest(test, 3, predictedLabels);  // k=3
		float prediction = predictedLabels.at<float>(0);
		cout << "테스트 샘플" << i << "의 라벨 = " << new_fashion[(int)prediction] << '\n';
		
		// 맞은 개수
		if ((int)prediction == i / 90) {
			cnt += 1;
			//cout << "correct" << endl;
			//cout << cnt;
		}
	}

	// 정답률 구하기 (정답률 = 맞힌 개수 / 총 개수 x 100)
	float correct = cnt / 900 * 100;
	cout << "정답률 : " << correct << "%\n";
}

int main() {	
	// 2번
	Mat img;
	img = imread("digits.png", IMREAD_GRAYSCALE);
	imshow("original", img);
	namedWindow("original", WINDOW_AUTOSIZE);
	Mat img2 = img.clone();

	while (1)
	{
		// 2번
		int key = waitKeyEx();	// 사용자로부터 키를 기다림

		if (key == 'q') break;	// 사용자가 ‘q'를 누르면 종료
		else if (key == '1') {	// 2-1번
			n2(img);
		}
		else if (key == '2') {	// 2-2번
			n3(img);
		}
		else if (key == '3') {	// 2-3번
			n4(img2);
		}
	}

	waitKey(0);
	return 0;
}