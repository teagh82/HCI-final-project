#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;
using namespace cv::ml;

// knn �н� ���� ����
Mat train_features(900, 400, CV_32FC1);
Mat labels(900, 1, CV_32FC1);
string fashion[10] = { "T-shirt", "pants", "shirt", "dress", "jacket", "shoes", "blouse", "sneakers", "bag", "boots" };

Ptr<ml::KNearest> knn_func(Mat img) {
	// �� ������ row vector�� ���� train_features�� �����Ѵ�. 
	for (int r = 0; r < 30; r++) {     // ���η� 30��
		for (int c = 0; c < 30; c++) { //���η� 30��
			int i = 0;
			for (int y = 0; y < 20; y++) { // 20x20
				for (int x = 0; x < 20; x++) {
					train_features.at<float>(r * 30 + c, i++)
						= img.at<uchar>(r * 20 + y, c * 20 + x);
				}
			}

		}
	}

	// �� �м� ���� ���� ���̺��� �����Ѵ�. 
	for (int i = 0; i < 900; i++) {
		labels.at<float>(i, 0) = (i / 90);
	}

	// �н� �ܰ�
	Ptr<ml::KNearest> knn = ml::KNearest::create();
	Ptr<ml::TrainData> trainData = ml::TrainData::create(train_features, ROW_SAMPLE, labels);
	knn->train(trainData);

	return knn;
}

// 2-1��
void n2(Mat img) {
	Ptr<ml::KNearest> knn = knn_func(img);

	// ���� ���� Ȯ��
	float cnt = 0;

	// �׽�Ʈ �ܰ�
	Mat predictedLabels;
	for (int i = 0; i < 900; i++) { // 900���� ���ؼ�
		Mat test = train_features.row(i); //train �����͸� test �����ͷ� ���
		knn->findNearest(test, 3, predictedLabels);  // k=3
		float prediction = predictedLabels.at<float>(0);
		cout << "�׽�Ʈ ����" << i << "�� �� = " << fashion[(int)prediction] << '\n';

		// ���� ����
		if ((int)prediction == i / 90) {
			cnt+=1;
			//cout << "correct" << endl;
			//cout << cnt;
		}
	}

	// ����� ���ϱ� (����� = ���� ���� / �� ���� x 100)
	float correct = cnt / 900 * 100;
	cout << "����� : " << correct << "%\n";
}

// �̹��� ��ó��
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
	
	inRange(gray_test, Scalar(0, 0, 0), Scalar(255, 100, 70), trans);	// HSV ������ black �� ������ ����ȭ
	morphologyEx(trans, trans, MORPH_OPEN, kernel);	// OPEN ���� �̿��� ��濡 ��Ÿ���� ���� ����
*/
	return trans;
}

// test ����
void test_f(Mat test_features, Mat trans) {
	// �� ���� ������ row vector�� ���� test_features�� �����Ѵ�. 
	for (int r = 0; r < 1; r++) {     // ���� ���η� 30��
		for (int c = 0; c < 1; c++) { //���� ���η� 30��
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

// 2-2��
void n3(Mat img) {
	Mat test, test2, test3, test4;
	test = imread("test.jpg");	//�÷��� �б�
	resize(test, test, Size(400, 400));

	test2 = imread("test2.jpg");	//�÷��� �б�
	resize(test2, test2, Size(400, 400));

	test3 = imread("test3.jpg");	//�÷��� �б�
	resize(test3, test3, Size(400, 400));

	test4 = imread("test4.jpg");	//�÷��� �б�
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
	
	// �Ʒ�
	Ptr<ml::KNearest> knn = knn_func(img);

	// �׽�Ʈ �ܰ�
	Mat predictedLabels;
	for (int i = 0; i < 1; i++) { // 900���� ���ؼ�
		Mat test1 = test_features.row(i); // test1
		knn->findNearest(test1, 3, predictedLabels);  // k=3
		float prediction = predictedLabels.at<float>(0);
		cout << "�׽�Ʈ 1�� �� = " << fashion[(int)prediction] << '\n';
	}

	for (int i = 0; i < 1; i++) { // 900���� ���ؼ�
		Mat test2 = test2_features.row(i); // test2
		knn->findNearest(test2, 3, predictedLabels);  // k=3
		float prediction = predictedLabels.at<float>(0);
		cout << "�׽�Ʈ 2�� �� = " << fashion[(int)prediction] << '\n';

	}

	for (int i = 0; i < 1; i++) { // 900���� ���ؼ�
		Mat test3 = test3_features.row(i); // test3
		knn->findNearest(test3, 3, predictedLabels);  // k=3
		float prediction = predictedLabels.at<float>(0);
		cout << "�׽�Ʈ 3�� �� = " << fashion[(int)prediction] << '\n';

	}

	for (int i = 0; i < 1; i++) { // 900���� ���ؼ�
		Mat test4 = test4_features.row(i); // test4
		knn->findNearest(test4, 3, predictedLabels);  // k=3
		float prediction = predictedLabels.at<float>(0);
		cout << "�׽�Ʈ 4�� �� = " << fashion[(int)prediction] << '\n';

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
		subtract(img, temp, temp);  // ���� : outline ���� �κ�
		bitwise_or(skel, temp, skel); //OR : ������ skeleto�� ��ħ.
		eroded.copyTo(img);
	} while ((countNonZero(img) != 0));

	imshow("Skeletonization", skel);

	string new_fashion[10] = { "top", "pants", "top", "dress", "top", "shoes", "top", "sneakers", "bag", "boots" };

	
	Ptr<ml::KNearest> knn = knn_func(skel);

	// ���� ���� Ȯ��
	float cnt = 0;

	// �׽�Ʈ �ܰ�
	Mat predictedLabels;
	for (int i = 0; i < 900; i++) { // 900���� ���ؼ�
		Mat test = train_features.row(i); //train �����͸� test �����ͷ� ���
		knn->findNearest(test, 3, predictedLabels);  // k=3
		float prediction = predictedLabels.at<float>(0);
		cout << "�׽�Ʈ ����" << i << "�� �� = " << new_fashion[(int)prediction] << '\n';
		
		// ���� ����
		if ((int)prediction == i / 90) {
			cnt += 1;
			//cout << "correct" << endl;
			//cout << cnt;
		}
	}

	// ����� ���ϱ� (����� = ���� ���� / �� ���� x 100)
	float correct = cnt / 900 * 100;
	cout << "����� : " << correct << "%\n";
}

int main() {	
	// 2��
	Mat img;
	img = imread("digits.png", IMREAD_GRAYSCALE);
	imshow("original", img);
	namedWindow("original", WINDOW_AUTOSIZE);
	Mat img2 = img.clone();

	while (1)
	{
		// 2��
		int key = waitKeyEx();	// ����ڷκ��� Ű�� ��ٸ�

		if (key == 'q') break;	// ����ڰ� ��q'�� ������ ����
		else if (key == '1') {	// 2-1��
			n2(img);
		}
		else if (key == '2') {	// 2-2��
			n3(img);
		}
		else if (key == '3') {	// 2-3��
			n4(img2);
		}
	}

	waitKey(0);
	return 0;
}