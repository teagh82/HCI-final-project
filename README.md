# HCI-final-project

## 실행결과

### chroma

<img src="https://user-images.githubusercontent.com/59547069/103400701-6c080600-4b89-11eb-95f6-7d9f7194cf26.png" width="700" height="370">
<br>

1. 파란색 부분에 보여줄 이미지를 back 이름으로 불러들임
2. back을 계속해서 resize를 해 휴대폰의 blue 부분에 보여주어야 하는데 그대로 back을 여러번 resize하면 이미지가 블러처리 된 것처럼 되기 때문에 이를 방지하기 위해 back의 복사본을 clone 이름으로 만든다.
3. 실시간 영상을 캡처한다.
4. 실시간 영상의 프레임을 HSV로 바꿔준다.
5. blueScreen에 HSV로 바꾼 imgHSV를 복제한다.
6. inRange로 blue 색상의 공간에 맞게 HSV 값을 주어서 이진화한다.
7. 배경에 나타나는 잡음을 제거하기 위해 모폴로지 OPEN 연산을 이용한다.
8. blueScreen 공간은 blue가 흰색이기 때문에 이 공간에서 픽셀 값이 255라면 그 픽셀의 값을 저장한다. 사각형을 그리기 위해 minx,miny, maxx, maxy가 필요하므로 이를 저장한다.
9. 원본 프레임의 일부분을 잘라서 그 공간에 내가 넣고 싶은 이미지를 넣어야 하기 때문에 roi를 만들어주는데 이때 minx,miny, maxx,maxy를 이용해서 잘라주는 과정에서 예외가 발생할 수 있기 때문에 maxx>50 && maxy>80 이라는 조건을 넣어주었다. 그리고 그 안에서 frame의 일부 영역을 자르고 내가 넣고 싶은 사진의 사이즈를 blue 크기에 맞게 변경시킨 후 roi에 copy 해준다.
10. minx,miny, maxx,maxy를 이용해 blue 영역에 맞도록 프레임에 사각형을 그려준다. 처음 시작점은 minx, miny 이고 width는 maxx-minx, height는 maxy-miny로 구한다. 마지막에 frame 이미지와 blueScreen을 띄운다.

### knn

<img src="https://user-images.githubusercontent.com/59547069/103400719-85a94d80-4b89-11eb-9b08-85fd068660d3.png" width="600" height="600">
<br>
<img src="https://user-images.githubusercontent.com/59547069/103400743-9c4fa480-4b89-11eb-810e-f4695b17f199.png" width="600" height="600">
<br>
<img src="https://user-images.githubusercontent.com/59547069/103400729-9063e280-4b89-11eb-9cc2-f6fef7d27692.png" width="300" height="220">
<br>
<img src="https://user-images.githubusercontent.com/59547069/103400609-0fa4e680-4b89-11eb-9df9-0fd7b0969ecf.png" width="600" height="600">
<br>

1. main에서 img에 제공받은 데이터셋 이미지를 저장한다. 그리고 키보드 이벤트를 위해 nameWindow도 설정하고 이 이미지를 “original ” 창으로 보여준다.
2. 사용자로부터 키를 기다리다가 q를 누르면 종료시키고 1을 누르면 n2(img) 함수가 실행된다.
3. n2()는 이미지를 파라미터로 받고 그 이미지로 훈련을 실시한다. 훈련을 하는 함수는 knn_func(img)로 따로 빼주었다.
4. knn_func()을 보면 데이터셋의 각 이미지를 row vector로 만들어서 train_features에 저장한다. train_features는 전역변수로 정의되어있고 총 900개의 패션이미지가 있고, 20x20 픽셀이므로 train_features(900, 400, CV_32FC1) 이렇게 정의한다. 흑백 이미지이기 때문에 채널은 한 개다. 그 후에는 각 패션 이미지에 대한 레이블을 저장한다. 총 900개이고 한 패션 아이템 당 90개가 있기 때문에 그것을 고려하여 저장한다. lables은 마찬가지로 전역변수로 정의하였다. 이렇게 하면 해당 레이블을 의미하는 숫자가 labels에 저장된다.
opencv의 ml의 Knearest::create를 이용하여 train_features로 knn 학습시킨다. 그리고 이렇게 학습된 knn을 반환한다.
5. 다시 n2()로 돌아와서 맞은 개수 확인을 위한 cnt를 정의했다. 그리고 테스트 단계를 거친다. 훈련 데이터를 테스트 데이터로 사용할 것이므로 총 900개에 대해서 반복하며 test에 현재 row의 train_features 값을 저장한다. 그리고 knn의 findNearest()를 이용하여 예측을 한다. 그 결과 값은 predictedLabels에 저장되고 0번째 값을 prediction 변수에 저장한다. 이 값은 0부터 9까지의 레이블 값이므로 이 값을 전역변수로 정의해놓은 패션 아이템 이름 배열의 인덱스로 넣어서 이 이미지의 예측 값을 보여준다.
6. 맞은 개수를 확인하기 위해 예측한 값과 그 레이블의 값인 (i/90)을 비교하여 같으면 cnt를 1 더해준다.
7. 마지막으로 cnt / 900 * 100로 정답률을 구해 출력한다.
8. 2번 키보드를 누르면 n3()이 실행된다. 훈련을 위한 이미지 데이터 셋을 파라미터로 받고 이것으로 훈련한 뒤 테스트를 실시한다.
9. 위와 다른 점은 훈련데이터가 아닌 실제 다른 데이터로 테스트를 한다는 점이다. 그래서 먼저 훈련 이미지와 유사하게 만들기 위해 이미지를 전처리하는 과정을 거친다. 4개의 이미지를 테스트 할 것이므로 test, test2, test3, test4 변수를 만들고 이미지를 읽는다. 그리고 사이즈를 조절한 뒤 보여준다. 이미지 전처리 과정은 pro_img() 함수로 따로 빼주었다. 그 함수를 보면 먼저 이미지를 gray_scale로 바꿔주고 가우시안 블러를 실시한 후 OTSU 이진화를 해준다. 블러를 위한 ksize는 5x5로 해주었다. 이렇게 이진화를 거친 이유는 패션 아이템의 부분만 찾아내기 위함이다. 이후 이진화된 이미지는 패션 아이템 부분만 어둡고 배경은 밝으므로 or 논리 연산을 실시한 후 not 연산을 하면 배경은 검고 패션 아이템만 볼 수 있다.
10. 마지막으로 3을 누르면 n4()가 실행된다. 입력받은 이미지를 임계값 127로 이진화한다. 임계값이 넘으면 255로 값을 변경한다. 그리고 골격화 이미지를 저장할 skel을 만들어준다. 색상 이미지가 아니므로 채널은 1개로 만들고 원본 이미지와 크기를 같게 한다. opening 연산을 하고 outline의 돌출 부분을 subtract 한다. 그리고 or 논리연산으로 기존의 skel과 합친다. 다음으로는 침식 연산을 하고 이를 반복한다. 그러면 이미지 골격화가 이루어진다. 이 이미지를 가지고 knn_func()에 파라미터로 넣어서 학습을 시킨 후 테스트 과정을 거친다. 그 방법을 n2() 함수에서의 테스트 과정과 유사하다.
<br>

### edge detection
<img src="https://user-images.githubusercontent.com/59547069/103400770-affb0b00-4b89-11eb-9fd6-d992573ffaa1.png" width="600" height="700">

1. 비디오를 읽어들인다.
2. 비디오의 프레임을 얻어서 color bgr에서 gray 이미지로 변환하고 vgframe 변수에 저장한다.
3. 엣지를 검출하기 위해 Canny()를 이용한다. low_threshold는 100, high_threshold는 200을 이용한다.
4. 라인을 저장할 벡터 lines를 정의하고 허프 변환을 이용한다. HoughLinesP()를 이용하는데 line_threshold는 15, minLineLength는 10, maxLineGap은 20으로 맞춘다.
5. lines 벡터 사이즈만큼 반복하며 컬러 프레임에 라인을 그려준다. lines 벡터의 원소에 저장된 포인트 좌표를 이용해 line을  컬러 영상위에 그린다.
6. Canny()를 이용해서 엣지를 검출한 gray 이미지를 띄운다.
7. 허프 연산으로 라인을 찾아 그린 컬러 영상을 띄운다.
8. 비디오가 끝나면 종료시키기 위해 영상의 프레임이 empty()라면 Video END라는 문구를 출력하고 break해 반복문을 빠져나온다.
9. esc 키를 누르면 종료시키기 위해 누른 키가 27이라면 beak한다.
10. 반복문을 빠져나왔다면 동영상을 띄웠던 창을 destroyWindow(“edge Video"), destroyWindow(“line Video")로 없애준다.
