#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <gsl/gsl_fit.h>
#include <iostream>

using namespace cv;
using namespace std;

//Hough Transform 파라미터
float rho = 2; // 해상도
float theta = 1 * CV_PI / 180; // 각도
float hough_threshold = 15;
float minLineLength = 10; // 검출할 선의 최소 길이
float maxLineGap = 20; // 선 위의 점들 사이 최대 거리

// 관심영역(사다리꼴)
float trap_bottom_width = 0.85;
float trap_top_width = 0.07;
float trap_height = 0.4;

// 흰색 선(RGB)
Scalar lower_white = Scalar(150, 150, 150);
Scalar upper_white = Scalar(255, 255, 255);

//파란색 선(HSV)
Scalar lower_blue = Scalar(10, 100, 100);
Scalar upper_blue = Scalar(260, 255, 255);

Mat region_limit(Mat img_edges, Point* points)
{
    Mat img_mask = Mat::zeros(img_edges.rows, img_edges.cols, CV_8UC1);

    Scalar ignore_mask_color = Scalar(255, 255, 255);
    const Point* ppt[1] = { points };
    int npt[] = { 4 };

    //다각형 채우기
    fillPoly(img_mask, ppt, npt, 1, Scalar(255, 255, 255), LINE_8);

    //마스크 픽셀이 0이 아닌 경우에만 이미지 반환
    Mat img_masked;
    bitwise_and(img_edges, img_mask, img_masked);

    return img_masked;
}

//흰색, 파란색 픽셀만 포함하도록 필터링
void filter_colors(Mat _img_bgr, Mat& img_filtered)
{
    UMat img_bgr;
    _img_bgr.copyTo(img_bgr);
    UMat img_hsv, img_combine;
    UMat white_mask, white_image;
    UMat blue_mask, blue_image;

    //흰색 픽셀 필터링
    inRange(img_bgr, lower_white, upper_white, white_mask);
    bitwise_and(img_bgr, img_bgr, white_image, white_mask);

    //파란색 필터 필터링
    cvtColor(img_bgr, img_hsv, COLOR_BGR2HSV);

    inRange(img_hsv, lower_blue, upper_blue, blue_mask);
    bitwise_and(img_bgr, img_bgr, blue_image, blue_mask);

    //이미지 합치기
    addWeighted(white_image, 1.0, blue_image, 1.0, 0.0, img_combine);
    img_combine.copyTo(img_filtered);
}

void draw_line(Mat& img_line, vector<Vec4i> lines)
{
    if (lines.size() == 0) return;

    bool draw_right = true;
    bool draw_left = true;
    int width = img_line.cols;
    int height = img_line.rows;

    //기울기가 너무 수평인 선은 제외
    float slope_threshold = 0.5;
    vector<float> slopes;
    vector<Vec4i> new_lines;

    for (int i = 0; i < lines.size(); i++)
    {
        Vec4i line = lines[i];
        int x1 = line[0];
        int y1 = line[1];
        int x2 = line[2];
        int y2 = line[3];

        float slope;
        //기울기 계산
        if (x2 - x1 == 0) // 코너인 경우
            slope = 999.0;
        else
            slope = (y2 - y1) / (float)(x2 - x1);


        if (abs(slope) > slope_threshold) {
            slopes.push_back(slope);
            new_lines.push_back(line);
        }
    }

    //왼쪽 선과 오르쪽 선으로 분류
    vector<Vec4i> right_lines;
    vector<Vec4i> left_lines;

    for (int i = 0; i < new_lines.size(); i++)
    {
        Vec4i line = new_lines[i];
        float slope = slopes[i];

        int x1 = line[0];
        int y1 = line[1];
        int x2 = line[2];
        int y2 = line[3];

        float cx = width * 0.5; // 이미지 중심의 x좌표

        if (slope > 0 && x1 > cx && x2 > cx)
            right_lines.push_back(line);
        else if (slope < 0 && x1 < cx && x2 < cx)
            left_lines.push_back(line);
    }

    //선형 회귀로 가장 적합한 선 찾기
    double right_lines_x[1000];
    double right_lines_y[1000];
    float right_m, right_b;

    int right_index = 0;
    for (int i = 0; i < right_lines.size(); i++) {

        Vec4i line = right_lines[i];
        int x1 = line[0];
        int y1 = line[1];
        int x2 = line[2];
        int y2 = line[3];

        right_lines_x[right_index] = x1;
        right_lines_y[right_index] = y1;
        right_index++;
        right_lines_x[right_index] = x2;
        right_lines_y[right_index] = y2;
        right_index++;
    }

    if (right_index > 0) {
        double c0, c1, cov00, cov01, cov11, sumsq;
        gsl_fit_linear(right_lines_x, 1, right_lines_y, 1, right_index,
            &c0, &c1, &cov00, &cov01, &cov11, &sumsq);

        right_m = c1;
        right_b = c0;
    }
    else {
        right_m = right_b = 1;
        draw_left = false;
    }

    double left_lines_x[1000];
    double left_lines_y[1000];
    float left_m, left_b;
    int left_index = 0;
    for (int i = 0; i < left_lines.size(); i++) {

        Vec4i line = left_lines[i];
        int x1 = line[0];
        int y1 = line[1];
        int x2 = line[2];
        int y2 = line[3];

        left_lines_x[left_index] = x1;
        left_lines_y[left_index] = y1;
        left_index++;
        left_lines_x[left_index] = x2;
        left_lines_y[left_index] = y2;
        left_index++;
    }


    if (left_index > 0) {
        double c0 = 0, c1 = 0, cov00, cov01, cov11, sumsq;
        gsl_fit_linear(left_lines_x, 1, left_lines_y, 1, left_index,
            &c0, &c1, &cov00, &cov01, &cov11, &sumsq);

        left_m = c1;
        left_b = c0;
    }
    else {
        left_m = left_b = 1;
        draw_left = false;
    }

    //좌우 선의 끝점 계산 (y = m*x + b --> x = (y - b) / m)
    int y1 = height;
    int y2 = height * (1 - trap_height);

    float right_x1 = (y1 - right_b) / right_m;
    float right_x2 = (y2 - right_b) / right_m;

    float left_x1 = (y1 - left_b) / left_m;
    float left_x2 = (y2 - left_b) / left_m;

    //float->int
    y1 = int(y1);
    y2 = int(y2);
    right_x1 = int(right_x1);
    right_x2 = int(right_x2);
    left_x1 = int(left_x1);
    left_x2 = int(left_x2);

    //좌우 선 그리기

    int left_offset = abs(width / 2 - (left_x1 + left_x2) / 2);
    int right_offset = abs((right_x1 + right_x2) / 2 - width / 2);

    if (draw_right != false && draw_left != false) {
        if (draw_right)
            line(img_line, Point(right_x1, y1), Point(right_x2, y2), Scalar(0, 255, 0), 10);
        if (draw_left)
            line(img_line, Point(left_x1, y1), Point(left_x2, y2), Scalar(0, 255, 0), 10);
        if (left_offset <= 200 || right_offset <= 100)
            putText(img_line, "Changing lane", Point(400, 500), FONT_HERSHEY_SIMPLEX, 5, Scalar(0, 0, 255), 10, LINE_AA);
    }
    else
        putText(img_line, "Cannot detect", Point(400, 500), FONT_HERSHEY_SIMPLEX, 5, Scalar(255, 255, 255), 10, LINE_AA);

}

int main(int, char** argv)
{
    char buf[256];
    Mat img_bgr, img_gray, img_edges, img_hough, img_annotated;

    VideoCapture videoCapture("C:/Users/user/Desktop/Computer Vision_LaneDetection/clip2.mp4");

    if (!videoCapture.isOpened())
    {
        cout << "Cannot open the video file.\n" << endl;
        char a;
        cin >> a;
        return 1;
    }

    videoCapture.read(img_bgr);
    if (img_bgr.empty()) return -1;

    VideoWriter writer;
    int codec = VideoWriter::fourcc('X', 'V', 'I', 'D'); // 코덱 선택
    double fps = 25.0; // 프레임 속도
    string filename = "./live.mp4";
    writer.open(filename, codec, fps, img_bgr.size(), CV_8UC3);


    if (!writer.isOpened()) {
        cerr << "Could not open the output video file for write\n";
        return -1;
    }

    videoCapture.read(img_bgr);
    int width = img_bgr.size().width;
    int height = img_bgr.size().height;

    int count = 0;

    while (1)
    {
        // 1. 동영상 읽어오기
        videoCapture.read(img_bgr);
        if (img_bgr.empty())break;

        // 2. 미리 정해둔 흰색, 파란색 범위 내에 있는 부분만 차선후보로 따로 저장함
        Mat img_filtered;
        filter_colors(img_bgr, img_filtered);

        // 3. 그레이스케일 영상으로 변환하여 에지 성분 추출
        cvtColor(img_filtered, img_gray, COLOR_BGR2GRAY);
        GaussianBlur(img_gray, img_gray, Size(3, 3), 0, 0);
        Canny(img_gray, img_edges, 50, 150);

        int width = img_filtered.cols;
        int height = img_filtered.rows;

        Point points[4];
        points[0] = Point((width * (1 - trap_bottom_width)) / 2, height);
        points[1] = Point((width * (1 - trap_top_width)) / 2, height - height * trap_height);
        points[2] = Point(width - (width * (1 - trap_top_width)) / 2, height - height * trap_height);
        points[3] = Point(width - (width * (1 - trap_bottom_width)) / 2, height);

        //.4 차선 검출할 영역을 제한함(진행방향 바닥에 존재하는 차선으로 한정)
        img_edges = region_limit(img_edges, points);

        UMat uImage_edges;
        img_edges.copyTo(uImage_edges);

        //5. 직선 성분을 추출(각 직선의 시작좌표와 끝좌표를 계산함)
        vector<Vec4i> lines;
        HoughLinesP(uImage_edges, lines, rho, theta, hough_threshold, minLineLength, maxLineGap);

        //6. 5번에서 추출한 직선 성분으로부터 좌우 차선에 있을 가능성이 있는 직선들만 따로 뽑아서
        // 좌우 각각 하나씩 직선을 계산함 (Linear Least-Squares Fitting)
        Mat img_line = Mat::zeros(img_bgr.rows, img_bgr.cols, CV_8UC3);
        draw_line(img_line, lines);

        //7. 원본 영상에 6번의 직선을 같이 보여줌
        addWeighted(img_bgr, 0.8, img_line, 1.0, 0.0, img_annotated);

        //8. 결과를 동영상 파일로 기록
        writer << img_annotated;

        count++;
        if (count == 10) imwrite("img_annotated.jpg", img_annotated);

        //9. 결과를 화면에 보여줌
        Mat img_result;
        resize(img_annotated, img_annotated, Size(width * 0.4, height * 0.4));
        resize(img_edges, img_edges, Size(width * 0.4, height * 0.4));
        cvtColor(img_edges, img_edges, COLOR_GRAY2BGR);
        hconcat(img_edges, img_annotated, img_result);
        imshow("lane detection video", img_result);

        if (waitKey(1) == 27) break; //esc키 누르면 종료
    }

    videoCapture.release();
    return 0;
}