#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

int main() {
    // 1. �������� ���� � �����������
    // ����������� ������� �������� ����� ��� ������ �����
    string imagePath = "C:\\Users\\parni\\Downloads\\58f8bdcd0ed2bdaf7c12830d.png";
    // �������������� ��������:
    // string imagePath = "C:/Users/parni/Downloads/58f8bdcd0ed2bdaf7c12830d.png";
    // string imagePath = R"(C:\Users\parni\Downloads\58f8bdcd0ed2bdaf7c12830d.png)";

    // 2. �������� ����������� � �����-�������
    Mat img = imread(imagePath, IMREAD_UNCHANGED);
    if (img.empty()) {
        cout << "�� ������� ��������� ����������� �� ����: " << imagePath << endl;
        return -1;
    }

    // 3. ���������� ������� (���� ���� �����-�����)
    Mat bgr, alpha;
    if (img.channels() == 4) {
        vector<Mat> channels;
        split(img, channels);
        // ���������� ������ ��� ������ � BGR
        merge(vector<Mat>{channels[0], channels[1], channels[2]}, bgr);
        // �����-�����
        alpha = channels[3];
    }
    else {
        // ���� �����-������ ���, ������� �� ����������� ������������
        bgr = img.clone();
        alpha = Mat::ones(img.size(), CV_8UC1) * 255;
    }

    // 4. ���������� ����� �����-������
    Mat mask;
    threshold(alpha, mask, 0, 255, THRESH_BINARY);
    // �����������: ����� ��������� ��������������� �������� ��� ������� �����
    // ��������, ������� ������ ��� � ��������� ��������� ���������
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);

    // 5. ���������� GrabCut ��� ���������� �����������
    // �������� ��������� ����� ��� GrabCut
    Mat grabcutMask;
    grabcutMask = Mat::zeros(img.size(), CV_8UC1);
    // ���������� ��� ������������ ����� �����-������ ��� ������
    mask.copyTo(grabcutMask);

    // ��������� �������������� ��� GrabCut (����� ��������� ��� ����������������)
    // ����� ��������������, ��� ������ �������� ����������� ����� �����������
    Rect rect(10, 10, img.cols - 20, img.rows - 20);

    // ���������� GrabCut
    grabCut(bgr, grabcutMask, rect, Mat(), Mat(), 5, GC_INIT_WITH_RECT);

    // �������� ��������� �����, ��� ��������� � ������ �������
    Mat finalMask;
    compare(grabcutMask, GC_PR_FGD, finalMask, CMP_EQ);
    // ����� ���������� � �������� ������ �����-������, ����� ������������ ���������� �������
    bitwise_and(finalMask, mask, finalMask);

    // ���������� ������� �� �����������
    Mat result;
    bgr.copyTo(result, finalMask);

    // ��������, ��� ����������� ���� ���������
    if (result.empty() || countNonZero(finalMask) == 0) {
        cout << "������: ���������������� ������ ���� ��� ����� �� �������� ��������!" << endl;
        return -1;
    }

    // 6. ����������� ��������� �����
    // �������������� ����������� � �������� ������������ Lab ��� ������ �������������
    Mat lab;
    cvtColor(result, lab, COLOR_BGR2Lab);
    // ������������������ � ���������� ������ ��������
    lab = lab.reshape(1, lab.rows * lab.cols);

    // �������������� ���� ������ � CV_32F
    lab.convertTo(lab, CV_32F);

    // �������� �� ������� � ����������� ���������� ������
    if (lab.empty()) {
        cout << "������: ������� Lab ����� ����� �����������!" << endl;
        return -1;
    }

    int K = 3; // ���������� ���������
    if (lab.rows < K) {
        cout << "������: ������������ ������ ��� ������������� � K = " << K << endl;
        return -1;
    }

    // ������������� � �������������� K-�������
    Mat labelsMat;
    int attempts = 5;
    Mat centers;
    kmeans(lab, K, labelsMat, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),
        attempts, KMEANS_PP_CENTERS, centers);

    // ������� ���������� �������� � ������ ��������
    vector<int> counts(K, 0);
    for (int i = 0; i < labelsMat.rows; i++) {
        counts[labelsMat.at<int>(i)]++;
    }

    // ���������� �������� � ���������� ����������� ��������
    int maxIdx = distance(counts.begin(), max_element(counts.begin(), counts.end()));
    Vec3f dominantColorLab = centers.at<Vec3f>(maxIdx);

    // �������������� ������� � BGR
    Mat dominantColorMat(1, 1, CV_32FC3, dominantColorLab);
    dominantColorMat.convertTo(dominantColorMat, CV_8UC3);
    cvtColor(dominantColorMat, dominantColorMat, COLOR_Lab2BGR);
    Vec3b dominantColorBGR = dominantColorMat.at<Vec3b>(0, 0);

    cout << "�������� ���� ������� (BGR): "
        << (int)dominantColorBGR[0] << ", "
        << (int)dominantColorBGR[1] << ", "
        << (int)dominantColorBGR[2] << endl;

    // 7. ������������ ��������� �����
    // �������� ����������� � �������� ������
    Mat colorSwatch(100, 100, CV_8UC3, dominantColorBGR);

    // ����������� �����������
    imshow("�������� �����������", img);
    imshow("���������������� ������", result);
    imshow("�������� ����", colorSwatch);
    waitKey(0);

    return 0;
}
