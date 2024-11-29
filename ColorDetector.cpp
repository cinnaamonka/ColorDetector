#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

int main() {
    // 1. Указание пути к изображению
    // Используйте двойные обратные слэши или прямые слэши
    string imagePath = "C:\\Users\\parni\\Downloads\\58f8bdcd0ed2bdaf7c12830d.png";
    // Альтернативные варианты:
    // string imagePath = "C:/Users/parni/Downloads/58f8bdcd0ed2bdaf7c12830d.png";
    // string imagePath = R"(C:\Users\parni\Downloads\58f8bdcd0ed2bdaf7c12830d.png)";

    // 2. Загрузка изображения с альфа-каналом
    Mat img = imread(imagePath, IMREAD_UNCHANGED);
    if (img.empty()) {
        cout << "Не удалось загрузить изображение по пути: " << imagePath << endl;
        return -1;
    }

    // 3. Разделение каналов (если есть альфа-канал)
    Mat bgr, alpha;
    if (img.channels() == 4) {
        vector<Mat> channels;
        split(img, channels);
        // Объединяем первые три канала в BGR
        merge(vector<Mat>{channels[0], channels[1], channels[2]}, bgr);
        // Альфа-канал
        alpha = channels[3];
    }
    else {
        // Если альфа-канала нет, считаем всё изображение непрозрачным
        bgr = img.clone();
        alpha = Mat::ones(img.size(), CV_8UC1) * 255;
    }

    // 4. Применение маски альфа-канала
    Mat mask;
    threshold(alpha, mask, 0, 255, THRESH_BINARY);
    // Опционально: можно применить морфологические операции для очистки маски
    // Например, удалить мелкий шум и заполнить небольшие отверстия
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);

    // 5. Применение GrabCut для улучшенной сегментации
    // Создание начальной маски для GrabCut
    Mat grabcutMask;
    grabcutMask = Mat::zeros(img.size(), CV_8UC1);
    // Используем уже существующую маску альфа-канала как основу
    mask.copyTo(grabcutMask);

    // Настройка прямоугольника для GrabCut (можно настроить или автоматизировать)
    // Здесь предполагается, что объект занимает центральную часть изображения
    Rect rect(10, 10, img.cols - 20, img.rows - 20);

    // Применение GrabCut
    grabCut(bgr, grabcutMask, rect, Mat(), Mat(), 5, GC_INIT_WITH_RECT);

    // Создание финальной маски, где вероятные и точные объекты
    Mat finalMask;
    compare(grabcutMask, GC_PR_FGD, finalMask, CMP_EQ);
    // Можно объединить с исходной маской альфа-канала, чтобы игнорировать прозрачные области
    bitwise_and(finalMask, mask, finalMask);

    // Извлечение объекта из изображения
    Mat result;
    bgr.copyTo(result, finalMask);

    // Проверка, что сегментация дала результат
    if (result.empty() || countNonZero(finalMask) == 0) {
        cout << "Ошибка: Сегментированный объект пуст или маска не содержит пикселей!" << endl;
        return -1;
    }

    // 6. Определение основного цвета
    // Преобразование изображения в цветовое пространство Lab для лучшей кластеризации
    Mat lab;
    cvtColor(result, lab, COLOR_BGR2Lab);
    // Переформатирование в одномерный массив пикселей
    lab = lab.reshape(1, lab.rows * lab.cols);

    // Преобразование типа данных в CV_32F
    lab.convertTo(lab, CV_32F);

    // Проверка на пустоту и достаточное количество данных
    if (lab.empty()) {
        cout << "Ошибка: Матрица Lab пуста после сегментации!" << endl;
        return -1;
    }

    int K = 3; // Количество кластеров
    if (lab.rows < K) {
        cout << "Ошибка: Недостаточно данных для кластеризации с K = " << K << endl;
        return -1;
    }

    // Кластеризация с использованием K-средних
    Mat labelsMat;
    int attempts = 5;
    Mat centers;
    kmeans(lab, K, labelsMat, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),
        attempts, KMEANS_PP_CENTERS, centers);

    // Подсчёт количества пикселей в каждом кластере
    vector<int> counts(K, 0);
    for (int i = 0; i < labelsMat.rows; i++) {
        counts[labelsMat.at<int>(i)]++;
    }

    // Нахождение кластера с наибольшим количеством пикселей
    int maxIdx = distance(counts.begin(), max_element(counts.begin(), counts.end()));
    Vec3f dominantColorLab = centers.at<Vec3f>(maxIdx);

    // Преобразование обратно в BGR
    Mat dominantColorMat(1, 1, CV_32FC3, dominantColorLab);
    dominantColorMat.convertTo(dominantColorMat, CV_8UC3);
    cvtColor(dominantColorMat, dominantColorMat, COLOR_Lab2BGR);
    Vec3b dominantColorBGR = dominantColorMat.at<Vec3b>(0, 0);

    cout << "Основной цвет объекта (BGR): "
        << (int)dominantColorBGR[0] << ", "
        << (int)dominantColorBGR[1] << ", "
        << (int)dominantColorBGR[2] << endl;

    // 7. Визуализация основного цвета
    // Создание изображения с основным цветом
    Mat colorSwatch(100, 100, CV_8UC3, dominantColorBGR);

    // Отображение результатов
    imshow("Исходное изображение", img);
    imshow("Сегментированный объект", result);
    imshow("Основной цвет", colorSwatch);
    waitKey(0);

    return 0;
}
