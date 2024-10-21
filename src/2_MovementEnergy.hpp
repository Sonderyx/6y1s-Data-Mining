#ifndef MOVEMENT_ENERGY_H
#define MOVEMENT_ENERGY_H

#include <vector>
#include <opencv2\opencv.hpp>
#include "logger.hpp"

using namespace cv;
using namespace std;

class Image {
public:
    Mat bgr1;
    Mat bgr2;

    Mat gray1;
    Mat gray2;

    Mat diff;
    Mat diffAndEdges;
    Mat edges;
    Mat morph;

    vector<int> horizontalProj;
    vector<int> verticalProj;

    Mat hist_hor;
    Mat hist_ver;
    Mat hist_dia;

    int minLength = 100;    // длина сегмента
    int thresh = 100;       // пороговое значение для определения сегментов
};

// Функция для морфологической обработки изображения
void morphProc(const Mat& src, Mat& dst) {
    // Применяем морфологическую операцию "закрытие" для удаления шумов
    Mat element = cv::getStructuringElement(MORPH_RECT, Size(31, 31), Point(16, 16));
    morphologyEx(src, dst, MORPH_ERODE, element);

    // morphologyEx(src, dst, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(5, 5)));
}

// Функция для подсчета ненулевых пикселей в строках и столбцах
void calculateProjections(const Mat& img_bin, vector<int>& horizontalProj, vector<int>& verticalProj) {
    int rows = img_bin.rows;
    int cols = img_bin.cols;

    horizontalProj.resize(rows, 0);
    verticalProj.resize(cols, 0);

    // Подсчет ненулевых пикселей по строкам и столбцам
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            if (img_bin.at<uchar>(i, j) > 0) {
                horizontalProj[i]++;
                verticalProj[j]++;
            }
}

// Функция для выделения проекций по заданному порогу
void findProjections(const vector<int>& proj, vector<pair<int, int>>& segments, int thresh, int minLength) {
    int start = -1;
    for (int i = 0; i < proj.size(); ++i) {
        if (proj[i] > thresh) {
            if (start == -1)
                start = i; // Начало сегмента
        }
        else
            if (start != -1) {
                if (i - start >= minLength)
                    segments.push_back({start, i - 1}); // Сохраняем сегмент
                start = -1; // Сброс
            }
    }
    // Проверка на конец массива
    if (start != -1 && proj.size() - start >= minLength)
        segments.push_back({start, proj.size() - 1});
}

void histShow(const Image& img, const vector<int>& horizontalProj, const vector<int>& verticalProj) {
    int hist_height = 200;

    // Создаем черные изображения для гистограмм
    Mat histImageH = Mat::zeros(hist_height, horizontalProj.size(), CV_8UC3);
    Mat histImageV = Mat::zeros(hist_height, verticalProj.size(), CV_8UC3);

    // Заполняем белым цветом
    histImageH.setTo(Scalar(255, 255, 255));
    histImageV.setTo(Scalar(255, 255, 255));

    // Нормализация и отрисовка гистограмм
    int horProjMax = *max_element(horizontalProj.begin(), horizontalProj.end());
    int vertProjMax = *max_element(verticalProj.begin(), verticalProj.end());

    if (horProjMax > 0)
        for (int i = 0; i < horizontalProj.size(); ++i)
            line(histImageH, Point(i, hist_height), Point(i, hist_height - (horizontalProj[i] * hist_height / horProjMax)), Scalar(0, 0, 0));

    if (vertProjMax > 0)
        for (int i = 0; i < verticalProj.size(); ++i)
            line(histImageV, Point(i, hist_height), Point(i, hist_height - (verticalProj[i] * hist_height / vertProjMax)), Scalar(0, 0, 0));

    // Рисуем уровень отсечки на гистограммах
    int thresholdYh = hist_height - (img.thresh * hist_height / horProjMax);
    int thresholdYv = hist_height - (img.thresh * hist_height / vertProjMax);

    line(histImageH, Point(0, thresholdYh), Point(horizontalProj.size(), thresholdYh), Scalar(255, 0, 0), 2);
    line(histImageV, Point(0, thresholdYv), Point(verticalProj.size(), thresholdYv), Scalar(255, 0, 0), 2);

    imshow("Horizontal Projection", histImageH);
    imshow("Vertical Projection", histImageV);
}

// Сегментация изображения на основе проекций
vector<Rect> segmentImage(const Image& img) {
    // Векторы для хранения проекций
    vector<int> horizontalProj, verticalProj;
    calculateProjections(img.morph, horizontalProj, verticalProj);
    // logger.info("Horizontal projection: {}", fmt::format("[{}]", fmt::join(horizontalProj, ", ")));
    // logger.info("Vertical projection: {}", fmt::format("[{}]", fmt::join(verticalProj, ", ")));


    // Векторы для хранения сегментов
    vector<pair<int, int>> horizontalSegments, verticalSegments;

    // находим сегменты по горизонтальной и вертикальной проекциям
    findProjections(horizontalProj, horizontalSegments, img.thresh, img.minLength);
    findProjections(verticalProj, verticalSegments, img.thresh, img.minLength);

    // int max1 = max_element(horizontalProj.begin(), horizontalProj.end());
    // отображение гистограмм
    // histShow(img, horizontalProj, verticalProj, , verticalProj.size());
    histShow(img, horizontalProj, verticalProj);

    // Вектор для хранения прямоугольников
    vector<Rect> rectangles;

    // выделяем прямоугольники на основе проекций
    for (const auto& hSeg : horizontalSegments)
        for (const auto& vSeg : verticalSegments)
            rectangles.push_back(Rect(Point(vSeg.first, hSeg.first), Point(vSeg.second, hSeg.second)));

    return rectangles; // возвращаем список прямоугольников
}

void updateSegmentedImage(const Image& img) {
    Mat result = img.bgr2.clone();

    vector<Rect> rectangles = segmentImage(img);

    for (const auto& rect : rectangles)
        rectangle(result, rect, Scalar(255, 0, 0), 2);

    imshow("Segmented Image", result);
}

void onMinLengthChange(int, void* userdata) {
    Image* image = static_cast<Image*>(userdata);
    updateSegmentedImage(*image);
}

void onThresholdChange(int, void* userdata) {
    Image* image = static_cast<Image*>(userdata);
    updateSegmentedImage(*image);
}

void lab2_MovementEnergy(Mat& img_bgr1, Mat& img_bgr2) {
    Image img;
    img.bgr1 = img_bgr1;
    img.bgr2 = img_bgr2;

    cvtColor(img_bgr1, img.gray1, COLOR_BGR2GRAY);
    cvtColor(img_bgr2, img.gray2, COLOR_BGR2GRAY);

    imshow("img bgr 1", img.bgr1);
    imshow("img bgr 2", img.bgr2);

    absdiff(img.gray1, img.gray2, img.diff);
    imshow("absdiff", img.diff);

    // // Получение контурного препарата с помощью алгоритма Собеля
    // Mat grad_x, grad_y;
    // Mat abs_grad_x, abs_grad_y;
    // int scale = 1;
    // int delta = 0;
    // int ddepth = CV_16S;

    // Sobel(img_bgr1_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    // Sobel(img_bgr1_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);

    // convertScaleAbs(grad_x, abs_grad_x);
    // convertScaleAbs(grad_y, abs_grad_y);

    // addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, edges);
    // imshow("edges", edges);

    Canny(img.gray2, img.edges, 50, 200);
    imshow("edges", img.edges);

    //применение логического И
    bitwise_and(img.diff, img.edges, img.diffAndEdges);
    imshow("absdiff AND edges", img.diffAndEdges);

    threshold(img.diffAndEdges, img.diffAndEdges, 5, 255, THRESH_BINARY_INV);

    //морфологическая обработка
    morphProc(img.diffAndEdges, img.morph);

    //инвертирование изображения
    bitwise_not(img.morph, img.morph);
    imshow("absdiff AND edges eroded", img.morph);

    // Создаем окна
    namedWindow("Segmented Image", WINDOW_AUTOSIZE);

    // Создаем ползунки
    createTrackbar("Min Length", "Segmented Image", &img.minLength, 500, onMinLengthChange, &img);
    createTrackbar("Threshold", "Segmented Image", &img.thresh, 500, onThresholdChange, &img);

    waitKey(0);
}

#endif MOVEMENT_ENERGY_H