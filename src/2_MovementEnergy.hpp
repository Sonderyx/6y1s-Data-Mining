#ifndef MOVEMENT_ENERGY_H
#define MOVEMENT_ENERGY_H

#include <vector>
#include <opencv2\opencv.hpp>
#include <direct.h>
#include <io.h>
#include "logger.hpp"

using namespace cv;
using namespace std;

class Image {
public:
    string output_path = "output/Lab 2 Movement Energy/";

    Mat src;
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

/**
 * @brief Функция для морфологической обработки изображения
 *
 * Функция morphProc() предназначена для морфологической обработки
 * двоичного изображения. Она удаляет шумы, оставляя только общие
 * контуры объектов на изображении.
 *
 * @param src двоичное изображение, которое подвергается
 *            морфологической обработке
 * @param dst двоичное изображение, которое является результатом
 *            морфологической обработки
 */
void morphProc(const Mat& src, Mat& dst) {
    // Применяем морфологическую операцию "закрытие" для удаления шумов
    // размер ядра - 31 на 31, центр ядра - (16, 16)
    Mat element = cv::getStructuringElement(MORPH_RECT, Size(31, 31), Point(16, 16));
    morphologyEx(src, dst, MORPH_ERODE, element);
}

/**
 * @brief Calculate horizontal and vertical projections of a binary image
 *
 * @param [in] image Binary image
 * @param [out] horizontalProjection Vector of horizontal projection values
 * @param [out] verticalProjection Vector of vertical projection values
 */
void calculateProjections(const Mat& image, std::vector<int>& horizontalProjection, std::vector<int>& verticalProjection)
{
    // Инициализизация нулевого вектора проекций изображения с размером, соответсвующим высоте и ширине изображения
    horizontalProjection.resize(image.rows, 0);
    verticalProjection.resize(image.cols, 0);

    // Подсчет ненулевых пикселей по строкам и столбцам
    for (int row = 0; row < image.rows; ++row)
        for (int col = 0; col < image.cols; ++col)
            if (image.at<uchar>(row, col) > 0) {
                ++horizontalProjection[row];
                ++verticalProjection[col];
            }
}

/**
 * @brief Выделяет проекции по заданному порогу
 *
 * @param[in] proj Вектор проекции
 * @param[out] segments Вектор сегментов
 * @param[in] thresh Пороговое значение
 * @param[in] minLength Минимальная длина сегмента
 */
void findProjections(const vector<int>& proj, vector<pair<int, int>>& segments, int thresh, int minLength) {
    int start = -1; // Начало сегмента

    // Проходим по вектору проекции
    for (int i = 0; i < proj.size(); ++i) {
        // Если значение проекции больше порога, то
        if (proj[i] > thresh) {
            // Если старт не установлен, то
            if (start == -1)
                start = i; // Установка стартового значения
        }
        else
            // Если старт установлен, то
            if (start != -1) {
                // Если длина сегмента больше пороговой длинны, то
                if (i - start >= minLength)
                    segments.push_back({start, i - 1}); // Сохраняем сегмент
                start = -1; // Сброс
            }
    }
    // Проверка на конец массива
    if (start != -1 && proj.size() - start >= minLength)
        segments.push_back({start, proj.size() - 1}); // Сохраняем последний сегмент
}


/**
 * @brief Функция отображения гистограмм горизонтальной и вертикальной проекций
 *
 * Данная функция создает и отображает гистограммы горизонтальной и вертикальной
 * проекций двоичного изображения. Гистограммы нормализуются и отображаются на
 * белом фоне. На гистограммах также отображается уровень отсечки, заданный
 * пользователем.
 *
 * @param img Структура Image, содержащая параметры изображения
 * @param horizontalProj Вектор значений горизонтальной проекции
 * @param verticalProj Вектор значений вертикальной проекции
 */
void histShow(const Image& img, const vector<int>& horizontalProj, const vector<int>& verticalProj) {
    int hist_height = 200; // Высота гистограммы

    // Создание черных изображений для гистограмм
    Mat histImageH = Mat::zeros(hist_height, horizontalProj.size(), CV_8UC3);
    Mat histImageV = Mat::zeros(hist_height, verticalProj.size(), CV_8UC3);

    // Заливка изображений белым цветом
    histImageH.setTo(Scalar(255, 255, 255));
    histImageV.setTo(Scalar(255, 255, 255));

    // Нахождение максимальных значений проекций для нормализации
    int horProjMax = *max_element(horizontalProj.begin(), horizontalProj.end());
    int vertProjMax = *max_element(verticalProj.begin(), verticalProj.end());

    // Отрисовка гистограммы горизонтальной проекции, если максимум больше нуля
    if (horProjMax > 0)
        for (int i = 0; i < horizontalProj.size(); ++i)
            line(histImageH, Point(i, hist_height), Point(i, hist_height - (horizontalProj[i] * hist_height / horProjMax)), Scalar(0, 0, 0));

    // Отрисовка гистограммы вертикальной проекции, если максимум больше нуля
    if (vertProjMax > 0)
        for (int i = 0; i < verticalProj.size(); ++i)
            line(histImageV, Point(i, hist_height), Point(i, hist_height - (verticalProj[i] * hist_height / vertProjMax)), Scalar(0, 0, 0));

    // Определение и отрисовка уровня отсечки на гистограммах
    int thresholdYh = hist_height - (img.thresh * hist_height / horProjMax);
    int thresholdYv = hist_height - (img.thresh * hist_height / vertProjMax);

    line(histImageH, Point(0, thresholdYh), Point(horizontalProj.size(), thresholdYh), Scalar(255, 0, 0), 2);
    line(histImageV, Point(0, thresholdYv), Point(verticalProj.size(), thresholdYv), Scalar(255, 0, 0), 2);

    // Отображение гистограмм в отдельных окнах
    imshow("Horizontal Projection", histImageH);
    imwrite(img.output_path + "Horizontal Projection.png", histImageH);
    imshow("Vertical Projection", histImageV);
    imwrite(img.output_path + "Vertical Projection.png", histImageV);
}

/**
 * @brief Сегментация изображения на основе проекций
 *
 * Данная функция проводит сегментацию двоичного изображения на основе
 * горизонтальной и вертикальной проекций. Она находит сегменты на каждой
 * проекции, а затем выделяет прямоугольники, пересекающиеся сегменты
 * которых образуют итоговые сегменты изображения.
 *
 * @param img Структура Image, содержащая параметры изображения
 *
 * @return Вектор прямоугольников, образующих сегменты изображения
 */
vector<Rect> segmentImage(const Image& img) {
    // Векторы для хранения проекций
    // vector<int> horizontalProj, verticalProj;
    vector<int> horizontalProj = img.horizontalProj;
    vector<int> verticalProj = img.verticalProj;

    calculateProjections(img.morph, horizontalProj, verticalProj);
    // logger.info("Horizontal projection: {}", fmt::format("[{}]", fmt::join(horizontalProj, ", ")));
    // logger.info("Vertical projection: {}", fmt::format("[{}]", fmt::join(verticalProj, ", ")));

    // Векторы для хранения сегментов
    vector<pair<int, int>> horizontalSegments, verticalSegments;

    // находим сегменты по горизонтальной и вертикальной проекциям
    findProjections(horizontalProj, horizontalSegments, img.thresh, img.minLength);
    findProjections(verticalProj, verticalSegments, img.thresh, img.minLength);

    // отображение гистограмм
    histShow(img, horizontalProj, verticalProj);

    // Вектор для хранения прямоугольников
    vector<Rect> rectangles;

    // выделяем прямоугольники на основе проекций
    for (const auto& hSeg : horizontalSegments)
        for (const auto& vSeg : verticalSegments)
            rectangles.push_back(Rect(Point(vSeg.first, hSeg.first), Point(vSeg.second, hSeg.second)));

    return rectangles; // возвращаем список прямоугольников
}

/**
 * @brief Обновляет сегментированное изображение
 *
 * Функция updateSegmentedImage() создает сегментированное изображение на основе входных данных,
 * рисуя прямоугольники вокруг обнаруженных сегментов.
 *
 * @param img Структура Image, содержащая параметры изображения
 */
void updateSegmentedImage(const Image& img) {
    // Клонируем второе изображение для последующего редактирования
    Mat result = img.bgr2.clone();

    // Получаем список прямоугольников, образующих сегменты изображения
    vector<Rect> rectangles = segmentImage(img);

    // Рисуем прямоугольники вокруг сегментов
    for (const auto& rect : rectangles)
        rectangle(result, rect, Scalar(255, 0, 0), 2);

    // Отображаем сегментированное изображение
    imshow("Segmented Image", result);
    imwrite(img.output_path + "Segmented Image.png", result);
}

void onMinLengthChange(int, void* userdata) {
    Image* img = static_cast<Image*>(userdata);
    updateSegmentedImage(*img);
}

void onThresholdChange(int, void* userdata) {
    Image* img = static_cast<Image*>(userdata);
    updateSegmentedImage(*img);
}

/**
 * @brief Функция для детекции движения на двух изображениях
 *
 * Функция lab2_MovementEnergy() предназначена для детекции движения на двух
 * изображениях. Она вычитает из второго изображения первое, получая изображение
 * разности. Затем она получает контурное препарата с помощью алгоритма Собеля,
 * применяет логическое И к изображению разности и контурному препарату,
 * проводит морфологическую обработку с помощью операции "закрытие" и
 * инвертирует полученное изображение.
 *
 * @param img_bgr1 первое изображение в формате BGR
 * @param img_bgr2 второе изображение в формате BGR
 */
void lab2_MovementEnergy(Mat& img_bgr1, Mat& img_bgr2) {
    logger.info("Lab 2: Movement Energy");
    Image img;

    // создаём папку для выходных изображений
    if (_access(img.output_path.c_str(), 0) != 0) {
        if (_mkdir(img.output_path.c_str()) == -1) {
            logger.error("Failed to create directory: {}", img.output_path);
            return;
        }
        logger.info("Created directory: {}", img.output_path);
    }

    //     // создаём папку для выходных изображений
    // if (_access(image_path.c_str(), 0) != 0) {
    //     if (_mkdir(image_path.c_str()) == -1) {
    //         logger.error("Failed to create directory: {}", image_path);
    //         return;
    //     }
    //     logger.info("Created directory: {}", image_path);
    // }

    img.bgr1 = img_bgr1;
    img.bgr2 = img_bgr2;

    // конвертируем изображения в градации серого
    cvtColor(img_bgr1, img.gray1, COLOR_BGR2GRAY);
    cvtColor(img_bgr2, img.gray2, COLOR_BGR2GRAY);

    // отображаем изображения
    imshow("img bgr 1", img.bgr1);
    imshow("img bgr 2", img.bgr2);

    // вычитаем из второго изображения первое, получая изображение разности
    absdiff(img.gray1, img.gray2, img.diff);
    imshow("absdiff", img.diff);
    imwrite(img.output_path + "absdiff.png", img.diff);

    // оператор Собеля
    // Mat grad_x, grad_y;
    // Mat abs_grad_x, abs_grad_y;
    // int scale = 1;
    // int delta = 0;
    // int ddepth = CV_16S;
    // Sobel(img.gray2, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    // Sobel(img.gray2, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
    // convertScaleAbs(grad_x, abs_grad_x);
    // convertScaleAbs(grad_y, abs_grad_y);
    // addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, img.edges);

    // оператор Кэнни
    Canny(img.gray2, img.edges, 50, 200);

    // отображаем контурное препарата
    imshow("edges", img.edges);
    imwrite(img.output_path + "edges.png", img.edges);

    // применяем логическое И
    bitwise_and(img.diff, img.edges, img.diffAndEdges);
    imshow("absdiff AND edges", img.diffAndEdges);
    imwrite(img.output_path + "absdiff AND edges.png", img.diffAndEdges);

    // задаем пороговое значение
    threshold(img.diffAndEdges, img.diffAndEdges, 5, 255, THRESH_BINARY_INV);

    // проводим морфологическую обработку
    morphProc(img.diffAndEdges, img.morph);
    imshow("absdiff AND edges eroded", img.morph);
    imwrite(img.output_path + "absdiff AND edges eroded.png", img.morph);

    // инвертируем изображение
    bitwise_not(img.morph, img.morph);
    imshow("absdiff AND edges eroded and inverted", img.morph);
    imwrite(img.output_path + "absdiff AND edges eroded and inverted.png", img.morph);

    // создаем окна
    namedWindow("Segmented Image", WINDOW_AUTOSIZE);

    // создаем ползунки
    createTrackbar("Min Length", "Segmented Image", &img.minLength, 500, onMinLengthChange, &img);
    createTrackbar("Threshold", "Segmented Image", &img.thresh, 500, onThresholdChange, &img);

    waitKey(0);
}

#endif MOVEMENT_ENERGY_H