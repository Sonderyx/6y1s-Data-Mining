#ifndef PANORAMA_H
#define PANORAMA_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <io.h>
#include <direct.h>
#include "logger.hpp"

using namespace cv;
using namespace std;

string output_path6 = "output/Lab 6 Panorama/";

/**
 * @brief Функция загрузки изображений
 *
 * Функция cutImg() загружает два изображения из входного изображения src.
 * Она возвращает два изображения img1 и img2, которые
 * соответствуют левой и правой частям входного изображения.
 *
 * @param[in] src - исходное изображение
 * @param[out] img1 - левая часть изображения
 * @param[out] img2 - правая часть изображения
 */
void cutImg(const Mat& src, Mat& img1, Mat& img2) {
    // выделяем левую часть изображения
    // Range - диапазон строк, Range - диапазон столбцов
    img1 = src(Range(0, 500), Range(0, 550));

    // выделяем правую часть изображения
    img2 = src(Range(0, 400), Range(300, 750));
}

/**
 * @brief Функция обнаружения и вычисления дескрипторов признаков
 *
 * Функция detectAndComputeFeatures() обнаруживает и вычисляет
 * дескрипторы признаков для входного изображения img.
 * Она использует алгоритм ORB.
 *
 * @param[in] img - входное изображение
 * @param[out] keypoints - вектор ключевых точек
 * @param[out] descriptors - матрица дескрипторов
 */
void detectAndComputeFeatures(const Mat& img, std::vector<KeyPoint>& keypoints, Mat& descriptors) {
    // создаем объект детектора ORB
    Ptr<ORB> detector = ORB::create();

    // обнаруживаем ключевые точки на изображении
    // и вычисляем дескрипторы для них
    detector->detectAndCompute(img, noArray(), keypoints, descriptors);
}

/**
 * @brief Функция поиска соответствий между дескрипторами двух изображений
 *
 * Функция matchDescriptors() ищет соответствия между дескрипторами
 * двух изображений, используя алгоритм Brute-Force Matching.
 * Она использует коэффициент match_ratio для фильтрации
 * не надежных соответствий.
 *
 * @param[in] descriptor1 - матрица дескрипторов первого изображения
 * @param[in] descriptor2 - матрица дескрипторов второго изображения
 * @param[in] keypoints1 - вектор ключевых точек первого изображения
 * @param[in] keypoints2 - вектор ключевых точек второго изображения
 * @param[out] img1_pts - вектор координат точек на первом изображении
 * @param[out] img2_pts - вектор координат точек на втором изображении
 * @param[out] good_matches - вектор надежных соответствий
 * @param[in] match_ratio - коэффициент фильтрации (0.1f по умолчанию)
 */
void matchDescriptors(const Mat& descriptor1, const Mat& descriptor2, 
                      const std::vector<KeyPoint>& keypoints1, const std::vector<KeyPoint>& keypoints2, 
                      std::vector<Point2f>& img1_pts, std::vector<Point2f>& img2_pts, 
                      std::vector<DMatch>& good_matches, float match_ratio = 0.1f) {
    // создаем объект матчера Brute-Force
    BFMatcher matcher(NORM_HAMMING);

    // ищем соответствия между дескрипторами
    std::vector<std::vector<DMatch>> matches;
    matcher.knnMatch(descriptor1, descriptor2, matches, 2);

    // фильтруем не надежные соответствия
    for (const auto& match : matches) {
        if (match[0].distance < match_ratio * match[1].distance) {
            // добавляем координаты точек на оба изображения
            img1_pts.push_back(keypoints1[match[0].queryIdx].pt);
            img2_pts.push_back(keypoints2[match[0].trainIdx].pt);

            // добавляем надежное соответствие
            good_matches.push_back(DMatch(static_cast<int>(img1_pts.size() - 1), static_cast<int>(img2_pts.size() - 1), 0));
        }
    }
}

/**
 * @brief Функция отрисовки и отображения соответствий между
 *        ключевыми точками двух изображений
 *
 * Функция drawAndShowMatches() отрисовывает соответствия между
 * ключевыми точками двух изображений img1 и img2, используя
 * алгоритм drawMatches(). Она отображает результат на экране,
 * используя функцию imshow(), и сохраняет его в файл,
 * используя функцию imwrite().
 *
 * @param[in] img1 - первое изображение
 * @param[in] img2 - второе изображение
 * @param[in] keypoints1 - вектор ключевых точек первого изображения
 * @param[in] keypoints2 - вектор ключевых точек второго изображения
 * @param[in] matches - вектор соответствий между ключевыми точками
 */
void drawAndShowMatches(const Mat& img1, const Mat& img2,
                        const std::vector<KeyPoint>& keypoints1, const std::vector<KeyPoint>& keypoints2,
                        const std::vector<DMatch>& matches) {
    Mat dMatches;
    drawMatches(img1, keypoints1, img2, keypoints2, matches, dMatches);
    // отображаем результат на экране
    imshow("Matches", dMatches);
    // сохраняем результат в файл
    imwrite(output_path6 + "Matches.png", dMatches);
}

/**
 * @brief Создание панорамы из двух изображений
 *
 * Функция createPanorama() создает панораму из двух изображений img1 и img2,
 * используя матрицу гомографии H, полученную с помощью функции findHomography().
 * Результат - Mat-объект, содержащий панораму.
 *
 * @param[in] img1 - первое изображение
 * @param[in] img2 - второе изображение
 * @param[in] H - матрица гомографии
 * @param[out] panorama - панорама
 */
void createPanorama(const Mat& img1, const Mat& img2, const Mat& H, Mat& panorama) {
    // Создаем матрицу результата, размером со сумму ширины двух изображений
    Mat result;
    warpPerspective(img2, result, H, Size(img1.cols + img2.cols, img1.rows));

    // Создаем матрицу панорамы, заполненную белым цветом
    panorama = Mat(result.rows, result.cols, result.type(), Scalar(220, 220, 220));
    // Создаем подматрицу из левой части панорамы
    Mat half(panorama, Rect(0, 0, img1.cols, img1.rows));
    // Копируем левое изображение в левую часть панорамы
    img1.copyTo(half);

    // Проходимся по всем пикселям панорамы
    for (int y = 0; y < panorama.rows; y++) {
        for (int x = 0; x < panorama.cols; x++) {
            // Берем цвет пикселя на левой и правой частях панорамы
            Vec3b color1 = panorama.at<Vec3b>(y, x);
            Vec3b color2 = result.at<Vec3b>(y, x);
            // Если правый пиксель не черный, то смешиваем цвета
            if (color2 != Vec3b(0, 0, 0)) {
                float alpha = 0.01f;
                panorama.at<Vec3b>(y, x) = color1 * alpha + color2 * (1.0f - alpha);
            }
        }
    }
}

/**
 * @brief Функция для выполнения лабораторной работы 6
 *        (создание панорамы из двух изображений)
 *
 * @param[in] img_bgr - входное RGB-изображение
 *
 * @details
 * 1. Создаём папку для выходных изображений.
 * 2. Разделяем исходное изображение на две части.
 * 3. Определяем ключевые точки (ключевые точки) на каждой из частей.
 * 4. Создаём матрицы дескрипторов для каждой из частей.
 * 5. Определяем соответствия между ключевыми точками.
 * 6. Создаём матрицу гомографии.
 * 7. Создаём панораму.
 * 8. Выводим панораму на экран.
 * 9. Ожидаем нажатие клавиши.
 */
void lab6_Panorama(const Mat& img_bgr) {
    logger.info("Lab 6: Panorama.");

    // 1. Создаём папку для выходных изображений
    if (_access(output_path6.c_str(), 0) != 0) {
        if (_mkdir(output_path6.c_str()) == -1) {
            logger.error("Failed to create directory: {}", output_path6);
            return;
        }
        logger.info("Created directory: {}", output_path6);
    }

    Mat img1, img2;
    // 2. Разделяем исходное изображение на две части
    cutImg(img_bgr, img1, img2);

    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    // 3. Определяем ключевые точки на каждой из частей
    detectAndComputeFeatures(img1, keypoints1, descriptors1);
    detectAndComputeFeatures(img2, keypoints2, descriptors2);

    Mat keypoints1draw, keypoints2draw;
    // 4. Создаём матрицы дескрипторов для каждой из частей
    drawKeypoints(img1, keypoints1, keypoints1draw);
    drawKeypoints(img2, keypoints2, keypoints2draw);
    imshow("Keypoints1", keypoints1draw);
    imwrite(output_path6 + "Keypoints1.png", keypoints1draw);
    imshow("Keypoints2", keypoints2draw);
    imwrite(output_path6 + "Keypoints2.png", keypoints2draw);

    std::vector<Point2f> img1_pts, img2_pts;
    std::vector<DMatch> good_matches;
    // 5. Определяем соответствия между ключевыми точками
    matchDescriptors(descriptors1, descriptors2, keypoints1, keypoints2, img1_pts, img2_pts, good_matches);

    drawAndShowMatches(img1, img2, keypoints1, keypoints2, good_matches);

    Mat H = findHomography(img2_pts, img1_pts, RANSAC);

    Mat panorama;
    // 7. Создаём панораму
    createPanorama(img1, img2, H, panorama);

    // 8. Выводим панораму на экран
    imshow("Panorama", panorama);
    imwrite(output_path6 + "Panorama.png", panorama);

    // 9. Ожидаем нажатие клавиши
    waitKey();
}
#endif // PANORAMA_H