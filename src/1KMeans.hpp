#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace std;
using namespace cv;

// Функция для вычисления евклидова расстояния между двумя точками
double calculateDistance(const std::vector<double>& point1, const std::vector<double>& point2) {
    double distance = 0.0;
    for (size_t i = 0; i < point1.size(); ++i) {
        distance += std::pow(point1[i] - point2[i], 2);
    }
    return std::sqrt(distance);
}

// Функция для инициализации центров кластеров
std::vector<std::vector<double>> initializeCenters(int k, int dimensions) {
    std::vector<std::vector<double>> centers;
    for (int i = 0; i < k; ++i) {
        std::vector<double> center;
        for (int j = 0; j < dimensions; ++j) {
            center.push_back((rand() % 256) / 255.0); // Для значений цвета от 0 до 1
        }
        centers.push_back(center);
    }
    return centers;
}

// Функция для сегментации изображения с использованием алгоритма К-средних
cv::Mat segmentImageKMeans(cv::Mat& image, int k, int max_iterations) {
    int dimensions = image.cols * image.rows;
    int channels = image.channels();

    // Преобразование изображения в вектор точек
    std::vector<std::vector<double>> points;
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            std::vector<double> pixel;
            if (channels == 1) {
                pixel.push_back(static_cast<double>(image.at<uchar>(i, j)) / 255.0);
            } else {
                for (int c = 0; c < channels; ++c) {
                    pixel.push_back(static_cast<double>(image.at<cv::Vec3b>(i, j)[c]) / 255.0);
                }
            }
            points.push_back(pixel);
        }
    }

    // Инициализация центров кластеров
    std::vector<std::vector<double>> centers = initializeCenters(k, channels);

    // Применение алгоритма К-средних
    std::vector<int> labels(points.size());
    int iteration = 0;
    bool converged = false;
    while (!converged && iteration < max_iterations) {
        // Вывод ключевых чисел для отслеживания
        std::cout << "Centers: ";
        for (int c = 0; c < channels; ++c) {
            std::cout << "Ch " << c << ": ";
            for (const auto& center : centers) {
                std::cout << center[c] << " ";
            }
            // std::cout << "\t";
        }
        std::cout << std::endl;

        std::vector<std::vector<double>> new_centers(k, std::vector<double>(channels, 0.0));
        std::vector<int> counts(k, 0);

        // Назначение каждой точке кластера
        for (size_t i = 0; i < points.size(); ++i) {
            double min_distance = std::numeric_limits<double>::max();
            int closest_center = 0;
            for (int j = 0; j < k; ++j) {
                double distance = calculateDistance(points[i], centers[j]);
                if (distance < min_distance) {
                    min_distance = distance;
                    closest_center = j;
                }
            }
            for (int c = 0; c < channels; ++c) {
                new_centers[closest_center][c] += points[i][c];
            }
            counts[closest_center]++;
            labels[i] = closest_center;
        }

        // Пересчет центров кластеров
        for (int i = 0; i < k; ++i) {
            if (counts[i] != 0) {
                for (int c = 0; c < channels; ++c) {
                    new_centers[i][c] /= counts[i];
                }
            }
        }

        // Проверка сходимости алгоритма
        if (new_centers == centers) {
            converged = true;
        } else {
            centers = new_centers;
        }

        iteration++;
    }

    // Создание нового изображения с цветами центров кластеров
    cv::Mat clustered_image(image.size(), image.type());
    for (size_t i = 0; i < points.size(); ++i) {
        int x = i / image.cols;
        int y = i % image.cols;
        if (channels == 1) {
            clustered_image.at<uchar>(x, y) = static_cast<uchar>(centers[labels[i]][0] * 255);
        } else {
            for (int c = 0; c < channels; ++c) {
                clustered_image.at<cv::Vec3b>(x, y)[c] = static_cast<uchar>(centers[labels[i]][c] * 255);
            }
        }
    }

    return clustered_image;
}

void lab1KMeans(cv::Mat& img_bgr, int k=5) {
    Mat img_gray;
    cvtColor(img_bgr, img_gray, COLOR_BGR2GRAY);

    imshow("image gray", img_gray);
    imshow("image bgr", img_bgr);

    int max_iterations = 7;
    cout << "Processing gray image with k = " << k << endl;
    cv::Mat img_k_gray = segmentImageKMeans(img_gray, k, max_iterations);

    cout << "\nProcessing bgr image with k = " << k << endl;
    cv::Mat img_k_bgr = segmentImageKMeans(img_bgr, k, max_iterations);

    imshow("image clustered gray", img_k_gray);
    imshow("image clustered bgr", img_k_bgr);

    waitKey();
}