#ifndef MAHALANOBIS_H
#define MAHALANOBIS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <random>
#include <numeric>
#include <chrono>
#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include "logger.hpp"

using namespace std;
using namespace cv;

struct IrisData {
    vector<double> features;  // Вектор признаков (длина чашелистика, ширина чашелистика, длина лепестка, ширина лепестка)
    string label;             // Метка класса: "Iris-setosa", "Iris-versicolor", "Iris-virginica"
};

/**
 * @brief Функция для загрузки данных из файла
 *
 * Функция принимает на вход путь к файлу и возвращает вектор структур IrisData,
 * каждый элемент которого содержит вектор признаков (features) и метку класса (label)
 *
 * @param filename путь к файлу
 * @return вектор структур IrisData
 */
vector<IrisData> loadData(const string& filename) {
    vector<IrisData> dataset;
    ifstream file(filename);
    string line;

    // Если файл не открыт, то выводим ошибку
    if (!file.is_open()) {
        logger.error("Failed to open the file: {}", filename);
        return dataset;
    }

    // Читаем файл построчно
    while (getline(file, line)) {
        IrisData data;
        stringstream ss(line);
        string token;

        // Читаем токены в строке
        while (getline(ss, token, ',')) {
            // Если токен начинается с цифры, то добавляем его к вектору признаков
            if (isdigit(token[0]))
                data.features.push_back(stod(token)); // Добавляем числовые данные в вектор признаков
            else
                data.label = token; // Последний токен — это метка класса
        }

        // Если вектор признаков не пустой и метка класса не пустая, то добавляем data к dataset
        if (!data.features.empty() && !data.label.empty())
            dataset.push_back(data);
        else
            logger.warn("Empty data row or invalid format: {}", line);
    }

    logger.info("Loaded {} samples", dataset.size());
    return dataset;
}

/**
 * @brief Функция для разделения данных на обучающую (train) и тестовую (test) выборки
 *
 * @details
 * Функция принимает на вход полный набор данных (dataset), а также
 *   2 пустых контейнеров (map<string, vector<IrisData>>) для хранения
 *   обучающей (trainData) и тестовой (testData) выборок.
 *   Она разделяет данные на 2 выборки в соотношении train_ratio : (1 - train_ratio)
 *   с помощью генератора случайных чисел.
 *
 * @param dataset полный набор данных
 * @param trainData контейнер для хранения обучающей выборки
 * @param testData контейнер для хранения тестовой выборки
 * @param train_ratio соотношение между обучающей и тестовой выборками
 */
void splitData(const vector<IrisData>& dataset, map<string, vector<IrisData>>& trainData, map<string, vector<IrisData>>& testData, double train_ratio) {
    logger.info("Splitting data into train and test sets with ratio {}/{}", 100 - train_ratio*100, train_ratio*100);
    // Получаем текущее время в качестве seed
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    // Создаем генератор случайных чисел
    std::default_random_engine generator(seed);

    // Определяем распределение (например, равномерное от 1 до 100)
    std::uniform_int_distribution<int> distribution(0, 100);

    // Проходим по каждому элементу данных (data)
    for (const auto& data : dataset) {
        // Генерируем случайное число от 0 до 1
        double random_number = (rand() / static_cast<double>(RAND_MAX));
        // double random_number = (distribution(generator) / static_cast<double>(100));

        // Если случайное число меньше заданного соотношения (train_ratio), то добавляем данные к обучающей выборке (trainData), иначе добавляем данные к тестовой выборке (testData)
        if (random_number <= train_ratio)
            trainData[data.label].push_back(data);
        else
            testData[data.label].push_back(data);
    }

    // Подсчитываем количество элементов в обучающей выборке
    size_t train_size = accumulate(trainData.begin(), trainData.end(), size_t(0), [](size_t sum, const auto& pair) {
        // Суммируем количество элементов в каждом векторе
        return sum + pair.second.size();
    });

    // Подсчитываем количество элементов в тестовой выборке
    size_t test_size = accumulate(testData.begin(), testData.end(), size_t(0), [](size_t sum, const auto& pair) {
        // Суммируем количество элементов в каждом векторе
        return sum + pair.second.size();
    });

    logger.info("Splited data ({} samples) into training ({} samples) and testing ({} samples)", train_size + test_size, train_size, test_size);
}

/**
 * @brief Функция для расчета среднего вектора
 *
 * @param[in] data Вектор структур данных, содержащих вектор признаков (features)
 * @return Вектор средних значений признаков
 */
vector<double> calculateMean(const vector<IrisData>& data) {
    // Определяем количество признаков
    int num_features = data[0].features.size();

    // Создаем вектор для хранения средних значений
    vector<double> mean(num_features, 0.0);

    // Проходим по каждому элементу данных (data)
    for (const auto& sample : data)
        // Проходим по каждому признаку (feature)
        for (int i = 0; i < num_features; ++i)
            // Складываем значения признака
            mean[i] += sample.features[i];

    // Делим сумму значений признака на количество элементов данных
    for (double& val : mean)
        val /= data.size();

    // Возвращаем расчитанный средний вектор
    return mean;
}

/**
 * @brief Функция для логгирования ковариационной матрицы
 *
 * @param matrix Матрица, которую необходимо вывести построчно
 */
void logCovarianceMatrix(const Mat& matrix) {
    // Проходим по строкам матрицы
    for (int i = 0; i < matrix.rows; ++i) {
        stringstream ss;
        ss << "[";

        for (int j = 0; j < matrix.cols; ++j) {
            // Добавляем элемент матрицы в строку с разделителем запятая
            ss << matrix.at<double>(i, j);
            if (j < matrix.cols - 1)
                ss << ", ";
        }

        ss << "]";

        // Логируем каждую строку матрицы
        logger.info("{}", ss.str());
    }
}

/**
 * @brief Функция для вычисления ковариационной матрицы
 *
 * @param data Вектор структур данных, содержащих вектор признаков (features) и метку класса (label)
 * @param mean Вектор средних значений признаков
 * @return Матрица ковариации размером (num_features x num_features)
 */
Mat calculateCovarianceMatrix(const vector<IrisData>& data, const vector<double>& mean) {
    int num_features = mean.size();
    Mat covariance = Mat::zeros(num_features, num_features, CV_64F);

    logger.info("Mean vector size: {}", num_features);

    // Проходим по каждому элементу данных
    for (const auto& sample : data) {
        // Создаем матрицу разности между текущим элементом данных и средним вектором
        Mat diff = Mat(sample.features) - Mat(mean);

        // Вычисляем произведение diff на транспонированную матрицу и добавляем к матрице ковариации
        covariance += diff * diff.t();
    }

    // Делим матрицу на количество элементов данных
    covariance /= data.size();

    // Если матрица ковариации пуста, выводим ошибку
    if (covariance.empty())
        logger.error("Covariance matrix is empty.");
    else {
        // Выводим матрицу ковариации в лог
        logger.info("Covariance matrix for feature {}x{}:", covariance.rows, covariance.cols);
        logCovarianceMatrix(covariance);
    }

    return covariance;
}

/**
 * @brief Функция для вычисления обратной матрицы
 *
 * @param matrix Матрица, для которой необходимо вычислить обратную
 * @return Обратная матрица (матрица, умноженная на которую,
 *          даст единичную матрицу)
 */
Mat calculateInverse(const Mat& matrix) {
    if (matrix.empty()) {
        logger.error("Matrix is empty, cannot calculate inverse.");
        return Mat();
    }

    Mat inv_matrix = matrix.inv();

    if (inv_matrix.empty())
        // Если матрица не может быть обращена, то выводим ошибку
        logger.error("Inversion of the matrix failed.");
    else
        // Если матрица может быть обращена, то выводим информацию о размере полученной матрицы
        logger.info("Matrix inversion successful. Size: {}x{}", inv_matrix.rows, inv_matrix.cols);

    return inv_matrix;
}

/**
 * @brief Рассчитывает расстояние Махаланобиса между вектором признаков (sample) и средним вектором (mean)
 *        с учетом ковариационной матрицы (cov_inv)
 *
 * @param sample Вектор признаков
 * @param mean Средний вектор
 * @param cov_inv Обратная ковариационная матрица
 * @return Расстояние Махаланобиса
 */
double mahalanobisDistance(const vector<double>& sample, const vector<double>& mean, const Mat& cov_inv) {
    // Создаем матрицу разности между вектором признаков (sample) и средним вектором (mean)
    Mat diff = Mat(sample) - Mat(mean);

    // Если матрица разности пуста, то выводим ошибку
    if (diff.empty()) {
        logger.error("One of the matrices (diff or cov_inv) is empty.");
        return -1.0;
    }

    // Создаем транспонированную матрицу разности
    Mat diff_t = diff.t();

    // Вычисляем скалярное произведение матрицы разности (diff_t) на
    // обратную ковариационную матрицу (cov_inv) и на матрицу разности (diff)
    // и сохраняем результат в матрице result
    Mat result = diff_t * cov_inv * diff;

    // Если матрица result пуста, то выводим ошибку
    if (result.empty()) {
        logger.error("Resulting matrix is empty.");
        return -1.0;
    }

    // Вычисляем сумму квадратов элементов матрицы result
    double sum_of_squares = sum(result)[0];

    // Возвращаем квадратный корень из суммы квадратов
    return sqrt(sum_of_squares);
}

/**
 * @brief Функция классификации по расстоянию Махаланобиса
 *
 * @param sample Вектор признаков, который нужно классифицировать
 * @param means Карта, где ключ - метка класса, а значение - средний вектор признаков для этого класса
 * @param cov_inverses Карта, где ключ - метка класса, а значение - обратная ковариационная матрица для этого класса
 * @return Метка класса, к которому принадлежит вектор признаков
 */
string classify(const IrisData& sample, const map<string, vector<double>>& means, const map<string, Mat>& cov_inverses) {
    string best_class;
    double min_distance = numeric_limits<double>::max();

    // Проходим по каждому классу
    for (const auto& [label, mean] : means) {
        // Рассчитываем расстояние Махаланобиса между вектором признаков (sample)
        // и средним вектором (mean) для текущего класса
        double distance = mahalanobisDistance(sample.features, mean, cov_inverses.at(label));

        // Если расстояние меньше минимального, то обновляем минимальное расстояние
        // и запоминаем метку класса
        if (distance < min_distance) {
            min_distance = distance;
            best_class = label;
        }
    }

    // Возвращаем метку класса, к которому принадлежит вектор признаков
    return best_class;
}

// Основная программа
int lab2_Mahalanobis() {
    logger.info("Lab 2: Mahalanobis clustering");

    // Шаг 1. Загрузка данных
    string iris_path = "../../resources/iris db/iris.data";
    vector<IrisData> dataset = loadData(iris_path);

    // Шаг 2. Разделение на обучающую (90%) и тестовую (10%) выборки
    map<string, vector<IrisData>> trainData, testData;
    splitData(dataset, trainData, testData, 0.9);

    // Шаг 3. Оценка средних векторов и ковариационных матриц
    map<string, vector<double>> means;
    map<string, Mat> covariances, cov_inverses;

    for (const auto& [label, data] : trainData) {
        means[label] = calculateMean(data);
        Mat covariance = calculateCovarianceMatrix(data, means[label]);
        covariances[label] = covariance;
        cov_inverses[label] = covariance.inv();  // Обратная ковариационная матрица
    }

    // Шаг 4. Тестирование классификатора на тестовой выборке
    int correct_predictions = 0, total_predictions = 0;

    for (const auto& [label, test_samples] : testData)
        for (const auto& sample : test_samples) {
            string predicted_label = classify(sample, means, cov_inverses);
            if (predicted_label == label)
                correct_predictions++;
            total_predictions++;
        }

    // Шаг 5. Вывод точности классификации
    double accuracy = static_cast<double>(correct_predictions) / total_predictions * 100;
    logger.info("Accuracy: {:.2f}%", accuracy);

    return 0;
}

#endif MAHALANOBIS_H