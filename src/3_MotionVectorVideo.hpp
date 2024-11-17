#ifndef MOTIONVECTORVIDEO_H
#define MOTIONVECTORVIDEO_H

#include <opencv2/opencv.hpp>
#include "3_MotionVector.hpp"

using namespace cv;
using namespace std;

void processVideoStream(const string& videoSourceURL) {
    logger.info("Starting video stream processing...");

    // Открываем видеопоток
    VideoCapture cap(videoSourceURL);
    if (!cap.isOpened()) {
        logger.error("Failed to open video source: " + videoSourceURL);
        return;
    }
    cap.set(cv::CAP_PROP_BUFFERSIZE, 300); // Уменьшаем буфер

    // Получаем параметры видеопотока
    int frameWidth = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(CAP_PROP_FPS));
    logger.info("Video stream opened: " + to_string(frameWidth) + "x" + to_string(frameHeight) + " @ " + to_string(fps) + " FPS");

    Mat prevFrame, currFrame;
    cap >> prevFrame; // Считываем первый кадр
    if (prevFrame.empty()) {
        logger.error("Failed to read initial frame from video stream.");
        return;
    }

    // Сжимаем кадр для ускорения обработки
    int scaleFactor = 2;
    resize(prevFrame, prevFrame, Size(prevFrame.cols / scaleFactor, prevFrame.rows / scaleFactor));
    cvtColor(prevFrame, prevFrame, COLOR_BGR2GRAY);

    // Параметры для анализа
    int blockSize = 10; // Размер блока для вычисления векторов движения
    vector<vector<MotionVector>> motionVectorsOrig;
    vector<vector<MotionVector>> motionVectorsFiltered;
    vector<vector<int>> labels;

    Mat clusterVisualization;
    Mat motionVectorImage;

    while (true) {
        cap >> currFrame; // Считываем текущий кадр
        if (currFrame.empty()) break;

        // Сжимаем кадр для обработки
        resize(currFrame, currFrame, Size(currFrame.cols / scaleFactor, currFrame.rows / scaleFactor));
        Mat grayFrame;
        cvtColor(currFrame, grayFrame, COLOR_BGR2GRAY);

        // Вычисляем векторы движения
        computeMotionVectors(prevFrame, grayFrame, motionVectorsOrig, blockSize);

        // Фильтруем векторы движения
        recursiveMedianFilter(motionVectorsOrig, motionVectorsFiltered, 3);

        // Кластеризация
        clusterMotionVectors(motionVectorsFiltered, labels, 30.0F);

        // Визуализация векторов движения
        drawMotionVectors(currFrame, motionVectorImage, motionVectorsFiltered);

        // Визуализация кластеров
        visualizeClusterContours(currFrame, clusterVisualization, labels, blockSize);

        // Отображение результатов
        imshow("Original Video", currFrame);
        imshow("Motion Vectors", motionVectorImage);
        imshow("Cluster Visualization", clusterVisualization);

        // Выход по клавише ESC
        if (waitKey(1) == 27) break;

        // Обновляем предыдущий кадр
        prevFrame = grayFrame.clone();
    }

    cap.release();
    destroyAllWindows();
}

#endif // MOTIONVECTORVIDEO_H