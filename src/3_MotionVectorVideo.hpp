#ifndef MOTIONVECTORVIDEO_H
#define MOTIONVECTORVIDEO_H

#include <opencv2/opencv.hpp>
#include "3_MotionVector.hpp"

using namespace cv;
using namespace std;

void processVideoStream(const string& videoSourceURL, const string& outputVideoPath) {
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

    // outputVideoPath = "output.mp4";
    // Создаём видеозаписывающий поток
    // VideoWriter outputVideo(outputVideoPath, VideoWriter::fourcc('M', 'P', '4', 'V'), fps, Size(frameWidth, frameHeight));
    // VideoWriter outputVideo(outputVideoPath, VideoWriter::fourcc('H', '2', '6', '4'), 25, Size(frameWidth, frameHeight));
    // VideoWriter outputVideo(outputVideoPath, VideoWriter::fourcc('X', 'V', 'I', 'D'), 25, Size(frameWidth, frameHeight));
    // if (!outputVideo.isOpened()) {
    //     logger.error("Failed to open output video file: " + outputVideoPath);
    //     // logger.error("Failed to open output video file: output.avi");
    //     return;
    // }

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
    Mat motionVectorImageFiltered;

    // int frameCount = 0;
    while (true) {
        cap >> currFrame; // Считываем текущий кадр
        if (currFrame.empty()) break;

        // Сжимаем кадр для обработки
        resize(currFrame, currFrame, Size(currFrame.cols / scaleFactor, currFrame.rows / scaleFactor));
        Mat grayFrame;
        cvtColor(currFrame, grayFrame, COLOR_BGR2GRAY);

        // Вычисляем векторы движения
        computeMotionVectors(prevFrame, grayFrame, motionVectorsOrig, blockSize, true);

        drawMotionVectors(currFrame, motionVectorImage, motionVectorsOrig);

        // Фильтруем векторы движения
        recursiveMedianFilter(motionVectorsOrig, motionVectorsFiltered, 3);

        // Кластеризация
        clusterMotionVectors(motionVectorsFiltered, labels, 30.0F);

        // Визуализация векторов движения
        drawMotionVectors(currFrame, motionVectorImageFiltered, motionVectorsFiltered);

        // Визуализация кластеров
        visualizeClusterContours(currFrame, clusterVisualization, labels, blockSize);

        // Отображение результатов
        imshow("Original Video", currFrame);
        imshow("Original Motion Vectors", motionVectorImage);
        imshow("Filtered Motion Vectors", motionVectorImageFiltered);
        imshow("Cluster Visualization", clusterVisualization);

        // Запись видео
        // outputVideo << clusterVisualization;
        // frameCount++;

        // Выход по клавише ESC
        if (waitKey(1) == 27) break;

        // Обновляем предыдущий кадр
        prevFrame = grayFrame.clone();
    }
    // logger.info("Total frames written: " + to_string(frameCount));

    cap.release();
    // outputVideo.release();
    destroyAllWindows();
}

#endif // MOTIONVECTORVIDEO_H
