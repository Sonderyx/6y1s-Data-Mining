#ifndef CLUSTERING_H
#define CLUSTERING_H

#include <iostream>
#include <vector>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include "logger.hpp"
#include <direct.h>
#include <io.h>

using namespace std;
using namespace cv;

/**
 * @brief Calculates the Euclidean distance between two points
 *
 * @param point1 The first point
 * @param point2 The second point
 * @return The Euclidean distance between the two points
 */
double calcEucDist(const vector<double>& point1, const vector<double>& point2) {
    double distance = 0.0;

    // Calculate the squared difference between the corresponding coordinates of the two points
    for (size_t i = 0; i < point1.size(); ++i)
        distance += pow(point1[i] - point2[i], 2);

    // Calculate the square root of the sum of the squared differences
    return sqrt(distance);
}

/**
 * @brief Initializes the centers of the clusters
 *
 * @param k The number of clusters
 * @param dimensions The number of dimensions in the data
 * @return A vector of vectors, where each inner vector represents a cluster center
 */
vector<vector<double>> initRandCenters(int k, int dimensions) {
    vector<vector<double>> centers;

    // Initialize k cluster centers with random values between 0 and 1
    for (int i = 0; i < k; ++i) {
        vector<double> center;
        for (int j = 0; j < dimensions; ++j)
            center.push_back((rand() % 256) / 255.0); // For color values between 0 and 1
        centers.push_back(center);
    }

    return centers;
}

/**
 * @brief Prints the centers of the clusters to the logger
 *
 * @param centers The centers of the clusters
 */
void printCenters(const vector<vector<double>>& centers) {
    for (size_t i = 0; i < centers.size(); ++i) {
        // Create an output stream to build the string
        std::ostringstream center_stream;

        // Add the center of the cluster to the stream
        center_stream << "Center of cluster " << (i + 1) << ": [";
        // center_stream << "             ├Center of cluster " << (i + 1) << ": [";

        // Add the coordinates of the center to the stream
        for (size_t j = 0; j < centers[i].size(); ++j) {
            center_stream << centers[i][j];

            // Add a comma if it's not the last coordinate
            if (j < centers[i].size() - 1)
                center_stream << ", ";
        }

        // Close the bracket
        center_stream << "]";

        // Log the string
        logger.info(center_stream.str());
    }
}

/**
 * @brief Runs the K-means clustering algorithm on the given image
 *
 * @param src The input image
 * @param dst The output image with the clustered colors
 * @param k The number of clusters to form
 * @param max_iterations The maximum number of iterations to run the algorithm
 */
void segmentImageKMeans(Mat& src, Mat& dst, int k, int max_iterations) {
    dst = Mat::zeros(src.size(), src.type());
    int channels = src.channels();

    // Convert the image to a vector of points
    vector<vector<double>> points;
    for (int i = 0; i < src.rows; ++i)
        for (int j = 0; j < src.cols; ++j) {
            vector<double> pixel;
            for (int c = 0; c < channels; ++c)
                pixel.push_back(static_cast<double>(src.at<Vec3b>(i, j)[c]) / 255.0);
            points.push_back(pixel);
        }

    // Initialize the cluster centers
    vector<vector<double>> centers = initRandCenters(k, channels);

    // Run the K-means algorithm
    vector<int> labels(points.size());
    int iteration = 0;
    bool converged = false;
    while (!converged && iteration < max_iterations) {
        vector<vector<double>> new_centers(k, vector<double>(channels, 0.0));
        vector<int> counts(k, 0);

        // Assign each point to a cluster
        for (size_t i = 0; i < points.size(); ++i) {
            double min_distance = numeric_limits<double>::max();
            int closest_center = 0;
            for (int j = 0; j < k; ++j) {
                double distance = calcEucDist(points[i], centers[j]);
                if (distance < min_distance) {
                    min_distance = distance;
                    closest_center = j;
                }
            }
            for (int c = 0; c < channels; ++c)
                new_centers[closest_center][c] += points[i][c];
            counts[closest_center]++;
            labels[i] = closest_center;
        }

        // Recalculate the cluster centers
        for (int i = 0; i < k; ++i)
            if (counts[i] != 0)
                for (int c = 0; c < channels; ++c)
                    new_centers[i][c] /= counts[i];

        // Check for convergence
        if (new_centers == centers)
            converged = true;
        else
            centers = new_centers;

        // Print the cluster centers after each iteration
        logger.info("Iteration {}:", (iteration + 1));
        // logger.info("Iteration {}: ┐", iteration + 1);
        printCenters(centers);

        iteration++;
    }

    // Create a new image with the colors of the cluster centers
    for (size_t i = 0; i < points.size(); ++i) {
        int x = i / src.cols;
        int y = i % src.cols;
        for (int c = 0; c < channels; ++c)
            dst.at<Vec3b>(x, y)[c] = static_cast<uchar>(centers[labels[i]][c] * 255);
    }
}

/**
 * @brief Runs the K-means clustering algorithm on the given image
 *
 * @param img_bgr The input image in BGR format
 * @param num_clusters The number of clusters to form
 * @param max_iterations The maximum number of iterations to run the algorithm
 */
void lab1_Clustering(Mat& img_bgr) {
    // Initialize the logger
    init_logger();
    logger.info("Lab 1: K-Means Clustering");
    string output_path = "output img/Lab 1 clustering/";

    // Resize the image
    resize(img_bgr, img_bgr, img_bgr.size() / 2);

    // Display the original image
    imshow("image bgr", img_bgr);
    waitKey(1);

    cout << "Enter the number of clusters: ";
    int num_clusters;
    cin >> num_clusters;
    if (num_clusters <= 0) {
        logger.error("Invalid number of clusters: {}", num_clusters);
        return;
    }

    // ----------------------Solo Run with k = num_clusters----------------------
    logger.info("Clustering BGR image with k = {}", num_clusters);
    Mat img_k_bgr;
    // Run the K-means algorithm on the BGR image
    segmentImageKMeans(img_bgr, img_k_bgr, num_clusters, 100);
    imshow(to_string(num_clusters) + " clusters", img_k_bgr);
    if (_access(output_path.c_str(), 0) == -1)
        _mkdir(output_path.c_str());
    imwrite(output_path + to_string(num_clusters) + " clusters.png", img_k_bgr);

    // ----------------------All Run up to k = num_clusters----------------------
    // for (int i = 1; i <= num_clusters; i++) {
    //     // Run the K-means algorithm on the BGR image
    //     logger.info("Clustering BGR image with k = {}", num_clusters);
    //     Mat img_k_bgr;
    //     segmentImageKMeans(img_bgr, img_k_bgr, i, 100);

    //     // Display and save the clustered images
    //     imshow(to_string(i) + " clusters", img_k_bgr);
    //     if (_access(output_path.c_str(), 0) == -1)
    //         _mkdir(output_path.c_str());
    //     imwrite(output_path + to_string(i) + " clusters.png", img_k_bgr);
    // }
    // --------------------------------------------------------------------------

    // Wait for the user to press a key
    waitKey();
}

#endif CLUSTERING_H