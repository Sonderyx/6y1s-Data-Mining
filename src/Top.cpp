// #define WIN32_LEAN_AND_MEAN
// #define _HAS_STD_BYTE 0
// #include <Windows.h>
#include "1_Clustering.hpp"
#include "2_Mahalanobis.hpp"
#include "logger.hpp"
// #include <nlohmann/json.hpp>

using namespace std;
using namespace cv;
// using namespace spdlog;

int main() {
    try {
        // string image_path = "../../resources/img/Orion.png";
        // string image_path = "../../resources/img/Shrek.png";
        string image_path = "../../resources/img/Lena 21.png";
        // string image_path = "../../resources/img/Spirit.jpg";
        // string image_path = "../../resources/img/Lena 70.jpg";
        // string image_path = "../../resources/img/Berserk.jpg";
        // string image_path = "../../resources/img/Berserk 2.jpg";
        // string image_path = "../../resources/img/New York.jpg";
        // string image_path = "../../resources/img/Dash.jpg";
        // string image_path = "../../resources/img/FP1.png";
        // string image_path = "../../resources/img/Morph1.png";
        // string image_path = "../../resources/img/Plane.jpg";
        // string image_path = "../../resources/img/Levitan.jpg";
        // string image_path = "../../resources/img/test.jpg";
        // string image_path = "../../resources/img/Anime.jpg";

        init_logger();
        Mat img_bgr = imread(image_path);

        // lab1_Clustering(img_bgr);
        lab2_Mahalanobis();

        spdlog::shutdown();
        return 0;
    }

    catch (const std::exception& e) {
        logger.error(e.what());
        spdlog::shutdown();
        // spdlog::error(e.what());
        return 1;
    }
}