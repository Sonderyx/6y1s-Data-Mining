#include "logger.hpp"
#include "1_Clustering.hpp"
#include "2_MovementEnergy.hpp"
#include "4_Classification.hpp"

using namespace std;
using namespace cv;

int main() {
    try {
        // string image_path = "../../resources/img/Orion.png";
        // string image_path = "../../resources/img/Shrek.png";
        // string image_path = "../../resources/img/Lena 21.png";
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

        string image_path = "../../resources/img/Car1.png";
        string image_path2 = "../../resources/img/Car2.png";

        init_logger();
        Mat img_bgr = imread(image_path);
        Mat img_bgr2 = imread(image_path2);

        // lab1_Clustering(img_bgr);
        lab2_MovementEnergy(img_bgr, img_bgr2);
        // lab4_Classification();

        spdlog::shutdown();
        return 0;
    }

    catch (const std::exception& e) {
        logger.error(e.what());
        spdlog::shutdown();
        return 1;
    }
}