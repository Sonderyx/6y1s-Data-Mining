#include "logger.hpp"
#include "1_Clustering.hpp"
#include "2_MovementEnergy.hpp"
#include "4_Classification.hpp"
#include "3_MotionVector.hpp"
#include "3_MotionVectorVideo.hpp"
#include "5_Regression.hpp"
#include "6_Panorama.hpp"

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
        string image_path = "../../resources/img/Greece.jpg";

        init_logger();

        string output_path = "output/";
        // создаём папку для выходных изображений
        if (_access(output_path.c_str(), 0) != 0)
            if (_mkdir(output_path.c_str()) == -1) {
                logger.error("Failed to create directory: {}", output_path);
                return 1;
                }
            logger.info("Created directory: {}", output_path);


        // string videoSourceURL = "rtsp://1701954d6d07.entrypoint.cloud.wowza.com:1935/app-m75436g0/27122ffc_stream2";
        // string videoSourceURL = "rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa";
        // string videoSourceURL = "rtsp://rtspstream.com/parking";
        // string videoSourceURL = "rtsp://62.109.19.230:554/iloveyou";
        // string videoSourceURL = "rtsp://rtspstream:e03758124d6f6fb0fbd4c2aaf36a3a7c@zephyr.rtsp.stream/movie";
        // string videoSourceURL = "rtsp://localhost:8554/newsky";
        // string videoSourceURL = "http://gifted-satoshi.192-185-7-210.plesk.page.192-185-7-210.hgws27.hgwin.temp.domains/api/ProveAttachmentApi/open-prove-attachment/765b0ad0-20c5-4a99-f798-08dcef7f61b1";
        // string videoSourceURL = "C:/Users/David Polis/Downloads/Video/Test.ts";

        // string videoSourceURL = "../../resources/video/WebStream4.ts";
        // processVideoStream(videoSourceURL, "test.avi");

        // string image_path = "../../resources/img/164.jpeg";
        // string image_path2 = "../../resources/img/168.jpeg";

        Mat img_bgr = imread(image_path);
        // Mat img_bgr2 = imread(image_path2);

        // lab1_Clustering(img_bgr);
        // lab2_MovementEnergy(img_bgr, img_bgr2);
        // lab3_MotionVector(img_bgr, img_bgr2);
        // lab4_Classification();
        lab6_Panorama(img_bgr);


        // string image_path = "../../resources/img/Chromatic aberration chess.png";
        // // string image_path = "../../resources/img/Chromatic aberration chess 2.bmp";
        // Mat img_bgr = imread(image_path);
        // // lab5_Regression(img_bgr, 1, 8, 20);
        // lab5_Regression(img_bgr, 1, 16, 40);

        spdlog::shutdown();
        return 0;
    }

    catch (const std::exception& e) {
        logger.error(e.what());
        spdlog::shutdown();
        return 1;
    }
}