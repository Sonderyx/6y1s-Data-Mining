#include "1KMeans.hpp"

using namespace std;
using namespace cv;

int main() {
    // string image_path = "../../resources/img/Orion.png";
    // string image_path = "../../resources/img/Lena 21.png";
    // string image_path = "../../resources/img/Lena 70.jpg";
    // string image_path = "../../resources/img/Berserk.jpg";
    // string image_path = "../../resources/img/Berserk 2.jpg";
    // string image_path = "../../resources/img/New York.jpg";
    // string image_path = "../../resources/img/Dash.jpg";
    // string image_path = "../../resources/img/FP1.png";
    string image_path = "../../resources/img/Shrek.png";
    // string image_path = "../../resources/img/Morph1.png";
    // string image_path = "../../resources/img/Plane.jpg";
    // string image_path = "../../resources/img/Levitan.jpg";
    // string image_path = "../../resources/img/test.jpg";
    // string image_path = "../../resources/img/Anime.jpg";

    Mat img_bgr = imread(image_path);

    lab1KMeans(img_bgr, 5);

    return 0;
}