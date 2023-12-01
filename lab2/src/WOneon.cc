#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <chrono>

using namespace std;
using namespace cv;

void normalize_grayscale(const uint8_t *gray, uint8_t *normalize_gray, int num_pixels, uint8_t min_val, uint8_t max_val)
{
    auto t1 = chrono::high_resolution_clock::now();
    for (int i = 0; i < num_pixels; ++i)
    {
        normalize_gray[i] = (uint8_t)((gray[i] - min_val) * 255.0 / (max_val - min_val));
    }
    auto t2 = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
    cout << "Time taken without NEON: " << duration << " us" << endl;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cout << "Usage: " << argv[0] << " image_path" << endl;
        return -1;
    }

    Mat gray_image = imread(argv[1], IMREAD_GRAYSCALE);
    if (!gray_image.data)
    {
        cout << "Could not open the image" << endl;
        return -1;
    }
    int num_pixels = gray_image.cols * gray_image.rows;

    uint8_t *normalized_gray_arr = gray_image.data;

    uint8_t min_val = 50;
    uint8_t max_val = 200;

    normalize_grayscale(normalized_gray_arr, normalized_gray_arr, num_pixels, min_val, max_val);

    imwrite("normalized_without_neon.png", gray_image);

    return 0;
}