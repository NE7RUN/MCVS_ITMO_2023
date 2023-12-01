#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <arm_neon.h>
#include <chrono>

using namespace std;
using namespace cv;

void normalize_grayscale_neon(const uint8_t *gray, uint8_t *normalized_gray, int num_pixels, uint8_t min_val, uint8_t max_val)
{
    num_pixels /= 8;
    uint8x8_t min_val_neon = vdup_n_u8(min_val);
    uint8x8_t max_val_neon = vdup_n_u8(max_val);
    uint8x8_t range_neon = vsub_u8(max_val_neon, min_val_neon);

    auto t1_neon = chrono::high_resolution_clock::now();
    for (int i = 0; i < num_pixels; ++i, gray += 8, normalized_gray += 8)
    {
        uint8x8_t input = vld1_u8(gray);
        uint8x8_t shifted_input = vqsub_u8(input, min_val_neon);
        uint8x8_t scaled_output = vmul_u8(shifted_input, vqmovn_u16(vmull_u8(shifted_input, range_neon)));
        vst1_u8(normalized_gray, scaled_output);
    }
    auto t2_neon = chrono::high_resolution_clock::now();
    auto duration_neon = chrono::duration_cast<chrono::microseconds>(t2_neon - t1_neon).count();
    cout << "Time taken with NEON: " << duration_neon << " us" << endl;
}

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        cout << "Usage: " << argv[0] << " image_path min_val max_val" << endl;
        return -1;
    }

    Mat gray_image = imread(argv[1], IMREAD_GRAYSCALE);
    if (!gray_image.data)
    {
        cout << "Could not open the image" << endl;
        return -1;
    }
    int num_pixels = gray_image.cols * gray_image.rows;

    uint8_t *normalized_gray_arr_neon = gray_image.data;

    uint8_t min_val = stoi(argv[2]);
    uint8_t max_val = stoi(argv[3]);

    Mat normalized_image_neon(gray_image.size(), CV_8UC1);
    uint8_t *normalized_gray_arr_neon_new = normalized_image_neon.data;

    normalize_grayscale_neon(normalized_gray_arr_neon, normalized_gray_arr_neon_new, num_pixels, min_val, max_val);

    imwrite("normalized_with_neon.png", normalized_image_neon);

    return 0;
}