#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <arm_neon.h>
#include <chrono>

using namespace std;
using namespace cv;

void normalise_gray(const uint8_t* gray, uint8_t* normalised,\
 int num_pixels, uint8_t x_min, uint8_t x_max)
{
  cout << "inside function normalise_gray" << endl;
  auto t1 = chrono::high_resolution_clock::now();

  for(int i=0; i<num_pixels; ++i) {
    float norm_val = (float)(gray[i] - x_min) / (float)(x_max - x_min);
    normalised[i] = norm_val * 255;
  }

  auto t2 = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::microseconds>(t2-t1).count();
  cout << duration << " us" << endl;
}

void normalise_gray_neon(const uint8_t* gray, uint8_t* normalised,\
  int num_pixels, uint8_t x_min, uint8_t x_max)
{
  num_pixels /= 8;
  float x_koef = (float)(x_max - x_min) * 255;

  float32x4_t x_mins = vdupq_n_f32(x_min);
  float32x4_t x_koefs = vdupq_n_f32(x_koef);

  auto t1_neon = chrono::high_resolution_clock::now();
  for(int i=0; i<num_pixels; ++i, gray+=4, normalised+=4) {
    float tmp_pixels_4_f32[] = { (float)gray[0], (float)gray[1], (float)gray[2], (float)gray[3] };
    float32x4_t pixels_4 = vld1q_f32(tmp_pixels_4_f32);
    pixels_4 = vsubq_f32(pixels_4, x_mins);
    pixels_4 = vmulq_f32(pixels_4, x_koefs);
    float pixels_4_f32[] = {0.0, 0.0, 0.0, 0.0};
    vst1q_f32(pixels_4_f32, pixels_4);
    uint8_t pixels_4_u8[] = { (uint8_t)pixels_4_f32[0], (uint8_t)pixels_4_f32[1], (uint8_t)pixels_4_f32[2], (uint8_t)pixels_4_f32[3] };
  }

  auto t2_neon = chrono::high_resolution_clock::now();
  auto duration_neon = chrono::duration_cast<chrono::microseconds>(t2_neon-t1_neon).count();
  cout << "inside function rgb_to_gray_neon" << endl;
  cout << duration_neon << " us" << endl;
}

int main(int argc,char** argv)
{
  uint8_t * gray_arr;
  uint8_t * normalised_arr;

  if (argc != 4) {
    cout << "Usage: <program> image_name x_min x_max" << endl;
    return -1;
  }

  Mat gray_image;
  gray_image = imread(argv[1], IMREAD_GRAYSCALE);
  if (!gray_image.data) {
    cout << "Could not open the image" << endl;
    return -1;
  }
  if (gray_image.isContinuous()) {
    gray_arr = gray_image.data;
  }
  else {
    cout << "data is not continuous" << endl;
    return -2;
  }

  int width = gray_image.cols;
  int height = gray_image.rows;
  int num_pixels = width*height;
  Mat normalised_image(height, width, CV_8UC1, Scalar(0));


  // find min and max values for normalisation
  uint8_t x_min = atoi(argv[2]);
  uint8_t x_max = atoi(argv[3]);

  cout << "min pixel value: " << (int)x_min << endl; 
  cout << "max pixel value: " << (int)x_max << endl;

  uint8_t* gray;
  
  gray = gray_arr;
  normalised_arr = normalised_image.data;
  normalise_gray(gray_arr, normalised_arr, num_pixels, x_min, x_max);
  imwrite("normalised_image.png", normalised_image);

  gray = gray_arr;
  normalised_arr = normalised_image.data;
  normalise_gray_neon(gray_arr, normalised_arr, num_pixels, x_min, x_max);
  imwrite("normalised_neon_image.png", normalised_image);

  return 0;
}
