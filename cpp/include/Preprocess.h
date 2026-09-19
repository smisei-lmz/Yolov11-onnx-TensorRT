#pragma once 

#include "Types.h"
#include <opencv2/opencv.hpp>


class Preprocessor{
public:
    static cv::Mat letterbox(const cv::Mat& image, LetterBoxInfo& info , int target_size=640);
    static void preprocess(const cv::Mat& image,float* output,LetterBoxInfo& info);
};
