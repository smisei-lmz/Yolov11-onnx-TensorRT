#pragma once

#include <string>


struct Detection
{
    /* data */
    int class_id;
    float confidence;
    float x1;
    float y1;
    float x2;
    float y2;
};


struct LetterBoxInfo {
    float scale{0.0f};
    int resized_w{0};
    int resized_h{0};
    int pad_x{0};
    int pad_y{0};
};