#pragma once 
#include <vector>
#include "Types.h"
#include "Preprocess.h"

class Postprocessor{
public:
    static std::vector<Detection> decode(float* output,int num_classes,int num_boxes,float conf_threshold,float nms_threshold,LetterBoxInfo info,int img_w,int img_h);
};

