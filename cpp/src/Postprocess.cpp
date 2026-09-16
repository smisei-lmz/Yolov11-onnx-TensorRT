#include <iostream>
#include "Postprocess.h"
#include <algorithm>

std::vector<Detection> Postprocessor::decode(float* output,int num_classes,int num_boxes,float conf_threshold,float nms_threshold,LetterBoxInfo info,int img_w,int img_h){
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    /*
        YOLO输出：[1,84,8400]
        output[0 * 8400 + i]  -> cx
        output[1 * 8400 + i]  -> cy
        output[2 * 8400 + i]  -> width
        output[3 * 8400 + i]  -> height
        output[4  * 8400 + i] -> 类别 0 的分数
        output[5  * 8400 + i] -> 类别 1 的分数
        ...
        output[83 * 8400 + i] -> 类别 79 的分数
    */
    for (int i=0;i<num_boxes;i++){
        float cx = output[i]; //遍历8400个框，
        float cy = output[num_boxes+i];
        float w =output[2*num_boxes+i];
        float h =output[3*num_boxes+i];
        float max_score=0.0f;
        int class_id=-1;
        // 找最大类别
        for(int c=0;c<num_classes;c++){
            float score =
                output[(4+c)*num_boxes+i];
            if(score > max_score){
                max_score = score;
                class_id = c;
            }
        } 
        // confidence过滤
        if(max_score<conf_threshold){
            continue;
        }
        //找到了符合大于置信度阈值的框了
        float x1 = cx - w/2.0f;
        float y1 = cy - h/2.0f;
        float x2 = cx + w/2.0f;
        float y2 = cy + h/2.0f;   
        //恢复原图坐标
        x1 -= info.pad_x;
        x2 -= info.pad_x;
        y1 -= info.pad_y;
        y2 -= info.pad_y;
        x1 /= info.scale;
        x2 /= info.scale;
        y1 /= info.scale;
        y2 /= info.scale;
        // clamp std::clamp由cpp17提供
        x1 = std::clamp(x1, 0.0f, static_cast<float>(img_w - 1)); //x1 限制在（0,img_w - 1）之间
        y1 = std::clamp(y1, 0.0f, static_cast<float>(img_h - 1));
        x2 = std::clamp(x2, 0.0f, static_cast<float>(img_w - 1));
        y2 = std::clamp(y2, 0.0f, static_cast<float>(img_h - 1));
        boxes.emplace_back(cv::Rect( cv::Point((int)x1,(int)y1), cv::Point((int)x2,(int)y2) ));
        scores.emplace_back(max_score);
        class_ids.emplace_back(class_id);
    }

    //NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes,scores,conf_threshold, nms_threshold,indices);
    std::vector<Detection> results;
    for(auto idx:indices){
        Detection det;
        det.class_id =class_ids[idx];
        det.confidence =scores[idx];
        det.x1 =boxes[idx].x;
        det.y1 =boxes[idx].y;
        det.x2 = boxes[idx].x +boxes[idx].width;
        det.y2 = boxes[idx].y +boxes[idx].height;
        results.push_back(det);
    }
    return results;
}

