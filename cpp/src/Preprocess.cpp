#include "Preprocess.h"

cv::Mat Preprocessor::letterbox(const cv::Mat& image, LetterBoxInfo& info, int target_size){
    int w = image.cols;
    int h = image.rows;
    float scale = std::min((target_size/float(w)) , (target_size/float(h)));
    int new_w = scale * w;
    int new_h = scale * h;
    cv::Mat resize;
    cv::resize(image,resize,cv::Size(new_w,new_h));
    int pad_x = target_size - new_w;
    int pad_y = target_size - new_h;
    pad_x /=2;
    pad_y /=2;
    cv::Mat output(target_size,target_size,CV_8UC3,cv::Scalar(114,114,114));
    resize.copyTo(output(cv::Rect(pad_x,pad_y,new_w,new_h)));
    info.scale = scale;
    info.pad_x = pad_x;
    info.pad_y = pad_y;
    return output;
}

void Preprocessor::preprocess(const cv::Mat& image,float* output,LetterBoxInfo& info){
    cv::Mat img = letterbox(image,info);
    cv::cvtColor(img,img,cv::COLOR_BGR2RGB);
    img.convertTo(img,CV_32FC3,1.0/255.0); //归一化 转fp32
    int channel = img.channels();
    int height = img.rows;
    int width = img.cols;
    int index=0;
    for(int c=0;c<channel;c++)
    {
        for(int h=0;h<height;h++)
        {
            for(int w=0;w<width;w++)
            {
                output[index++]=img.at<cv::Vec3f>(h,w)[c];
            }

        }

    }
}
