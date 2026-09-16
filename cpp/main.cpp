#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>

#include "TrtEngine.h"
#include "Preprocess.h"
#include "Postprocess.h"
#include "Labels.h"

int main(){
    std::string engine_path ="/home/smisei/Yolov11-ONNX-TensorRT/models/yolo11s_static_Nodynamic_Nosimplify_Nonms_fp32.engine";
    std::string image_path ="/home/smisei/Yolov11-ONNX-TensorRT/data/test.jpg";

    cv::Mat image = cv::imread(image_path);
    if (image.empty()){
        std::cout << "no test images in " << std::endl;
        return -1;
    }
    int img_w =image.cols;
    int img_h =image.rows;
    std::cout<<"Image:"<<img_w<<"x"<<img_h<<std::endl;
    TrtEngine engine;
    if (!engine.loadEngine(engine_path)){
        return -1;
    }
    //preprocess
    std::vector<float> input(engine.inputSize());
    LetterBoxInfo info;
    Preprocessor::preprocess(image,input.data(),info);
    //inference
    std::vector<float> output(engine.outputSize());

    constexpr int warmup_iterations = 10;
    constexpr int benchmark_iterations = 100;
    for (int i = 0; i < warmup_iterations; ++i) {
        if (!engine.infer(input.data(), output.data())) {
            return -1;
        }
    }

    InferenceTiming average;
    for (int i = 0; i < benchmark_iterations; ++i) {
        InferenceTiming current;
        if (!engine.infer(input.data(), output.data(), &current)) {
            return -1;
        }
        average.h2d_ms += current.h2d_ms;
        average.execute_ms += current.execute_ms;
        average.d2h_ms += current.d2h_ms;
        average.gpu_total_ms += current.gpu_total_ms;
        average.sync_wait_ms += current.sync_wait_ms;
    }
    average.h2d_ms /= benchmark_iterations;
    average.execute_ms /= benchmark_iterations;
    average.d2h_ms /= benchmark_iterations;
    average.gpu_total_ms /= benchmark_iterations;
    average.sync_wait_ms /= benchmark_iterations;

    std::cout << std::fixed << std::setprecision(3)
              << "Average over " << benchmark_iterations << " runs:\n"
              << "  H2D:          " << average.h2d_ms << " ms\n"
              << "  TensorRT:     " << average.execute_ms << " ms\n"
              << "  D2H:          " << average.d2h_ms << " ms\n"
              << "  GPU pipeline: " << average.gpu_total_ms << " ms\n"
              << "  Host sync wait: " << average.sync_wait_ms << " ms\n"
              << "  Pipeline FPS: " << 1000.0f / average.gpu_total_ms << std::endl;
    std::cout<<"Inference done"<<std::endl;
    auto detections = Postprocessor::decode(output.data(),80,8400,0.25f,0.45f,info,img_w,img_h);
    std::cout<<"Detection num:"<<detections.size()<<std::endl;
    //draw
    for (auto detection:detections){
        cv::rectangle(image,cv::Rect(detection.x1,detection.y1,detection.x2-detection.x1,detection.y2-detection.y1),cv::Scalar(0,255,0),2);
        std::string label = "unknown";
        if (detection.class_id >= 0
            && static_cast<std::size_t>(detection.class_id) < kCocoLabels.size()) {
            label = kCocoLabels[detection.class_id];
        }
        std::string text = label + ":" + std::to_string(detection.confidence).substr(0,4);
        cv::putText(image,text,cv::Point(detection.x1,detection.y1-5),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,255,0), 2);
    }
    cv::imwrite("/home/smisei/Yolov11-ONNX-TensorRT/results/test_result.jpg",image);
    std::cout<<"result saved in /home/smisei/Yolov11-ONNX-TensorRT/results/test_result.jpg"<<std::endl;
    return 0;
}
