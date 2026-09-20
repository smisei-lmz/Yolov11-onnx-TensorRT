#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "TrtEngine.h"
//#include "Preprocess.h"
#include "CudaPreprocess.h"
#include "Postprocess.h"
#include "Labels.h"

int main(){
    std::string engine_path ="/home/smisei/Yolov11-ONNX-TensorRT/models/yolo11s_static_Nodynamic_Nosimplify_Nonms_mixed_fp16.engine";
    std::string image_path ="/home/smisei/Yolov11-ONNX-TensorRT/data/test0.jpg";
    CudaPreprocessor cudaPreprocessor;
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
    const std::size_t row_bytes = static_cast<std::size_t>(img_w) * 3;
    const std::size_t raw_bytes = row_bytes * img_h;
    // 给原图分配GPU内存
    if (!engine.buffers_info().ensureRawCapacity(raw_bytes)) {
        std::cerr << "Allocate raw GPU buffer failed\n";
        return -1;
    }
    // CPU OpenCV图像复制到GPU
    cudaError_t error = cudaMemcpy2D(engine.buffers_info().raw_device(),row_bytes,image.data,image.step,row_bytes,img_h,cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "Image H2D failed: "
                << cudaGetErrorString(error) << '\n';
        return -1;
    }
    auto preprocess_start = std::chrono::steady_clock::now();
    error = cudaPreprocessor.Cuda_Preprocess((unsigned char * )engine.buffers_info().raw_device(),img_w,img_h,img_w * 3,
                                                (float * )engine.buffers_info().input_device(),engine.inputWidth(),engine.inputHeight());
    if (error != cudaSuccess) {
        std::cerr << "CUDA preprocess failed: "
                << cudaGetErrorString(error) << '\n';
        return -1;
    }
    LetterBoxInfo info = cudaPreprocessor.Get_CudaLetterBoxinfo();
    //Preprocessor::preprocess(image,input.data(),info);
    auto preprocess_end = std::chrono::steady_clock::now();
    double total_preprocess =std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();
    std::cout << "Preprocess time:" << total_preprocess <<"ms"  <<std::endl;
    //inference
    std::vector<float> output(engine.outputSize());
    constexpr int warmup_iterations = 10;
    constexpr int benchmark_iterations = 100;
    for (int i = 0; i < warmup_iterations; ++i) {
        if (!engine.infer_gpubuffer( output.data())) {
            return -1;
        }
    }

    InferenceTiming average;
    for (int i = 0; i < benchmark_iterations; ++i) {
        InferenceTiming current;
        if (!engine.infer_gpubuffer(output.data(), &current)) {
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
              << "  Pipeline FPS: " << 1000.0f / average.gpu_total_ms << "\n"<<std::endl;
    std::cout<<"Inference done"<<std::endl;
    auto Postprecess_start = std::chrono::steady_clock::now();
    auto detections = Postprocessor::decode(output.data(),80,8400,0.25f,0.45f,info,img_w,img_h);
    auto Postprecess_end = std::chrono::steady_clock::now();
    double total_postprocess =std::chrono::duration<double, std::milli>(Postprecess_end - Postprecess_start).count();
    std::cout<<"Postprocess time:"<< total_postprocess << "ms\n"<<std::endl;
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
    cv::imwrite("/home/smisei/Yolov11-ONNX-TensorRT/results/test0_result.jpg",image);
    std::cout<<"result saved in /home/smisei/Yolov11-ONNX-TensorRT/results/test0_result.jpg"<<std::endl;
    return 0;
}
