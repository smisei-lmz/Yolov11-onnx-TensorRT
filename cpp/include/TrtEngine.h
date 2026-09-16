#pragma once
#include <NvInfer.h>

#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <memory>
#include "Logger.h"
#include "BufferManager.h"
#include <vector>
class TrtEngine{
public:
    TrtEngine();
    ~TrtEngine();
    bool loadEngine(const std::string& path);
    bool infer(float* input , float* output);
    int inputSize(){
        return input_size_;
    }
    int outputSize(){
        return output_size_;
    }
private:
    int input_size_;
    int output_size_;
    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    BufferManager buffers_;
    cudaStream_t stream_;
    std::string input_name_;
    std::string output_name_;
private:
    std::vector<char> loadFile(const std::string& path);
};