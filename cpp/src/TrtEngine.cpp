#include "TrtEngine.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>


using namespace nvinfer1;

TrtEngine::TrtEngine(){
    cudaStreamCreate(&stream_);
    cudaEventCreate(&start_event_);
    cudaEventCreate(&h2d_event_);
    cudaEventCreate(&execute_event_);
    cudaEventCreate(&d2h_event_);
}
TrtEngine::~TrtEngine(){
    cudaEventDestroy(start_event_);
    cudaEventDestroy(h2d_event_);
    cudaEventDestroy(execute_event_);
    cudaEventDestroy(d2h_event_);
    cudaStreamDestroy(stream_);
}
std ::vector<char> TrtEngine::loadFile(const std::string& path){
    std::ifstream file(path,std::ios::binary);
    file.seekg(0,std::ios::end);
    size_t size =file.tellg();
    file.seekg(0,std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(),size);
    return buffer;
}
bool TrtEngine::loadEngine(const std::string& path){
    auto data = loadFile(path);
    runtime_.reset(createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(data.data(),data.size()));
    if(!engine_){
        std::cout<<"engine load failed"<<std::endl;
        return false;
    }
    context_.reset(engine_->createExecutionContext());
    if (!context_){
        std::cout << "context creation failed" << std::endl;
        return false;
    }
    int nb = engine_ -> getNbIOTensors();
    std::cout<<"IO number:"<<nb<<std::endl;
    for(int i=0;i<nb;i++){
        const char* name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);
        if (mode == TensorIOMode::kINPUT){
            input_name_ = name;
        }else{
            output_name_ = name;
        }
        auto shape = engine_->getTensorShape(name);
        int size = 1;
        for (int j=0;j<shape.nbDims;j++){
            size *= shape.d[j];
        }
        if (mode == TensorIOMode::kINPUT){
            input_size_ = size;
        }else{
            output_size_ = size;
        }
        std::cout << "Tensor:" << name <<"size:" << size <<std::endl;
    }

    if (input_name_.empty() || output_name_.empty() || input_size_ <= 0 || output_size_ <= 0){
        std::cout << "invalid input/output tensors" << std::endl;
        return false;
    }

    buffers_.allocate(input_size_*sizeof(float),output_size_*sizeof(float));
    if (!context_->setTensorAddress(input_name_.c_str(),buffers_.input_device())
        || !context_->setTensorAddress(output_name_.c_str(),buffers_.output_device())){
        std::cout << "failed to bind tensor addresses" << std::endl;
        return false;
    }
    return true;
}


bool TrtEngine::infer(float* input, float* output, InferenceTiming* timing){
    if (timing){
        cudaEventRecord(start_event_, stream_);
    }
    //H2D
    cudaMemcpyAsync(buffers_.input_device(),input,input_size_*sizeof(float),cudaMemcpyHostToDevice,stream_);
    if (timing){
        cudaEventRecord(h2d_event_, stream_);
    }
    //execute
    bool success = context_->enqueueV3(stream_);
    if (!success){
        std::cout << "TensorRT inference failed" <<std::endl;
        return false;
    }
    if (timing){
        cudaEventRecord(execute_event_, stream_);
    }
    //D2H
    cudaMemcpyAsync(output,buffers_.output_device(),output_size_*sizeof(float),cudaMemcpyDeviceToHost,stream_);
    if (timing){
        cudaEventRecord(d2h_event_, stream_);
    }

    const auto sync_start = std::chrono::steady_clock::now();
    const cudaError_t sync_status = cudaStreamSynchronize(stream_);
    const auto sync_end = std::chrono::steady_clock::now();
    if (sync_status != cudaSuccess){
        std::cout << "CUDA stream synchronization failed: "
                  << cudaGetErrorString(sync_status) << std::endl;
        return false;
    }

    if (timing){
        cudaEventElapsedTime(&timing->h2d_ms, start_event_, h2d_event_);
        cudaEventElapsedTime(&timing->execute_ms, h2d_event_, execute_event_);
        cudaEventElapsedTime(&timing->d2h_ms, execute_event_, d2h_event_);
        cudaEventElapsedTime(&timing->gpu_total_ms, start_event_, d2h_event_);
        timing->sync_wait_ms = std::chrono::duration<double, std::milli>(
            sync_end - sync_start).count();
    }
    return true;
}








