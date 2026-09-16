#include "TrtEngine.h"

#include <fstream>
#include <iostream>
#include <vector>


using namespace nvinfer1;

TrtEngine::TrtEngine(){
    cudaStreamCreate(&stream_);
}
TrtEngine::~TrtEngine(){
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
    int nb = engine_ -> getNbIOTensors();
    std::cout<<"IO number:"<<nb<<std::endl;
    for(int i=0;i<nb;i++){
        const char* name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);
        if (mode == TensorIOMode::kINPUT){
            input_name_ = name;\
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
        buffers_.allocate(input_size_*sizeof(float),output_size_*sizeof(float));
        context_->setTensorAddress(input_name_.c_str(),buffers_.input_device());
        context_->setTensorAddress(output_name_.c_str(),buffers_.output_device());
        return true;
    }
}


bool TrtEngine::infer(float* input , float* output){
    //H2D
    cudaMemcpyAsync(buffers_.input_device(),input,input_size_*sizeof(float),cudaMemcpyHostToDevice,stream_);
    //execute
    bool success = context_->enqueueV3(stream_);
    if (!success){
        std::cout << "TensorRT inference failed" <<std::endl;
        return false;
    }
    //D2H
    cudaMemcpyAsync(output,buffers_.output_device(),output_size_*sizeof(float),cudaMemcpyDeviceToHost,stream_);
    cudaStreamSynchronize(stream_);

}










