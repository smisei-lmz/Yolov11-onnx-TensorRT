#include "BufferManager.h"

#include <stdexcept>
#include <string>

BufferManager::BufferManager(){
    input_ = nullptr;
    output_ = nullptr;
}


BufferManager::~BufferManager(){
    release();
}

void BufferManager::allocate(std::size_t inputBytes, std::size_t outputBytes){
    cudaMalloc(&input_,inputBytes);
    cudaMalloc(&output_,outputBytes);
    std::cout << "GPU buffer allocated"<< std::endl;
}

void BufferManager::release() noexcept{
    if (input_){
        cudaFree(input_);
        input_ = nullptr;
    }
    if (output_){
        cudaFree(output_);
        output_ = nullptr;
    }
}
