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

bool BufferManager::ensureRawCapacity(std::size_t bytes) {
    if (bytes == 0) {
        return false;
    }

    if (raw_ != nullptr && bytes <= raw_capacity_) {
        return true;
    }

    if (raw_ != nullptr) {
        cudaFree(raw_);
        raw_ = nullptr;
        raw_capacity_ = 0;
    }

    cudaError_t error = cudaMalloc(&raw_, bytes);

    if (error != cudaSuccess) {
        raw_ = nullptr;
        return false;
    }

    raw_capacity_ = bytes;
    return true;
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
    if (raw_) {
        cudaFree(raw_);
        raw_ = nullptr;
        raw_capacity_ = 0;
    }
}
