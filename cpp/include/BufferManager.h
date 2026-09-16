#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <cstddef>

class BufferManager {
public:
    BufferManager();
    ~BufferManager();

    BufferManager(BufferManager const&) = delete;
    BufferManager& operator=(BufferManager const&) = delete;

    void allocate(std::size_t inputBytes, std::size_t outputBytes);
    void release() noexcept;

    void* input_device() const noexcept {
        return input_;
    }
    void* output_device() const noexcept {
        return output_;
    }

private:
    void* input_{nullptr};
    void* output_{nullptr};
};
