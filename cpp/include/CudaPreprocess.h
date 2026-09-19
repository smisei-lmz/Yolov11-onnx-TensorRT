#pragma once

#include <cuda_runtime_api.h>
#include "Types.h"

class CudaPreprocessor{
public:
    CudaPreprocessor() = default;
    ~CudaPreprocessor()= default;
    cudaError_t CudaNormalize(unsigned char* d_input, float* d_output, int size);
    cudaError_t CudaBGR2RGB_HWC2CHW_Normalize(unsigned char* d_input, float* d_output, int width , int height);
    cudaError_t Cuda_Preprocess(const unsigned char* src,int src_width,int src_height,int src_stride,
                                                float* dst,int dst_width,int dst_height);
    struct LetterBoxInfo Get_CudaLetterBoxinfo(){
        return info_;
    }
private:
    struct LetterBoxInfo info_;    
};

