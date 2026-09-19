#include "CudaPreprocess.h"

#include <cuda_runtime.h>

namespace {
/*
    这是cuda用于进行yolo前处理的kernel命名空间
    流程输入cv::Mat -> 进行resize，padding -> 进行BGR2RGB HWC2CHW Normalize 
    最终kernel 直接输入原始size图像 输出一个resize+ letterbox+ padding+ BGR→RGB+ HWC→CHW + normalize后的新的size的图像，这样就不用额外申请内存了
*/

//读一个像素：给定xyc，读取像素位置(x,y)的c通道的值  给bilinear_kernel用
__device__ inline unsigned char getPixel(const unsigned char* input,int width, int height , int stride , int x, int y , int c){
    x = max(0, min(x,width-1));
    y = max(0, min(y,height-1));
    return input[y*stride + x*3 + c];
}
//双线性差值kernel:读取每个像素的四周的像素做均值 给resize_kernel用
__device__ inline float bilinear_kernel(const unsigned char* input , int width ,int height , int stride, float x,float y ,int channel){
    int x0 =static_cast<int>(floorf(x));
    int y0 =static_cast<int>(floorf(y));
    int x1 =x0 + 1;
    int y1 =y0 + 1;
    float dx =x - x0;
    float dy =y - y0;
    float p00 =static_cast<float>(getPixel(input,width,height,stride,x0,y0, channel));
    float p01 =static_cast<float>(getPixel(input,width,height,stride,x1,y0, channel));
    float p10 =static_cast<float>(getPixel(input,width,height,stride,x0,y1, channel));
    float p11 =static_cast<float>(getPixel(input,width,height,stride,x1,y1, channel));
    float top =p00 * (1.0f - dx)+p01 * dx;
    float bottom =p10 * (1.0f - dx)+p11 * dx;
    return top * (1.0f - dy)+bottom * dy; 
}

//进行Resize
__global__ void Resize_kernel(unsigned char* input,float* output,int target_width,int target_height){
    
}
//同时进行BGR2RGB HWC2CHW Normalize
__global__ void BGR2RGB_HWC2CHW_Normalize_kernel(unsigned char* input,float* output,int width,int height){
    //这里调用二维线程，因为图像天生二维，dim3 threads(16,16);
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if(idx>=width||idy>=height) return;
    //获取本次处理的像素位置，虽然图像是二维的，但是内存本质上是一维的，所以根据index找到真实地址 
    int src = (idy*width+idx)*3;  //src表示原图像一个像素的R的地址  因为opencv的内存方式是BGRBGRBGR而不是BBBBB GGGGGG RRRRR 所以需要*3
    int area = width*height;      //area指的是整个图像分辨率大小，整体内存占用是area*3，因为是3通道图像
    int dst = idy*width+idx;      //dst表示目标图像一个像素的R的地址
    // RGB
    output[0*area+dst]=input[src+2]/255.0f;  // input[src+2] 
    output[1*area+dst]=input[src+1]/255.0f;
    output[2*area+dst]=input[src+0]/255.0f;
}

__global__ void Yolo_Preprocess_kernel(const unsigned char* src,int src_width,int src_height,int src_stride,
                                            float* dst,int dst_width,int dst_height,
                                            float scale,int resized_width,int resized_height,int pad_x,int pad_y) {
    //
    //padding + resize+ letterbox+ BGR→RGB+ HWC→CHW + normalize
    int dx = blockIdx.x * blockDim.x +threadIdx.x;
    int dy = blockIdx.y * blockDim.y +threadIdx.y;
    if (dx >= dst_width || dy >= dst_height){
        return;
    }           
    int area = dst_height * dst_width;
    int dst_index = dy * dst_width +dx;    
    //如果处理的位置处理padding区域                        
    if (dx < pad_x || dx >= pad_x+resized_width || dy < pad_y || dy >= pad_y+resized_height){
        float padding_value = 114.0f/255.0f;
        dst[0*area + dst_index] = padding_value;
        dst[1*area + dst_index] = padding_value;
        dst[2*area + dst_index] = padding_value;
        return; //padding的像素处理完直接返回
    }
    /*
            y = 0
        ┌────────────────────────┐
        │      上 padding         │ 140 行
        ├────────────────────────┤ y = 140
        │                        │
        │   resized 原图区域      │ 360 行
        │                        │
        ├────────────────────────┤ y = 500
        │      下 padding         │ 140 行
        └────────────────────────┘
        y = 640
    */
    //处理正常像素，把dst映射回src
    const int resized_x = dx - pad_x;
    const int resized_y = dy - pad_y;
    const float src_x =  (static_cast<float>(resized_x) + 0.5f) /scale - 0.5f;
    const float src_y =  (static_cast<float>(resized_y) + 0.5f) /scale - 0.5f;

    const float b = bilinear_kernel(src,src_width,src_height,src_stride,src_x,src_y,0);
    const float g = bilinear_kernel(src,src_width,src_height,src_stride,src_x,src_y,1);
    const float r = bilinear_kernel(src,src_width,src_height,src_stride,src_x,src_y,2);

    // BGR -> RGB
    // HWC -> CHW
    // unsigned char -> float，并归一化到 [0, 1]
    dst[0 * area + dst_index] = r / 255.0f;
    dst[1 * area + dst_index] = g / 255.0f;
    dst[2 * area + dst_index] = b / 255.0f;

}

__global__ void normalize_kernel(const unsigned char* input,float* output,int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }
    output[idx] = input[idx] / 255.0f;
}



}  // namespace

cudaError_t CudaPreprocessor::CudaNormalize(unsigned char* d_input, float* d_output, int size) {
    if (d_input == nullptr || d_output == nullptr || size <= 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    normalize_kernel<<<blocks, threads>>>(d_input, d_output, size);
    
    const cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        return launch_error;
    }

    return cudaDeviceSynchronize();
}


//BGR2RGB HWC2CHW Normalize 的cpp实现，可以在preprocess.h中声明，在此定义
cudaError_t CudaPreprocessor::CudaBGR2RGB_HWC2CHW_Normalize(unsigned char* d_input, float* d_output, int width , int height) {
    if (d_input == nullptr || d_output == nullptr || width <= 0 || height <= 0) {
        return cudaErrorInvalidValue;
    }

    dim3 threads(16, 16);
    // 3. 计算 Grid 中 Block 的数量
    dim3 blocks(
        (width + threads.x - 1) / threads.x,
        (height + threads.y - 1) / threads.y
    );
    BGR2RGB_HWC2CHW_Normalize_kernel<<<blocks, threads>>>(d_input, d_output, width , height);
    const cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        return launch_error;
    }
    return cudaDeviceSynchronize();
}


cudaError_t CudaPreprocessor::Cuda_Preprocess(const unsigned char* src,int src_width,int src_height,int src_stride,
                                            float* dst,int dst_width,int dst_height) {

    if (src == nullptr ||
        dst == nullptr ||
        src_width <= 0 ||
        src_height <= 0 ||
        dst_width <= 0 ||
        dst_height <= 0 ||
        src_stride < src_width * 3)
    {
        return cudaErrorInvalidValue;
    }
    dim3 threads(16,16);
    // 3. 计算 Grid 中 Block 的数量
    dim3 blocks(
        (dst_width + threads.x - 1) / threads.x,
        (dst_height + threads.y - 1) / threads.y
    );


        // 保持宽高比，选择更小的缩放比例
    info_.scale = std::min(
        dst_width / static_cast<float>(src_width),
        dst_height / static_cast<float>(src_height));

    // 缩放后真实图像的尺寸
    info_.resized_w = static_cast<int>(
        std::round(src_width * info_.scale));

    info_.resized_h = static_cast<int>(
        std::round(src_height * info_.scale));

    // 左侧和上侧 padding
    info_.pad_x = (dst_width - info_.resized_w) / 2;
    info_.pad_y = (dst_height - info_.resized_h) / 2;

    Yolo_Preprocess_kernel<<<blocks, threads>>>(src,src_width,src_height,src_stride,dst,dst_width,dst_height,info_.scale,info_.resized_w,info_.resized_h,info_.pad_x,info_.pad_y);
    const cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        return launch_error;
    }
    return cudaDeviceSynchronize();
}

