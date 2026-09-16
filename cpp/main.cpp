#include <NvInfer.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "TrtEngine.h"
std::string enginePath="/home/smisei/Yolov11-ONNX-TensorRT/models/yolo11s_static_Nodynamic_Nosimplify_Nonms_fp32.engine";

std::vector<float> readBin(
    const std::string& path,
    int size
)
{

    std::vector<float> data(
        size
    );


    std::ifstream file(
        path,
        std::ios::binary
    );


    if(!file)
    {
        throw std::runtime_error(
            "cannot open input"
        );
    }



    file.read(
        reinterpret_cast<char*>(
            data.data()
        ),
        size*sizeof(float)
    );


    return data;

}
void saveBin(
    const std::string& path,
    std::vector<float>& data
)
{

    std::ofstream file(
        path,
        std::ios::binary
    );


    file.write(
        reinterpret_cast<char*>(
            data.data()
        ),
        data.size()*sizeof(float)
    );


}
int main(int argc,char* argv[]) {
    if (argc < 4){
        std::cout << "Usage:\n" << "./yolo_trt" << "engine input.bin output.bin"<< std::endl;
        return -1;
    }
    std::string engine_path = argv[1];
    std::string input_path =argv[2];
    std::string output_path = argv[3];
    //load TensotRT engine 
    TrtEngine engine;
    if (!engine.loadEngine(engine_path)){
        return -1;
    }
    std::cout << "Input elements:" << engine.inputSize() << std::endl;
    std::cout << "Output elements:" << engine.outputSize() << std::endl;
    auto input = readBin(input_path,engine.inputSize());
    auto output = readBin(output_path,engine.outputSize());
    std::vector<float> output(engine.outputSize());
    //inference
    bool ok =engine.infer(input.data(),output.data());
    if(!ok){
        return -1;
    } 
    // Save output
    saveBin(output_path,output);
    std::cout<<"Inference finished"<<std::endl;
    std::cout<<"Saved:"<<output_path<<std::endl;
    return 0;
}
