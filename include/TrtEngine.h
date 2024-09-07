#pragma once

#include <string>
#include <memory>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include "utils.hpp"

class TrtEngine {

public:
    TrtEngine();
    TrtEngine(Logger logger);
    void BuildEngine(const std::string& onnx_filepath, const std::string& engine_filepath);
    void LoadEngine(const std::string& engine_filepath);
    void Inference(void *input_tensor, void *output_tensor);
    void AsyncInference(cudaStream_t stream, void *input_tensor, void *output_tensor);
    std::vector<size_t> GetInputDim();
    std::vector<size_t> GetOutputDim();

private:
    Logger logger_;

    std::string input_tensor_name_;
    std::string output_tensor_name_;

    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;


};