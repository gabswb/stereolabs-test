#include <iostream>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/core/cuda.hpp>

#include "TrtEngine.h"
#include "Yolov10.hpp"


int main() {

    Logger logger;
    
    auto trt_engine = std::make_unique<TrtEngine>(logger);
    // trt_engine->BuildEngine("../models/yolov10s.onnx", "../models/yolov10s.engine");
    trt_engine->LoadEngine("../models/yolov10s.engine");

    YOLOv10 yolo{std::move(trt_engine)};

    cv::Mat img = cv::imread("../images/image.jpg", cv::IMREAD_COLOR);

    cv::Mat out = yolo.Detect(img);

    cv::imwrite("output.jpg", out);
}
