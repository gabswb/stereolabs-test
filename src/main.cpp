#include <iostream>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/core/cuda.hpp>

#include "TrtEngine.h"
#include "Yolov10.hpp"
#include "Yolov8.h"


int main() {

    Logger logger;
    
    auto trt_engine = std::make_unique<TrtEngine>(logger);
    // trt_engine->BuildEngine("../models/yolov10s.onnx", "../models/yolov10s.engine");
    trt_engine->LoadEngine("../models/yolov8s.engine");

    YOLOv8 yolo{std::move(trt_engine)};

    cv::Mat img = cv::imread("../images/image1.jpg", cv::IMREAD_COLOR);

    cv::Mat out = yolo.Detect(img);

    cv::imwrite("output1.jpg", out);


    // cv::VideoCapture cap("../images/video.mp4");

    // if (!cap.isOpened()) {
    //     std::cout << "Cannot open the video file. \n";
    //     return -1;
    // }

    // int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    // int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    // cv::VideoWriter video("outcpp.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, cv::Size(frame_width,frame_height));    
    // cv::Mat frame;

    // while(true){
    //     cap >> frame;

    //     if (frame.empty())
    //         break;

    //     cv::Mat out = yolo.Detect(frame);
    //     video.write(out);
    //     std::cout << "write\n";
    // }
    
    // cap.release();
    // video.release();
}
