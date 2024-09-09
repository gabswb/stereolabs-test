#include <iostream>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/core/cuda.hpp>

#include "TrtEngine.h"
#include "Yolov10.hpp"
#include "Yolov8.h"

struct Options {
    bool build = false;
    bool timing = false;
    std::string video_path;
    std::string image_path;
    std::string load_path;
    std::string precision;
};

bool parse_arguments(int argc, char* argv[], Options &opts) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--video") {
            if (i + 1 < argc) {
                opts.video_path = argv[++i];
            } else {
                std::cerr << "Error: --video requires a path\n";
                return false;
            }
        } 
        else if (arg == "--image") {
            if (i + 1 < argc) {
                opts.image_path = argv[++i];
            } else {
                std::cerr << "Error: --image requires a path\n";
                return false;
            }
        } 
        else if (arg == "--build") {
            if (i + 1 < argc) {
                opts.precision = argv[++i];
            } else {
                std::cerr << "Error: --image requires a path\n";
                return false;
            }
        }
        else if (arg == "--timing") {
            opts.timing = true;
        } 
        else {
            std::cerr << "Error: Unknown option " << arg << '\n';
            return false;
        }
    }
    return true;
}




int main(int argc, char* argv[]) {

    Options opts;
    if(!parse_arguments(argc, argv, opts)) {
        return 1;
    }
    
    Logger logger;
    auto trt_engine = std::make_unique<TrtEngine>(logger);
    
    if(opts.build) {
        if(opts.precision == "fp32") {
            trt_engine->BuildEngine("../models/yolov8n.onnx", "../models/yolov8n.engine", TrtPrecision::kFP32);
        } else if (opts.precision == "fp16"){
            trt_engine->BuildEngine("../models/yolov8n.onnx", "../models/yolov8n.engine", TrtPrecision::kFP16);
        } else if (opts.precision == "int8") {
            trt_engine->BuildEngine("../models/yolov8n.onnx", "../models/yolov8n.engine", TrtPrecision::kINT8);
        } else {
            std::cerr << "Precision not supported, only fp32, fp18 and int8 are supported" << std::endl;
        }
    }


    if(!opts.image_path.empty()) {
        trt_engine->LoadEngine("../models/yolov8n.engine");
        YOLOv8 yolo{std::move(trt_engine)};

        cv::Mat img = cv::imread(opts.image_path, cv::IMREAD_COLOR);

        auto begin = std::chrono::steady_clock::now();
        cv::Mat detection = yolo.Detect(img);
        auto end = std::chrono::steady_clock::now();
        
        if(opts.timing)
            std::cout << "Inference time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

        cv::imwrite("detection_output.jpg", detection);
    }


    else if(!opts.video_path.empty()){

        trt_engine->LoadEngine("../models/yolov8nfp16.engine");
        YOLOv8 yolo{std::move(trt_engine)};

        cv::VideoCapture cap(opts.video_path);

        if (!cap.isOpened()) {
            std::cout << "Cannot open the video file. :(\n";
            return -1;
        }

        int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        cv::VideoWriter video("outcpp.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, cv::Size(frame_width,frame_height));    
        cv::Mat frame;

        while(true){
            cap >> frame;

            if (frame.empty())
                break;

            auto begin = std::chrono::steady_clock::now();
            cv::Mat detection = yolo.Detect(frame);
            auto end = std::chrono::steady_clock::now();
        
            if(opts.timing)
                std::cout << "Inference time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
            
            video.write(detection);
        }
        
        cap.release();
        video.release();
    }
}
