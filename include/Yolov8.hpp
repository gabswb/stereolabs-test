# pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>

#include "TrtEngine.hpp"


class YOLOv8 {

public:
    YOLOv8(std::unique_ptr<TrtEngine> trt_engine);

    // Perform preprocessing, inference and postprocessing 
    cv::Mat Detect(cv::Mat& input_img);
    cv::Mat PreprocessImage(const cv::Mat& original_img);
    cv::Mat PostprocessImage(cv::Mat& output, const cv::Mat& original_img, const cv::Size& original_size);

protected:

    void DrawBBoxes(const cv::Mat& image, std::vector<cv::Rect> bboxes, std::vector<int> class_ids, std::vector<float> confidences);
    std::vector<cv::Rect> RescaleBoxes(const std::vector<cv::Rect>& boxes, const cv::Size& original_size, int input_size);

    size_t bbox_pred_dim_; //dimension of bounding box encoding
    size_t num_anchors_; 
    size_t input_size_; // input_size of the nn
    float score_treshold_; 
    float nms_treshold_;
    std::unique_ptr<TrtEngine> trt_engine_;

    static const std::array<std::string, 80> kClassNames;
    static const std::array<cv::Scalar, 80> kClassColors;
};
