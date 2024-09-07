# pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>

#include "TrtEngine.h"


class YOLOv10 {

public:
    YOLOv10(std::unique_ptr<TrtEngine> trt_engine);

    cv::Mat PreprocessImage(const cv::Mat& original_img);
    cv::Mat Detect(cv::Mat& input_img);
    cv::Mat PostprocessImage(const cv::Mat& output, const cv::Mat& original_img, const cv::Size& original_size);

protected:

    void DrawBBoxes(const cv::Mat& image, std::vector<cv::Rect2f> bboxes, std::vector<int> class_ids, std::vector<float> confidences);
    std::vector<cv::Rect2f> RescaleBoxes(const std::vector<cv::Rect2f>& boxes, const cv::Size& original_size, int input_size);

    size_t n_anchors_;
    size_t input_size_;
    float conf_treshold_;
    std::unique_ptr<TrtEngine> trt_engine_;

    static const std::array<std::string, 80> kClassNames;
    static const std::array<cv::Scalar, 80> kClassColors;
};
