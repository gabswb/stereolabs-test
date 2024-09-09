#include "Yolov10.hpp"

YOLOv10::YOLOv10(std::unique_ptr<TrtEngine> trt_engine) : conf_treshold_{0.2}, trt_engine_{std::move(trt_engine)} {
    std::vector<size_t> input_dim = trt_engine_->GetInputDim();
    assert(input_dim.size() == 4); // support only dim [batch_size, channels, height, width]
    assert(input_dim[0] == 1); // support only batch size of 1
    assert(input_dim[1] == 3); // support only 3 channels

    std::vector<size_t> output_dim = trt_engine_->GetOutputDim();
    assert(output_dim.size() == 3); // support only dim [batch_size, anchors, xyhw + conf + label]
    assert(output_dim[0] = 1); // support only batch_size of 1
    assert(output_dim[2] == 6); // support only xyhw + conf + label
    assert(input_dim[2] == input_dim[3]); // support only square images
    
    input_size_  = input_dim[2];
    n_anchors_ = output_dim[1];
}

cv::Mat YOLOv10::PreprocessImage(const cv::Mat& original_img) {
    
    // Convert BGR to RGB
    cv::Mat rgb_img;
    cv::cvtColor(original_img, rgb_img, cv::COLOR_BGR2RGB);

    // Resize the image
    cv::Mat resized_img;
    cv::resize(rgb_img, resized_img, cv::Size(input_size_, input_size_));

    // Normalize the image
    resized_img.convertTo(resized_img, CV_32FC3, 1.0 / 255.0);

    return cv::dnn::blobFromImage(resized_img);
}

std::vector<cv::Rect2f> YOLOv10::RescaleBoxes(const std::vector<cv::Rect2f>& boxes, const cv::Size& original_size, int input_size) {
    std::vector<cv::Rect2f> rescaled_boxes;

    // Scale factors
    float scale_x = static_cast<float>(original_size.width) / input_size;
    float scale_y = static_cast<float>(original_size.height) / input_size;

    // Rescale each box
    for (const auto& box : boxes) {
        cv::Rect2f rescaled_box;
        rescaled_box.x = box.x * scale_x;
        rescaled_box.y = box.y * scale_y;
        rescaled_box.width = box.width * scale_x;
        rescaled_box.height = box.height * scale_y;
        rescaled_boxes.push_back(rescaled_box);
    }

    return rescaled_boxes;
}

void YOLOv10::DrawBBoxes(const cv::Mat& image, std::vector<cv::Rect2f> bboxes, std::vector<int> class_ids, std::vector<float> confidences) {

    assert(bboxes.size() == class_ids.size());
    assert(bboxes.size() == confidences.size());

    for(size_t i=0; i < bboxes.size(); ++i) {
        std::stringstream stream;
        stream << std::fixed << std::setprecision(2) << confidences[i];
        std::string conf_str = stream.str();
        
        cv::rectangle(image, bboxes[i], kClassColors[class_ids[i]], 3);
        cv::putText(image, kClassNames[class_ids[i]] + " " + conf_str, cv::Point(bboxes[i].x, bboxes[i].y-10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 255, 255), 2);
    }
}


cv::Mat YOLOv10::PostprocessImage(const cv::Mat& output, const cv::Mat& original_img, const cv::Size& original_size) {

    std::vector<int> class_ids;
    std::vector<cv::Rect2f> boxes;
    std::vector<float> confidences;

    for (size_t i=0; i<output.rows; ++i) {

        float conf = output.at<float>(i, 4);
        if(conf < conf_treshold_) continue;

        float x1 = output.at<float>(i, 0);
        float y1 = output.at<float>(i, 1);
        float x2 = output.at<float>(i, 2);
        float y2 = output.at<float>(i, 3);
        int class_id = static_cast<int>(output.at<float>(i, 5));

        boxes.push_back(cv::Rect2f{cv::Point2f{x1,y1}, cv::Point2f{x2, y2}});
        class_ids.push_back(class_id);
        confidences.push_back(conf);
    }

    boxes = RescaleBoxes(boxes, original_size, input_size_);

    cv::Mat output_img{original_img};

    DrawBBoxes(output_img, boxes, class_ids, confidences);

    return output_img;
}



cv::Mat YOLOv10::Detect(cv::Mat& input_img) {

    cv::Size original_size = input_img.size();
    cv::Mat preprocessed_img = PreprocessImage(input_img);

    float *input_tensor, *output_tensor;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&input_tensor), sizeof(float) * 3 * input_size_ * input_size_));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&output_tensor), sizeof(float) * n_anchors_ * 6));

    CUDA_CHECK(cudaMemcpy(input_tensor, preprocessed_img.ptr<float>(0), sizeof(float) * 3 * input_size_ * input_size_, cudaMemcpyHostToDevice));

    trt_engine_->Inference(input_tensor, output_tensor);

    cv::Mat host_ouput_tensor{n_anchors_, 6, CV_32FC2};
    CUDA_CHECK(cudaMemcpy(host_ouput_tensor.ptr<float>(0), output_tensor, sizeof(float) * n_anchors_ * 6, cudaMemcpyDeviceToHost));

    cv::Mat postprocessed_image = PostprocessImage(host_ouput_tensor, input_img, original_size);

    CUDA_CHECK(cudaFree(input_tensor));
    CUDA_CHECK(cudaFree(output_tensor));

    return postprocessed_image;
}





const std::array<std::string, 80> YOLOv10::kClassNames = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"
};

const std::array<cv::Scalar, 80> YOLOv10::kClassColors = {
    cv::Scalar(0, 114, 189),   cv::Scalar(217, 83, 25),   cv::Scalar(237, 177, 32),  cv::Scalar(126, 47, 142),
    cv::Scalar(119, 172, 48),  cv::Scalar(77, 190, 238),  cv::Scalar(162, 20, 47),   cv::Scalar(76, 76, 76),
    cv::Scalar(153, 153, 153), cv::Scalar(255, 0, 0),     cv::Scalar(255, 128, 0),   cv::Scalar(191, 191, 0),
    cv::Scalar(0, 255, 0),     cv::Scalar(0, 0, 255),     cv::Scalar(170, 0, 255),   cv::Scalar(85, 85, 0),
    cv::Scalar(85, 170, 0),    cv::Scalar(85, 255, 0),    cv::Scalar(170, 85, 0),    cv::Scalar(170, 170, 0),
    cv::Scalar(170, 255, 0),   cv::Scalar(255, 85, 0),    cv::Scalar(255, 170, 0),   cv::Scalar(255, 255, 0),
    cv::Scalar(0, 85, 128),    cv::Scalar(0, 170, 128),   cv::Scalar(0, 255, 128),   cv::Scalar(85, 0, 128),
    cv::Scalar(85, 85, 128),   cv::Scalar(85, 170, 128),  cv::Scalar(85, 255, 128),  cv::Scalar(170, 0, 128),
    cv::Scalar(170, 85, 128),  cv::Scalar(170, 170, 128), cv::Scalar(170, 255, 128), cv::Scalar(255, 0, 128),
    cv::Scalar(255, 85, 128),  cv::Scalar(255, 170, 128), cv::Scalar(255, 255, 128), cv::Scalar(0, 85, 255),
    cv::Scalar(0, 170, 255),   cv::Scalar(0, 255, 255),   cv::Scalar(85, 0, 255),    cv::Scalar(85, 85, 255),
    cv::Scalar(85, 170, 255),  cv::Scalar(85, 255, 255),  cv::Scalar(170, 0, 255),   cv::Scalar(170, 85, 255),
    cv::Scalar(170, 170, 255), cv::Scalar(170, 255, 255), cv::Scalar(255, 0, 255),   cv::Scalar(255, 85, 255),
    cv::Scalar(255, 170, 255), cv::Scalar(85, 0, 0),      cv::Scalar(128, 0, 0),     cv::Scalar(170, 0, 0),
    cv::Scalar(212, 0, 0),     cv::Scalar(255, 0, 0),     cv::Scalar(0, 43, 0),      cv::Scalar(0, 85, 0),
    cv::Scalar(0, 128, 0),     cv::Scalar(0, 170, 0),     cv::Scalar(0, 212, 0),     cv::Scalar(0, 255, 0),
    cv::Scalar(0, 0, 43),      cv::Scalar(0, 0, 85),      cv::Scalar(0, 0, 128),     cv::Scalar(0, 0, 170),
    cv::Scalar(0, 0, 212),     cv::Scalar(0, 0, 255),     cv::Scalar(0, 0, 0),       cv::Scalar(36, 36, 36),
    cv::Scalar(73, 73, 73),    cv::Scalar(109, 109, 109), cv::Scalar(146, 146, 146), cv::Scalar(182, 182, 182),
    cv::Scalar(219, 219, 219), cv::Scalar(0, 114, 189),   cv::Scalar(80, 183, 189),  cv::Scalar(128, 128, 0)
};